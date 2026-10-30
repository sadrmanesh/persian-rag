"""Hybrid retrieval: BM25, dense search, and reciprocal rank fusion."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from persian_rag.bm25 import BM25Index
from persian_rag.cache import LRUCache, result_cache_key
from persian_rag.chunk import Chunk
from persian_rag.embeddings import Embedder
from persian_rag.normalize import NormalizeConfig, normalize
from persian_rag.store import InMemoryVectorStore, VectorHit, VectorStore
from persian_rag.tokenize import tokenize

RetrievalMode = Literal["hybrid", "dense", "sparse"]


class SparseBackend(Protocol):
    """A server-side lexical index, such as Elasticsearch or OpenSearch BM25.

    ``search_text`` receives the normalized query and returns hits with the
    best score first.
    """

    def search_text(self, query: str, k: int = 5) -> list[VectorHit]:
        """Return up to ``k`` lexical hits for ``query``."""


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A chunk selected for a query."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    start: int
    end: int
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_rank: int | None = None
    sparse_rank: int | None = None
    retrieval_score: float | None = None
    """First-stage score, set when a reranker replaced :attr:`score`."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
            "dense_rank": self.dense_rank,
            "sparse_rank": self.sparse_rank,
            "retrieval_score": self.retrieval_score,
        }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = 60,
    weights: Sequence[float] | None = None,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists with reciprocal rank fusion.

    ``k`` is the RRF constant (commonly 60), not the number of results.
    Ranks are 1-based. Duplicate ids inside a single list count once, at
    their first rank. Ties are broken by id so the order is stable.
    """

    if k < 0:
        raise ValueError("k must be >= 0")
    if not rankings:
        return []
    applied = list(weights) if weights is not None else [1.0] * len(rankings)
    if len(applied) != len(rankings):
        raise ValueError("weights must have one value per ranking")
    if any(weight < 0 for weight in applied):
        raise ValueError("weights must be >= 0")

    scores: dict[str, float] = {}
    for weight, ranking in zip(applied, rankings, strict=True):
        seen: set[str] = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + weight / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridRetriever:
    """Retrieve chunks with dense vectors, BM25, or both.

    Document text should already be normalized with ``normalize_config``
    when it is added. Queries are normalized with the same config.
    Hybrid mode ranks with reciprocal rank fusion. Dense-only mode returns
    cosine similarity (or the store's native score). Sparse-only mode
    returns BM25 scores.

    Pass ``result_cache`` (for example ``LRUCache(max_size=512, ttl=300)``)
    to reuse results for repeated queries. The cache is cleared whenever
    chunks are added or deleted.

    Pass ``sparse_backend`` (for example an
    :class:`~persian_rag.vectorstores.ElasticsearchStore`) to run the lexical
    side on a search server instead of the in-process BM25 index. Hits for
    chunks that are not held locally are rebuilt from the server payload, so
    a retriever can query an index that another process filled.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
        *,
        normalize_config: NormalizeConfig | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        candidate_multiplier: int = 4,
        result_cache: LRUCache[list[RetrievalResult]] | None = None,
        sparse_backend: SparseBackend | None = None,
    ) -> None:
        if rrf_k < 0:
            raise ValueError("rrf_k must be >= 0")
        if dense_weight < 0 or sparse_weight < 0:
            raise ValueError("retrieval weights must be >= 0")
        if dense_weight == 0 and sparse_weight == 0:
            raise ValueError("at least one retrieval weight must be > 0")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        self.embedder = embedder
        self.store = store if store is not None else InMemoryVectorStore()
        self.normalize_config = normalize_config or NormalizeConfig()
        self.bm25 = BM25Index(k1=bm25_k1, b=bm25_b)
        self.rrf_k = rrf_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.candidate_multiplier = candidate_multiplier
        self._chunks: dict[str, Chunk] = {}
        self._embeddings: dict[str, list[float]] = {}
        self._order: list[str] = []
        self.result_cache = result_cache
        self.sparse_backend = sparse_backend

    def __len__(self) -> int:
        return len(self._order)

    @property
    def chunks(self) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in self._order]

    def embeddings(self) -> list[list[float]]:
        return [list(self._embeddings[chunk_id]) for chunk_id in self._order]

    def __contains__(self, chunk_id: object) -> bool:
        return chunk_id in self._chunks

    def chunk_ids_for_document(self, document_id: str) -> list[str]:
        """Ids of the indexed chunks that belong to ``document_id``, in order."""

        return [
            chunk_id
            for chunk_id in self._order
            if self._chunks[chunk_id].document_id == document_id
        ]

    def document_ids(self) -> list[str]:
        """Distinct document ids in insertion order."""

        return list(dict.fromkeys(self._chunks[chunk_id].document_id for chunk_id in self._order))

    def delete_chunks(self, chunk_ids: Sequence[str]) -> int:
        """Remove chunks from the store and the BM25 index.

        Returns the number of chunks removed. Unknown ids are ignored. The
        vector store must provide ``delete(ids)``; the in-memory, Qdrant,
        pgvector, Elasticsearch, and OpenSearch stores do, FAISS does not.
        """

        doomed = [chunk_id for chunk_id in dict.fromkeys(chunk_ids) if chunk_id in self._chunks]
        if not doomed:
            return 0
        delete = getattr(self.store, "delete", None)
        if not callable(delete):
            raise TypeError(f"{type(self.store).__name__} does not support deletion")
        delete(doomed)
        self.bm25.delete(doomed)
        gone = set(doomed)
        for chunk_id in doomed:
            del self._chunks[chunk_id]
            del self._embeddings[chunk_id]
        self._order = [chunk_id for chunk_id in self._order if chunk_id not in gone]
        if self.result_cache is not None:
            self.result_cache.clear()
        return len(doomed)

    def delete_document(self, document_id: str) -> int:
        """Remove every chunk of ``document_id`` and return how many were removed."""

        return self.delete_chunks(self.chunk_ids_for_document(document_id))

    def add_chunks(
        self,
        chunks: Sequence[Chunk],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Index chunks, embedding them unless vectors are already provided."""

        if not chunks:
            return
        ids = [chunk.id for chunk in chunks]
        if len(set(ids)) != len(ids):
            raise ValueError("chunk ids must be unique")
        for chunk_id in ids:
            if chunk_id in self._chunks:
                raise ValueError(f"chunk id already indexed: {chunk_id}")

        if embeddings is None:
            vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        else:
            if len(embeddings) != len(chunks):
                raise ValueError("embeddings and chunks must have the same length")
            vectors = [[float(value) for value in vector] for vector in embeddings]
        if len(vectors) != len(chunks):
            raise ValueError("embedder returned an unexpected number of vectors")
        width = self.embedder.dimension
        for vector in vectors:
            if len(vector) != width:
                raise ValueError(f"Expected embedding dimension {width}, got {len(vector)}")

        if getattr(self.store, "stores_chunk_fields", False):
            # Stores that can be queried by other processes keep the chunk
            # position so results can be rebuilt without the local index.
            metadatas = [_with_chunk_fields(chunk) for chunk in chunks]
        else:
            metadatas = [chunk.metadata for chunk in chunks]
        self.store.add(ids, vectors, [chunk.text for chunk in chunks], metadatas)
        self.bm25.add(
            ids,
            [
                tokenize(chunk.text, normalize_text=True, config=self.normalize_config)
                for chunk in chunks
            ],
        )
        for chunk, vector in zip(chunks, vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._embeddings[chunk.id] = list(vector)
            self._order.append(chunk.id)
        if self.result_cache is not None:
            self.result_cache.clear()

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        """Return the top ``k`` chunks for ``query``."""

        if k < 1:
            raise ValueError("k must be >= 1")
        if mode not in ("hybrid", "dense", "sparse"):
            raise ValueError("mode must be 'hybrid', 'dense', or 'sparse'")
        if not self._chunks and self.sparse_backend is None:
            return []

        normalized = normalize(query, self.normalize_config)
        cache_key = None
        if self.result_cache is not None:
            cache_key = result_cache_key(normalized, k, mode)
            cached = self.result_cache.get(cache_key)
            if cached is not None:
                return list(cached)
        pool = max(k, k * self.candidate_multiplier)
        if self.sparse_backend is None:
            pool = min(len(self._chunks), pool)
        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        sparse_ids: list[str] = []
        sparse_scores: dict[str, float] = {}
        remote: dict[str, VectorHit] = {}

        if mode in ("hybrid", "dense") and len(self.store) > 0:
            vector = self.embedder.embed_query(normalized)
            for hit in self.store.search(vector, k=pool):
                dense_ids.append(hit.id)
                dense_scores[hit.id] = hit.score
                remote.setdefault(hit.id, hit)
        if mode in ("hybrid", "sparse") and self.sparse_backend is not None:
            for hit in self.sparse_backend.search_text(normalized, k=pool):
                sparse_ids.append(hit.id)
                sparse_scores[hit.id] = hit.score
                remote.setdefault(hit.id, hit)
        elif mode in ("hybrid", "sparse"):
            tokens = tokenize(normalized, normalize_text=False, config=self.normalize_config)
            for doc_id, score in self.bm25.search(tokens, k=pool):
                sparse_ids.append(doc_id)
                sparse_scores[doc_id] = score

        if mode == "dense":
            fused = [(doc_id, dense_scores[doc_id]) for doc_id in dense_ids]
        elif mode == "sparse":
            fused = [(doc_id, sparse_scores[doc_id]) for doc_id in sparse_ids]
        else:
            fused = reciprocal_rank_fusion(
                [dense_ids, sparse_ids],
                k=self.rrf_k,
                weights=(self.dense_weight, self.sparse_weight),
            )

        dense_ranks = {doc_id: rank for rank, doc_id in enumerate(dense_ids, start=1)}
        sparse_ranks = {doc_id: rank for rank, doc_id in enumerate(sparse_ids, start=1)}
        results: list[RetrievalResult] = []
        for doc_id, score in fused:
            if len(results) == k:
                break
            chunk = self._chunks.get(doc_id) or _chunk_from_hit(remote.get(doc_id))
            if chunk is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    text=chunk.text,
                    score=score,
                    start=chunk.start,
                    end=chunk.end,
                    metadata=dict(chunk.metadata),
                    dense_rank=dense_ranks.get(doc_id),
                    sparse_rank=sparse_ranks.get(doc_id),
                )
            )
        if self.result_cache is not None and cache_key is not None:
            self.result_cache.set(cache_key, list(results))
        return results

    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        """Async variant of :meth:`retrieve`; runs in a worker thread."""

        return await asyncio.to_thread(self.retrieve, query, k, mode=mode)


def _with_chunk_fields(chunk: Chunk) -> dict[str, Any]:
    return {
        **chunk.metadata,
        "_document_id": chunk.document_id,
        "_start": chunk.start,
        "_end": chunk.end,
        "_index": chunk.index,
    }


def _chunk_from_hit(hit: VectorHit | None) -> Chunk | None:
    """Rebuild a chunk from a store payload when it is not held in memory."""

    if hit is None:
        return None
    metadata = dict(hit.metadata)
    document_id = str(metadata.pop("_document_id", "") or metadata.get("source", "") or "")
    start = _int_or(metadata.pop("_start", 0), 0)
    end = _int_or(metadata.pop("_end", start + len(hit.text)), start + len(hit.text))
    index = _int_or(metadata.pop("_index", 0), 0)
    return Chunk(
        id=hit.id,
        text=hit.text,
        start=start,
        end=end,
        index=index,
        document_id=document_id,
        metadata=metadata,
    )


def _int_or(value: object, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return default
