"""Hybrid retrieval: BM25, dense search, and reciprocal rank fusion."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from persian_rag.bm25 import BM25Index
from persian_rag.chunk import Chunk
from persian_rag.embeddings import Embedder
from persian_rag.normalize import NormalizeConfig, normalize
from persian_rag.store import InMemoryVectorStore, VectorStore
from persian_rag.tokenize import tokenize

RetrievalMode = Literal["hybrid", "dense", "sparse"]


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

    def __len__(self) -> int:
        return len(self._order)

    @property
    def chunks(self) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in self._order]

    def embeddings(self) -> list[list[float]]:
        return [list(self._embeddings[chunk_id]) for chunk_id in self._order]

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

        self.store.add(
            ids,
            vectors,
            [chunk.text for chunk in chunks],
            [chunk.metadata for chunk in chunks],
        )
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
        if not self._chunks:
            return []

        normalized = normalize(query, self.normalize_config)
        pool = min(len(self._chunks), max(k, k * self.candidate_multiplier))
        dense_ids: list[str] = []
        dense_scores: dict[str, float] = {}
        sparse_ids: list[str] = []
        sparse_scores: dict[str, float] = {}

        if mode in ("hybrid", "dense"):
            vector = self.embedder.embed_query(normalized)
            for hit in self.store.search(vector, k=pool):
                dense_ids.append(hit.id)
                dense_scores[hit.id] = hit.score
        if mode in ("hybrid", "sparse"):
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
        for doc_id, score in fused[:k]:
            chunk = self._chunks[doc_id]
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
        return results
