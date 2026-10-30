"""Vector stores.

:class:`InMemoryVectorStore` is pure Python. :class:`FaissVectorStore` is an
optional adapter around ``faiss-cpu`` and is add-only.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

FaissMetric = Literal["cosine", "l2"]


@dataclass(frozen=True, slots=True)
class VectorHit:
    """One nearest-neighbor hit. Larger scores are closer."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorStore(Protocol):
    """Minimal dense index used by the hybrid retriever."""

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Add vectors. Implementations reject duplicate ids."""

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        """Return up to ``k`` hits, best score first."""

    def __len__(self) -> int:
        """Number of stored vectors."""


@dataclass(slots=True)
class _MemoryRecord:
    id: str
    vector: tuple[float, ...]
    norm: float
    text: str
    metadata: dict[str, Any]


class InMemoryVectorStore:
    """Cosine-similarity index kept in process memory."""

    def __init__(self) -> None:
        self._records: list[_MemoryRecord] = []
        self._positions: dict[str, int] = {}
        self._dimension: int | None = None

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        _check_parallel(ids, vectors, texts, metadatas)
        if not ids:
            return
        metas = _metadata_rows(metadatas, len(ids))
        width = _shared_dimension(vectors)
        if self._dimension is None:
            self._dimension = width
        elif width != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {width}")

        for doc_id, vector, text, meta in zip(ids, vectors, texts, metas, strict=True):
            if doc_id in self._positions:
                raise ValueError(f"Duplicate vector id: {doc_id}")
            values = tuple(float(item) for item in vector)
            self._positions[doc_id] = len(self._records)
            self._records.append(
                _MemoryRecord(
                    id=doc_id,
                    vector=values,
                    norm=_norm(values),
                    text=text,
                    metadata=meta,
                )
            )

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._records:
            return []
        if self._dimension is not None and len(vector) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(vector)}")

        query = tuple(float(item) for item in vector)
        query_norm = _norm(query)
        scored: list[tuple[float, _MemoryRecord]] = []
        for record in self._records:
            if query_norm == 0.0 or record.norm == 0.0:
                score = 0.0
            else:
                score = _dot(query, record.vector) / (query_norm * record.norm)
            scored.append((score, record))
        scored.sort(key=lambda item: (-item[0], item[1].id))
        return [
            VectorHit(
                id=record.id,
                score=score,
                text=record.text,
                metadata=dict(record.metadata),
            )
            for score, record in scored[:k]
        ]

    def __len__(self) -> int:
        return len(self._records)


class FaissVectorStore:
    """FAISS flat index.

    ``cosine`` stores L2-normalized vectors and searches with inner product.
    ``l2`` returns the negative Euclidean distance so larger scores are closer.
    Deletion is not supported. Install ``persian-rag[faiss]`` before use.
    """

    def __init__(self, dimension: int, *, metric: FaissMetric = "cosine") -> None:
        if dimension < 1:
            raise ValueError("dimension must be >= 1")
        if metric not in ("cosine", "l2"):
            raise ValueError("metric must be 'cosine' or 'l2'")
        faiss = _import_faiss()
        self.dimension = dimension
        self.metric = metric
        if metric == "cosine":
            self._index = faiss.IndexFlatIP(dimension)
        else:
            self._index = faiss.IndexFlatL2(dimension)
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._known: set[str] = set()

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        _check_parallel(ids, vectors, texts, metadatas)
        if not ids:
            return
        width = _shared_dimension(vectors)
        if width != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {width}")
        for doc_id in ids:
            if doc_id in self._known:
                raise ValueError(f"Duplicate vector id: {doc_id}")

        faiss, np = _import_faiss_and_numpy()
        matrix = np.asarray(vectors, dtype="float32")
        if self.metric == "cosine":
            faiss.normalize_L2(matrix)
        self._index.add(matrix)
        metas = _metadata_rows(metadatas, len(ids))
        for doc_id, text, meta in zip(ids, texts, metas, strict=True):
            self._known.add(doc_id)
            self._ids.append(doc_id)
            self._texts.append(text)
            self._metadatas.append(meta)

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._ids:
            return []
        if len(vector) != self.dimension:
            raise ValueError(f"Expected dimension {self.dimension}, got {len(vector)}")

        faiss, np = _import_faiss_and_numpy()
        query = np.asarray([list(vector)], dtype="float32")
        if self.metric == "cosine":
            faiss.normalize_L2(query)
        limit = min(k, len(self._ids))
        scores, indices = self._index.search(query, limit)
        hits: list[VectorHit] = []
        for score, raw_index in zip(scores[0], indices[0], strict=True):
            position = int(raw_index)
            if position < 0:
                continue
            distance = float(score)
            if self.metric == "l2":
                distance = -distance
            hits.append(
                VectorHit(
                    id=self._ids[position],
                    score=distance,
                    text=self._texts[position],
                    metadata=dict(self._metadatas[position]),
                )
            )
        return hits

    def __len__(self) -> int:
        return len(self._ids)


def _import_faiss() -> Any:
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "faiss is not installed. Install it with: pip install 'persian-rag[faiss]'"
        ) from exc
    return faiss


def _import_faiss_and_numpy() -> tuple[Any, Any]:
    faiss = _import_faiss()
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "numpy is not installed. Install it with: pip install 'persian-rag[faiss]'"
        ) from exc
    return faiss, np


def _check_parallel(
    ids: Sequence[str],
    vectors: Sequence[Sequence[float]],
    texts: Sequence[str],
    metadatas: Sequence[Mapping[str, Any]] | None,
) -> None:
    if not (len(ids) == len(vectors) == len(texts)):
        raise ValueError("ids, vectors, and texts must have the same length")
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("metadatas length must match ids")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")


def _metadata_rows(
    metadatas: Sequence[Mapping[str, Any]] | None,
    count: int,
) -> list[dict[str, Any]]:
    if metadatas is None:
        return [{} for _ in range(count)]
    return [dict(item) for item in metadatas]


def _shared_dimension(vectors: Sequence[Sequence[float]]) -> int:
    width = len(vectors[0])
    if width < 1:
        raise ValueError("vectors must be non-empty")
    for vector in vectors:
        if len(vector) != width:
            raise ValueError("inconsistent vector dimensions")
    return width


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values))
