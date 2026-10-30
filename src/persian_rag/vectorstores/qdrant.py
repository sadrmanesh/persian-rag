"""Qdrant vector store adapter.

Install with ``pip install 'persian-rag[qdrant]'``. Works with a Qdrant server,
Qdrant Cloud, a local on-disk database, or the in-process ``:memory:`` mode,
which is handy for tests.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from persian_rag.store import VectorHit, _check_parallel, _metadata_rows, _shared_dimension

QdrantDistance = Literal["cosine", "dot", "euclid"]

#: Namespace used to derive stable Qdrant point ids from string ids.
POINT_NAMESPACE = uuid.UUID("6f1c3c38-56a5-4c55-9f0b-6b1c1b6f9a10")
_ID_KEY = "_pr_id"
_TEXT_KEY = "_pr_text"
_META_KEY = "_pr_metadata"


def point_id(doc_id: str) -> str:
    """Map an arbitrary string id to the UUID Qdrant stores."""

    return str(uuid.uuid5(POINT_NAMESPACE, doc_id))


def _import_qdrant() -> tuple[Any, Any]:
    try:
        from qdrant_client import QdrantClient, models
    except ImportError as exc:
        raise ImportError(
            "qdrant-client is not installed. Install it with: pip install 'persian-rag[qdrant]'"
        ) from exc
    return QdrantClient, models


class QdrantVectorStore:
    """Store chunks in a Qdrant collection.

    String ids are mapped to deterministic UUIDs; the original id, text, and
    metadata are kept in the payload. The collection is created on first use
    when ``create_collection`` is true. Scores follow Qdrant: larger is closer
    for every distance.
    """

    def __init__(
        self,
        collection_name: str = "persian_rag",
        dimension: int | None = None,
        *,
        client: Any | None = None,
        url: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
        distance: QdrantDistance = "cosine",
        create_collection: bool = True,
    ) -> None:
        if distance not in ("cosine", "dot", "euclid"):
            raise ValueError("distance must be 'cosine', 'dot', or 'euclid'")
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be >= 1")
        client_cls, models = _import_qdrant()
        self._models = models
        if client is None:
            if url is not None:
                client = client_cls(url=url, api_key=api_key)
            elif path is not None:
                client = client_cls(path=path)
            else:
                client = client_cls(location=":memory:")
        self.client = client
        self.collection_name = collection_name
        self.distance = distance
        self.create_collection = create_collection
        self._dimension = dimension
        if self.client.collection_exists(collection_name):
            self._dimension = self._collection_dimension()
            if dimension is not None and self._dimension not in (None, dimension):
                raise ValueError(
                    f"Collection {collection_name!r} has dimension {self._dimension}, "
                    f"expected {dimension}"
                )
        elif dimension is not None and create_collection:
            self._create(dimension)

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def _distance(self) -> Any:
        mapping = {
            "cosine": self._models.Distance.COSINE,
            "dot": self._models.Distance.DOT,
            "euclid": self._models.Distance.EUCLID,
        }
        return mapping[self.distance]

    def _create(self, dimension: int) -> None:
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=self._models.VectorParams(size=dimension, distance=self._distance()),
        )
        self._dimension = dimension

    def _collection_dimension(self) -> int | None:
        info = self.client.get_collection(self.collection_name)
        vectors = info.config.params.vectors
        size = getattr(vectors, "size", None)
        return int(size) if size is not None else None

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
        if self._dimension is None:
            if not self.create_collection:
                raise ValueError(f"Collection {self.collection_name!r} does not exist")
            self._create(width)
        elif width != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {width}")

        existing = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[point_id(doc_id) for doc_id in ids],
            with_payload=True,
            with_vectors=False,
        )
        if existing:
            taken = sorted(str((point.payload or {}).get(_ID_KEY, point.id)) for point in existing)
            raise ValueError(f"Duplicate vector id: {', '.join(taken)}")

        metas = _metadata_rows(metadatas, len(ids))
        points = [
            self._models.PointStruct(
                id=point_id(doc_id),
                vector=[float(value) for value in vector],
                payload={_ID_KEY: doc_id, _TEXT_KEY: text, _META_KEY: meta},
            )
            for doc_id, vector, text, meta in zip(ids, vectors, texts, metas, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        if k < 1:
            raise ValueError("k must be >= 1")
        if self._dimension is None:
            return []
        if len(vector) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(vector)}")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in vector],
            limit=k,
            with_payload=True,
        )
        hits: list[VectorHit] = []
        for point in response.points:
            payload = point.payload or {}
            metadata = payload.get(_META_KEY) or {}
            hits.append(
                VectorHit(
                    id=str(payload.get(_ID_KEY, point.id)),
                    score=float(point.score),
                    text=str(payload.get(_TEXT_KEY, "")),
                    metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
                )
            )
        return hits

    def delete(self, ids: Sequence[str]) -> None:
        """Remove points by their original string ids. Unknown ids are ignored."""

        if not ids or self._dimension is None:
            return
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=self._models.PointIdsList(points=[point_id(i) for i in ids]),
            wait=True,
        )

    def __len__(self) -> int:
        if self._dimension is None:
            return 0
        return int(self.client.count(collection_name=self.collection_name, exact=True).count)
