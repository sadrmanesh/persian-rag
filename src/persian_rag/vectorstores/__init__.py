"""Vector store backends.

The in-memory and FAISS stores live in :mod:`persian_rag.store`. Server-backed
stores are imported lazily so their client libraries stay optional:

* :class:`QdrantVectorStore` needs ``persian-rag[qdrant]``.
* :class:`PgVectorStore` needs ``persian-rag[pgvector]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from persian_rag.store import FaissVectorStore, InMemoryVectorStore, VectorHit, VectorStore

if TYPE_CHECKING:
    from persian_rag.vectorstores.pgvector import PgVectorStore
    from persian_rag.vectorstores.qdrant import QdrantVectorStore

__all__ = [
    "FaissVectorStore",
    "InMemoryVectorStore",
    "PgVectorStore",
    "QdrantVectorStore",
    "VectorHit",
    "VectorStore",
]


def __getattr__(name: str) -> Any:
    if name == "QdrantVectorStore":
        from persian_rag.vectorstores.qdrant import QdrantVectorStore

        return QdrantVectorStore
    if name == "PgVectorStore":
        from persian_rag.vectorstores.pgvector import PgVectorStore

        return PgVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
