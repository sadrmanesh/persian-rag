"""Vector store backends.

The in-memory and FAISS stores live in :mod:`persian_rag.store`. Server-backed
stores are imported lazily so their client libraries stay optional:

* :class:`QdrantVectorStore` needs ``persian-rag[qdrant]``.
* :class:`PgVectorStore` needs ``persian-rag[pgvector]``.
* :class:`ElasticsearchStore` and :class:`OpenSearchStore` use the REST API
  through the standard library and need no extra package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from persian_rag.store import FaissVectorStore, InMemoryVectorStore, VectorHit, VectorStore

if TYPE_CHECKING:
    from persian_rag.vectorstores.elasticsearch import ElasticsearchStore, OpenSearchStore
    from persian_rag.vectorstores.pgvector import PgVectorStore
    from persian_rag.vectorstores.qdrant import QdrantVectorStore

__all__ = [
    "ElasticsearchStore",
    "FaissVectorStore",
    "InMemoryVectorStore",
    "OpenSearchStore",
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
    if name in ("ElasticsearchStore", "OpenSearchStore"):
        from persian_rag.vectorstores import elasticsearch

        return getattr(elasticsearch, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
