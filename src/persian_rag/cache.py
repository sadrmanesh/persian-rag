"""Embedding and retrieval caches.

* :class:`CachedEmbedder` wraps any :class:`~persian_rag.embeddings.Embedder`
  and reuses vectors for texts it has already embedded.
* :class:`LRUCache` is a small thread-safe in-memory cache with optional
  expiry. :class:`HybridRetriever` uses it for query results.
* :class:`SQLiteEmbeddingCache` persists vectors on disk with only the
  standard library, so repeated ingestion runs skip paid embedding calls.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from persian_rag.embeddings import Embedder, EmbedderSpec

V = TypeVar("V")


class LRUCache(Generic[V]):
    """Thread-safe least-recently-used cache with an optional time to live.

    ``ttl`` is in seconds; ``None`` keeps entries until they are evicted.
    """

    def __init__(
        self,
        max_size: int = 1024,
        ttl: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        if ttl is not None and ttl <= 0:
            raise ValueError("ttl must be > 0")
        self.max_size = max_size
        self.ttl = ttl
        self._clock = clock
        self._data: OrderedDict[str, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> V | None:
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                self.misses += 1
                return None
            stored_at, value = entry
            if self.ttl is not None and self._clock() - stored_at > self.ttl:
                del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: V) -> None:
        with self._lock:
            self._data[key] = (self._clock(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_size:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data


@runtime_checkable
class EmbeddingCache(Protocol):
    """Key-value storage for embedding vectors."""

    def get(self, key: str) -> list[float] | None: ...

    def set(self, key: str, value: list[float]) -> None: ...


class InMemoryEmbeddingCache(LRUCache[list[float]]):
    """In-process embedding cache. Defaults to 100,000 vectors."""

    def __init__(self, max_size: int = 100_000, ttl: float | None = None) -> None:
        super().__init__(max_size=max_size, ttl=ttl)


class SQLiteEmbeddingCache:
    """Persistent embedding cache in a single SQLite file.

    Vectors are stored as packed float64 values. The connection is shared
    between threads and guarded by a lock.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            self._conn.commit()

    def get(self, key: str) -> list[float] | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT vector FROM embeddings WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        blob = bytes(row[0])
        return list(struct.unpack(f"<{len(blob) // 8}d", blob))

    def set(self, key: str, value: list[float]) -> None:
        blob = struct.pack(f"<{len(value)}d", *value)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)", (key, blob)
            )
            self._conn.commit()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT count(*) FROM embeddings").fetchone()
        return int(row[0])

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM embeddings")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SQLiteEmbeddingCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def embedder_namespace(embedder: Embedder) -> str:
    """A stable cache namespace for an embedder.

    Uses the embedder's :class:`EmbedderSpec` when it has one, so changing the
    model, dimension, or hashing settings never returns stale vectors.
    """

    to_spec = getattr(embedder, "to_spec", None)
    if callable(to_spec):
        spec = to_spec()
        if isinstance(spec, EmbedderSpec):
            return json.dumps(spec.to_dict(), sort_keys=True, ensure_ascii=False)
    return f"{type(embedder).__module__}.{type(embedder).__qualname__}:{embedder.dimension}"


class CachedEmbedder:
    """Wrap an embedder and cache its vectors.

    Document and query vectors are cached separately because some models
    (for example E5) embed them differently. Only missing texts are sent to
    the wrapped embedder, in one batch.
    """

    def __init__(
        self,
        embedder: Embedder,
        cache: EmbeddingCache | None = None,
        *,
        namespace: str | None = None,
    ) -> None:
        self.embedder = embedder
        self.cache: EmbeddingCache = cache if cache is not None else InMemoryEmbeddingCache()
        self.namespace = namespace if namespace is not None else embedder_namespace(embedder)

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    def to_spec(self) -> EmbedderSpec:
        """Return the wrapped embedder's spec so indexes stay portable."""

        to_spec = getattr(self.embedder, "to_spec", None)
        if not callable(to_spec):
            raise AttributeError(f"{type(self.embedder).__name__} has no to_spec()")
        spec = to_spec()
        if not isinstance(spec, EmbedderSpec):
            raise TypeError("to_spec() must return an EmbedderSpec")
        return spec

    def key(self, text: str, kind: str = "doc") -> str:
        digest = hashlib.sha256()
        for part in (self.namespace, kind, text):
            digest.update(part.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        keys = [self.key(text) for text in texts]
        found: dict[int, list[float]] = {}
        missing: dict[str, list[int]] = {}
        for index, (text, key) in enumerate(zip(texts, keys, strict=True)):
            cached = self.cache.get(key)
            if cached is not None:
                found[index] = list(cached)
            else:
                missing.setdefault(text, []).append(index)
        if missing:
            todo = list(missing)
            vectors = self.embedder.embed_documents(todo)
            if len(vectors) != len(todo):
                raise ValueError("embedder returned an unexpected number of vectors")
            for text, vector in zip(todo, vectors, strict=True):
                values = [float(value) for value in vector]
                self.cache.set(self.key(text), values)
                for index in missing[text]:
                    found[index] = list(values)
        return [found[index] for index in range(len(texts))]

    def embed_query(self, text: str) -> list[float]:
        key = self.key(text, "query")
        cached = self.cache.get(key)
        if cached is not None:
            return list(cached)
        vector = [float(value) for value in self.embedder.embed_query(text)]
        self.cache.set(key, vector)
        return vector


def result_cache_key(*parts: Any) -> str:
    """Build a cache key from JSON-serializable parts."""

    return json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
