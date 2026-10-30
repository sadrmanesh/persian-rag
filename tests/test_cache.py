from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import pytest

from persian_rag.cache import (
    CachedEmbedder,
    InMemoryEmbeddingCache,
    LRUCache,
    SQLiteEmbeddingCache,
    embedder_namespace,
)
from persian_rag.embeddings import HashingEmbedder
from persian_rag.pipeline import build_retriever
from persian_rag.retrieval import RetrievalResult


class CountingEmbedder:
    def __init__(self, dimension: int = 4) -> None:
        self._dimension = dimension
        self.documents: list[str] = []
        self.queries: list[str] = []

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        self.documents.extend(texts)
        return [[float(len(text))] * self._dimension for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [1.0] * self._dimension


def test_lru_cache_eviction_and_ttl() -> None:
    now = [0.0]
    cache: LRUCache[int] = LRUCache(max_size=2, ttl=10, clock=lambda: now[0])
    cache.set("a", 1)
    cache.set("b", 2)
    assert cache.get("a") == 1
    cache.set("c", 3)
    assert "b" not in cache
    assert len(cache) == 2
    now[0] = 11
    assert cache.get("a") is None
    assert cache.hits == 1 and cache.misses == 1
    cache.clear()
    assert len(cache) == 0
    with pytest.raises(ValueError):
        LRUCache(max_size=0)
    with pytest.raises(ValueError):
        LRUCache(ttl=0)


def test_cached_embedder_only_embeds_misses() -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner)
    first = cached.embed_documents(["سلام", "دنیا", "سلام"])
    assert inner.documents == ["سلام", "دنیا"]
    second = cached.embed_documents(["دنیا", "کتاب"])
    assert inner.documents == ["سلام", "دنیا", "کتاب"]
    assert first[1] == second[0]
    assert cached.dimension == 4
    cached.embed_query("سلام")
    cached.embed_query("سلام")
    assert inner.queries == ["سلام"]
    # Returned vectors are copies, so callers cannot corrupt the cache.
    first[0][0] = 99.0
    assert cached.embed_documents(["سلام"])[0][0] == 4.0


def test_namespace_tracks_embedder_settings() -> None:
    small = embedder_namespace(HashingEmbedder(dimension=32))
    large = embedder_namespace(HashingEmbedder(dimension=64))
    assert small != large
    assert "CountingEmbedder" in embedder_namespace(CountingEmbedder())
    cached = CachedEmbedder(HashingEmbedder(dimension=32))
    assert cached.to_spec().dimension == 32
    assert cached.key("x") != cached.key("x", "query")
    with pytest.raises(AttributeError):
        CachedEmbedder(CountingEmbedder()).to_spec()


def test_sqlite_cache_persists(tmp_path: Path) -> None:
    path = tmp_path / "cache" / "emb.sqlite"
    inner = CountingEmbedder(dimension=3)
    with SQLiteEmbeddingCache(path) as store:
        CachedEmbedder(inner, store).embed_documents(["الف", "ب"])
        assert len(store) == 2
    inner_again = CountingEmbedder(dimension=3)
    with SQLiteEmbeddingCache(path) as store:
        vectors = CachedEmbedder(inner_again, store).embed_documents(["الف", "ب"])
        assert inner_again.documents == []
        assert vectors == [[3.0] * 3, [1.0] * 3]
        assert store.get("missing") is None
        store.clear()
        assert len(store) == 0


def test_sqlite_cache_is_thread_safe(tmp_path: Path) -> None:
    store = SQLiteEmbeddingCache(tmp_path / "t.sqlite")
    errors: list[BaseException] = []

    def work(offset: int) -> None:
        try:
            for index in range(20):
                store.set(f"{offset}-{index}", [float(index)])
                assert store.get(f"{offset}-{index}") == [float(index)]
        except BaseException as exc:  # pragma: no cover - only on failure
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(store) == 80
    store.close()


def test_retriever_result_cache() -> None:
    inner = CountingEmbedder(dimension=8)
    retriever = build_retriever(["حافظ در شیراز", "فردوسی در توس"], inner)
    cache: LRUCache[list[RetrievalResult]] = LRUCache(max_size=8)
    retriever.result_cache = cache
    first = retriever.retrieve("حافظ", k=1)
    second = retriever.retrieve("حافظ", k=1)
    assert first == second
    assert len(inner.queries) == 1
    assert cache.hits == 1
    retriever.retrieve("حافظ", k=1, mode="sparse")
    assert len(cache) == 2
    from persian_rag.chunk import chunk_text

    retriever.add_chunks(chunk_text("سعدی در شیراز", document_id="saadi"))
    assert len(cache) == 0


def test_in_memory_embedding_cache_defaults() -> None:
    cache = InMemoryEmbeddingCache()
    assert cache.max_size == 100_000
    cache.set("k", [1.0])
    assert cache.get("k") == [1.0]
