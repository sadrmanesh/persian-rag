from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Sequence
from typing import Any

import pytest

from persian_rag.embeddings import HashingEmbedder
from persian_rag.pipeline import build_retriever
from persian_rag.vectorstores import PgVectorStore
from persian_rag.vectorstores.pgvector import vector_literal


def test_lazy_exports() -> None:
    import persian_rag.vectorstores as stores

    assert stores.PgVectorStore is PgVectorStore
    with pytest.raises(AttributeError):
        _ = stores.NoSuchStore  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- Qdrant


def test_qdrant_in_memory_roundtrip() -> None:
    pytest.importorskip("qdrant_client")
    from persian_rag.vectorstores import QdrantVectorStore

    store = QdrantVectorStore("docs", dimension=3)
    assert len(store) == 0
    store.add(
        ["a:0", "b:0", "c:0"],
        [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]],
        ["الف", "ب", "پ"],
        [{"lang": "fa"}, {}, {"n": 3}],
    )
    assert len(store) == 3
    hits = store.search([1, 0, 0], k=2)
    assert [hit.id for hit in hits] == ["a:0", "c:0"]
    assert hits[0].text == "الف"
    assert hits[0].metadata == {"lang": "fa"}
    assert hits[0].score >= hits[1].score
    with pytest.raises(ValueError, match="Duplicate"):
        store.add(["a:0"], [[1, 0, 0]], ["x"])
    with pytest.raises(ValueError):
        store.add(["z"], [[1, 0]], ["x"])
    with pytest.raises(ValueError):
        store.search([1, 0], k=1)
    with pytest.raises(ValueError):
        store.search([1, 0, 0], k=0)
    store.delete(["a:0", "missing"])
    assert len(store) == 2


def test_qdrant_lazy_collection_and_reuse() -> None:
    pytest.importorskip("qdrant_client")
    from qdrant_client import QdrantClient

    from persian_rag.vectorstores import QdrantVectorStore

    client = QdrantClient(location=":memory:")
    store = QdrantVectorStore("lazy", client=client)
    assert store.dimension is None
    assert store.search([1.0], k=1) == []
    assert len(store) == 0
    store.delete(["x"])
    store.add(["x"], [[0.0, 1.0]], ["متن"])
    assert store.dimension == 2
    reopened = QdrantVectorStore("lazy", client=client)
    assert reopened.dimension == 2
    assert reopened.search([0.0, 1.0], k=1)[0].id == "x"
    with pytest.raises(ValueError):
        QdrantVectorStore("lazy", dimension=5, client=client)
    with pytest.raises(ValueError):
        QdrantVectorStore("other", client=client, create_collection=False).add(
            ["y"], [[1.0]], ["t"]
        )
    with pytest.raises(ValueError):
        QdrantVectorStore("bad", distance="manhattan", client=client)  # type: ignore[arg-type]


def test_qdrant_with_hybrid_retriever() -> None:
    pytest.importorskip("qdrant_client")
    from persian_rag.vectorstores import QdrantVectorStore

    retriever = build_retriever(
        ["حافظ در شیراز زاده شد.", "فوتبال ورزشی گروهی است."],
        HashingEmbedder(dimension=64),
        document_ids=["hafez", "sport"],
        store=QdrantVectorStore("hybrid", dimension=64),
    )
    results = retriever.retrieve("حافظ شیراز", k=1, mode="dense")
    assert results[0].document_id == "hafez"


def test_qdrant_point_ids_are_stable() -> None:
    pytest.importorskip("qdrant_client")
    from persian_rag.vectorstores.qdrant import point_id

    assert point_id("doc:1") == point_id("doc:1")
    assert uuid.UUID(point_id("doc:1")).version == 5


def test_qdrant_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    from persian_rag.vectorstores import qdrant

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("qdrant_client"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"persian-rag\[qdrant\]"):
        qdrant.QdrantVectorStore(dimension=3)


# --------------------------------------------------------------------------- pgvector


def _parse_vector(text: str) -> list[float]:
    return [float(value) for value in text.strip("[]").split(",")]


class FakeResult:
    def __init__(self, rows: list[tuple[Any, ...]]) -> None:
        self._rows = rows

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None


class FakePg:
    """Stands in for a psycopg 3 connection, which exposes ``execute`` directly."""

    def __init__(self) -> None:
        self.rows: dict[str, tuple[list[float], str, str]] = {}
        self.log: list[str] = []

    def execute(self, query: str, params: Sequence[Any] | None = None) -> FakeResult:
        self.log.append(query)
        values = list(params or [])
        if query.startswith("SELECT id FROM"):
            wanted = set(values[0])
            return FakeResult([(key,) for key in self.rows if key in wanted])
        if query.startswith("SELECT id, text, metadata"):
            vector = _parse_vector(values[0])
            scored = []
            for key, (embedding, text, metadata) in self.rows.items():
                dot = sum(a * b for a, b in zip(vector, embedding, strict=True))
                norm = math.sqrt(sum(a * a for a in vector)) * math.sqrt(
                    sum(b * b for b in embedding)
                )
                scored.append((key, text, json.loads(metadata), 1 - dot / norm))
            scored.sort(key=lambda row: (row[3], row[0]))
            return FakeResult(scored[: values[1]])
        if query.startswith("SELECT count(*)"):
            return FakeResult([(len(self.rows),)])
        if query.startswith("INSERT"):
            assert query.count("(%s, %s::vector, %s, %s::jsonb)") * 4 == len(values)
            for offset in range(0, len(values), 4):
                key, embedding, text, metadata = values[offset : offset + 4]
                self.rows[key] = (_parse_vector(embedding), text, metadata)
        elif query.startswith("DELETE"):
            for key in values[0]:
                self.rows.pop(key, None)
        return FakeResult([])


def test_pgvector_with_fake_connection() -> None:
    db = FakePg()
    store = PgVectorStore(connection=db, dimension=2, table="fa_chunks", create_index=True)
    assert db.log[0] == "CREATE EXTENSION IF NOT EXISTS vector"
    assert 'CREATE TABLE IF NOT EXISTS "fa_chunks"' in db.log[1]
    assert "vector(2)" in db.log[1]
    assert "USING hnsw (embedding vector_cosine_ops)" in db.log[2]
    store.add(["a", "b"], [[1, 0], [0, 1]], ["الف", "ب"], [{"lang": "fa"}, {}])
    assert len(store) == 2
    hits = store.search([1, 0.2], k=1)
    assert hits[0].id == "a"
    assert hits[0].metadata == {"lang": "fa"}
    assert hits[0].score == pytest.approx(1 / math.sqrt(1.04))
    with pytest.raises(ValueError, match="Duplicate"):
        store.add(["a"], [[1, 0]], ["x"])
    with pytest.raises(ValueError):
        store.add(["c"], [[1, 0, 0]], ["x"])
    with pytest.raises(ValueError):
        store.search([1], k=1)
    with pytest.raises(ValueError):
        store.search([1, 0], k=0)
    store.add([], [], [])
    store.delete(["a"])
    store.delete([])
    assert len(store) == 1


def test_pgvector_inserts_in_batches(monkeypatch: pytest.MonkeyPatch) -> None:
    from persian_rag.vectorstores import pgvector

    monkeypatch.setattr(pgvector, "_INSERT_BATCH", 2)
    db = FakePg()
    store = PgVectorStore(connection=db, dimension=2, create_table=False)
    store.add(["a", "b", "c"], [[1, 0], [0, 1], [1, 1]], ["x", "y", "z"])
    inserts = [query for query in db.log if query.startswith("INSERT")]
    assert len(inserts) == 2
    assert len(store) == 3


def test_pgvector_validation() -> None:
    with pytest.raises(ValueError):
        PgVectorStore(connection=FakePg(), dimension=2, table="bad-name; drop")
    with pytest.raises(ValueError):
        PgVectorStore(connection=FakePg(), dimension=0)
    with pytest.raises(ValueError):
        PgVectorStore(connection=FakePg(), dimension=2, distance="hamming")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        PgVectorStore(dimension=2)
    l2 = PgVectorStore(connection=FakePg(), dimension=2, distance="l2", create_table=False)
    assert l2._score(2.0) == -2.0
    assert vector_literal([1, 2.5]) == "[1.0,2.5]"


@pytest.mark.postgres
def test_pgvector_against_real_database() -> None:
    dsn = os.environ.get("PERSIAN_RAG_PG_DSN")
    if not dsn:
        pytest.skip("set PERSIAN_RAG_PG_DSN to run against PostgreSQL with pgvector")
    pytest.importorskip("psycopg")
    table = f"persian_rag_test_{uuid.uuid4().hex[:8]}"
    store = PgVectorStore(dsn, dimension=3, table=table, create_index=True)
    try:
        store.add(
            ["a", "b", "c"],
            [[1, 0, 0], [0, 1, 0], [0.9, 0.1, 0]],
            ["حافظ", "سعدی", "شیراز"],
            [{"lang": "fa"}, {}, {}],
        )
        hits = store.search([1, 0, 0], k=2)
        assert [hit.id for hit in hits] == ["a", "c"]
        assert hits[0].metadata == {"lang": "fa"}
        assert hits[0].score == pytest.approx(1.0)
        with pytest.raises(ValueError):
            store.add(["a"], [[1, 0, 0]], ["x"])
        store.delete(["a"])
        assert len(store) == 2
    finally:
        store.connection.execute(f'DROP TABLE IF EXISTS "{table}"')
        store.connection.close()
