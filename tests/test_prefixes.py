"""Default model and query/passage prefixes for sentence-transformers.

A fake ``sentence_transformers`` module records what would be encoded, so
these tests need no model download.
"""

from __future__ import annotations

import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any, ClassVar

import pytest

from persian_rag import (
    DEFAULT_ST_MODEL,
    E5_BASE_MODEL,
    LEGACY_ST_MODEL,
    ST_MODEL_ALIASES,
    ChunkingConfig,
    EmbedderSpec,
    HashingEmbedder,
    IndexFormatError,
    SentenceTransformerEmbedder,
    build_retriever,
    create_embedder,
    is_e5_model,
    load_index,
    resolve_model_name,
    resolve_prefixes,
    retriever_from_saved,
    save_index,
)


class FakeSentenceTransformer:
    prompts_by_model: ClassVar[dict[str, dict[str, str]]] = {}
    calls: ClassVar[list[tuple[str, str, list[str]]]] = []

    def __init__(self, model_name: str, device: str | None = None) -> None:
        self.model_name = model_name
        # sentence-transformers 5+ gives every model empty query/document prompts.
        self.prompts = dict(self.prompts_by_model.get(model_name, {"query": "", "document": ""}))

    def get_sentence_embedding_dimension(self) -> int:
        return 3

    def _vectors(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def encode(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append((self.model_name, "encode", list(texts)))
        return self._vectors(texts)

    def encode_query(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append((self.model_name, "encode_query", list(texts)))
        return self._vectors([self.prompts["query"] + text for text in texts])

    def encode_document(self, texts: list[str], **_: Any) -> list[list[float]]:
        self.calls.append((self.model_name, "encode_document", list(texts)))
        return self._vectors([self.prompts["document"] + text for text in texts])


@pytest.fixture
def fake_st(monkeypatch: pytest.MonkeyPatch) -> Iterator[type[FakeSentenceTransformer]]:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    FakeSentenceTransformer.calls = []
    FakeSentenceTransformer.prompts_by_model = {}
    yield FakeSentenceTransformer


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("intfloat/multilingual-e5-small", True),
        ("intfloat/multilingual-e5-base", True),
        ("intfloat/e5-base-v2", True),
        ("sadrmanesh/persian-rag-embed-small", True),
        ("intfloat/multilingual-e5-large-instruct", False),
        ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False),
        ("BAAI/bge-m3", False),
        ("some/model-e50", False),
    ],
)
def test_is_e5_model(name: str, expected: bool) -> None:
    assert is_e5_model(name) is expected


def test_resolve_prefixes_modes() -> None:
    assert resolve_prefixes("intfloat/multilingual-e5-small") == ("query: ", "passage: ")
    assert resolve_prefixes("BAAI/bge-m3") == ("", "")
    assert resolve_prefixes("BAAI/bge-m3", "e5") == ("query: ", "passage: ")
    assert resolve_prefixes("intfloat/multilingual-e5-small", "none") == ("", "")
    prompts = {"query": "q> ", "document": "d> "}
    assert resolve_prefixes("intfloat/multilingual-e5-small", prompts=prompts) == ("q> ", "d> ")
    empty = {"query": "", "document": ""}
    assert resolve_prefixes("intfloat/multilingual-e5-small", prompts=empty) == (
        "query: ",
        "passage: ",
    )
    assert resolve_prefixes("BAAI/bge-m3", prompts=empty) == ("", "")
    with pytest.raises(ValueError):
        resolve_prefixes("x", "always")  # type: ignore[arg-type]


def test_default_model_is_e5_small_with_prefixes(fake_st: type[FakeSentenceTransformer]) -> None:
    assert DEFAULT_ST_MODEL == "intfloat/multilingual-e5-small"
    embedder = SentenceTransformerEmbedder()
    assert embedder.model_name == DEFAULT_ST_MODEL
    embedder.embed_documents(["حافظ"])
    embedder.embed_query("حافظ کجاست")
    assert fake_st.calls == [
        (DEFAULT_ST_MODEL, "encode", ["passage: حافظ"]),
        (DEFAULT_ST_MODEL, "encode", ["query: حافظ کجاست"]),
    ]
    spec = embedder.to_spec()
    assert (spec.model, spec.query_prefix, spec.passage_prefix) == (
        DEFAULT_ST_MODEL,
        "query: ",
        "passage: ",
    )


def test_other_models_get_no_prefix(fake_st: type[FakeSentenceTransformer]) -> None:
    embedder = SentenceTransformerEmbedder(LEGACY_ST_MODEL)
    embedder.embed_query("سلام")
    assert fake_st.calls == [(LEGACY_ST_MODEL, "encode", ["سلام"])]
    assert embedder.to_spec().query_prefix == ""


def test_prefix_none_and_custom_override(fake_st: type[FakeSentenceTransformer]) -> None:
    off = SentenceTransformerEmbedder(prefix="none")
    off.embed_query("a")
    custom = SentenceTransformerEmbedder("BAAI/bge-m3", query_prefix="Q: ", passage_prefix="")
    custom.embed_query("b")
    custom.embed_documents(["c"])
    assert [call[2] for call in fake_st.calls] == [["a"], ["Q: b"], ["c"]]
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder(prefix="sometimes")  # type: ignore[arg-type]


def test_model_prompts_use_encode_query_and_document(
    fake_st: type[FakeSentenceTransformer],
) -> None:
    name = "sadrmanesh/persian-rag-embed-small"
    fake_st.prompts_by_model[name] = {"query": "query: ", "document": "passage: "}
    embedder = SentenceTransformerEmbedder(name)
    embedder.embed_query("x")
    embedder.embed_documents(["y"])
    assert fake_st.calls == [(name, "encode_query", ["x"]), (name, "encode_document", ["y"])]
    assert embedder.to_spec().passage_prefix == "passage: "


def test_spec_round_trip_keeps_prefixes() -> None:
    spec = EmbedderSpec(
        kind="sentence-transformers",
        dimension=3,
        model=DEFAULT_ST_MODEL,
        query_prefix="query: ",
        passage_prefix="passage: ",
    )
    data = spec.to_dict()
    assert data["query_prefix"] == "query: "
    assert EmbedderSpec.from_dict(data) == spec
    assert "query_prefix" not in HashingEmbedder(dimension=8).to_spec().to_dict()


def test_legacy_spec_without_model_means_old_default() -> None:
    spec = EmbedderSpec.from_dict({"kind": "sentence-transformers", "dimension": 384})
    assert spec.model == LEGACY_ST_MODEL
    assert spec.query_prefix is None


def test_legacy_index_is_queried_without_prefixes(
    fake_st: type[FakeSentenceTransformer],
) -> None:
    old_minilm = EmbedderSpec(kind="sentence-transformers", dimension=3, model=LEGACY_ST_MODEL)
    embedder = create_embedder(old_minilm)
    assert isinstance(embedder, SentenceTransformerEmbedder)
    assert (embedder.model_name, embedder.query_prefix) == (LEGACY_ST_MODEL, "")

    old_e5 = EmbedderSpec(kind="sentence-transformers", dimension=3, model=DEFAULT_ST_MODEL)
    with pytest.warns(UserWarning, match="rebuilding it"):
        e5 = create_embedder(old_e5)
    assert isinstance(e5, SentenceTransformerEmbedder)
    e5.embed_query("x")
    assert fake_st.calls[-1] == (DEFAULT_ST_MODEL, "encode", ["x"])


def _saved_index(tmp_path: Path, embedder: SentenceTransformerEmbedder) -> Path:
    retriever = build_retriever(["شیراز شهر حافظ است", "تهران پایتخت است"], embedder)
    return save_index(tmp_path / "index", retriever, embedder.to_spec(), ChunkingConfig())


def test_model_mismatch_is_refused(tmp_path: Path, fake_st: type[FakeSentenceTransformer]) -> None:
    path = _saved_index(tmp_path, SentenceTransformerEmbedder(LEGACY_ST_MODEL))
    saved = load_index(path)
    assert saved.embedder.model == LEGACY_ST_MODEL
    with pytest.raises(IndexFormatError, match="Rebuild the index"):
        retriever_from_saved(saved, SentenceTransformerEmbedder())
    with pytest.raises(IndexFormatError, match="hashing"):
        retriever_from_saved(saved, HashingEmbedder(dimension=3))


def test_saved_index_restores_model_and_prefixes(
    tmp_path: Path, fake_st: type[FakeSentenceTransformer]
) -> None:
    path = _saved_index(tmp_path, SentenceTransformerEmbedder())
    saved = load_index(path)
    assert (saved.embedder.query_prefix, saved.embedder.passage_prefix) == ("query: ", "passage: ")
    retriever = retriever_from_saved(saved)
    assert len(retriever) == 2
    fake_st.calls.clear()
    retriever.retrieve("حافظ", k=1, mode="dense")
    assert fake_st.calls[-1][2][0].startswith("query: ")


def test_prefix_mismatch_warns(tmp_path: Path, fake_st: type[FakeSentenceTransformer]) -> None:
    path = _saved_index(tmp_path, SentenceTransformerEmbedder())
    with pytest.warns(UserWarning, match="Prefixes differ"):
        retriever_from_saved(load_index(path), SentenceTransformerEmbedder(prefix="none"))


def test_cli_ingest_defaults_to_e5_small_with_prefixes(
    tmp_path: Path, fake_st: type[FakeSentenceTransformer]
) -> None:
    from persian_rag.cli import main

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hafez.txt").write_text("حافظ در شهر شیراز زاده شد.\n", encoding="utf-8")
    index = tmp_path / "index"
    assert (
        main(["ingest", str(docs), "--index", str(index), "--embedder", "sentence-transformers"])
        == 0
    )
    spec = load_index(index).embedder
    assert (spec.model, spec.query_prefix, spec.passage_prefix) == (
        DEFAULT_ST_MODEL,
        "query: ",
        "passage: ",
    )
    assert all(text.startswith("passage: ") for _, _, texts in fake_st.calls for text in texts)


def test_cli_ingest_prefix_flags(tmp_path: Path, fake_st: type[FakeSentenceTransformer]) -> None:
    from persian_rag.cli import main

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("کتاب فارسی\n", encoding="utf-8")
    off = tmp_path / "off"
    custom = tmp_path / "custom"
    base = ["ingest", str(docs), "--embedder", "sentence-transformers"]
    assert main([*base, "--index", str(off), "--prefix", "none"]) == 0
    assert (
        main([*base, "--index", str(custom), "--model", "BAAI/bge-m3", "--query-prefix", "Q: "])
        == 0
    )
    assert load_index(off).embedder.query_prefix == ""
    restored = load_index(custom).embedder
    assert (restored.model, restored.query_prefix, restored.passage_prefix) == (
        "BAAI/bge-m3",
        "Q: ",
        "",
    )


def test_model_aliases() -> None:
    assert E5_BASE_MODEL == "intfloat/multilingual-e5-base"
    assert ST_MODEL_ALIASES == {"e5-small": DEFAULT_ST_MODEL, "e5-base": E5_BASE_MODEL}
    assert resolve_model_name("e5-base") == E5_BASE_MODEL
    assert resolve_model_name(" E5-Small ") == DEFAULT_ST_MODEL
    assert resolve_model_name("BAAI/bge-m3") == "BAAI/bge-m3"
    assert resolve_model_name(E5_BASE_MODEL) == E5_BASE_MODEL


@pytest.mark.parametrize("name", [E5_BASE_MODEL, "e5-base"])
def test_e5_base_gets_prefixes_and_full_name(
    name: str, fake_st: type[FakeSentenceTransformer]
) -> None:
    embedder = SentenceTransformerEmbedder(name)
    assert embedder.model_name == E5_BASE_MODEL
    embedder.embed_documents(["حافظ"])
    embedder.embed_query("حافظ کجاست")
    assert fake_st.calls == [
        (E5_BASE_MODEL, "encode", ["passage: حافظ"]),
        (E5_BASE_MODEL, "encode", ["query: حافظ کجاست"]),
    ]
    spec = embedder.to_spec()
    assert (spec.model, spec.query_prefix, spec.passage_prefix) == (
        E5_BASE_MODEL,
        "query: ",
        "passage: ",
    )
    assert create_embedder(spec).model_name == E5_BASE_MODEL


def test_e5_base_index_refuses_e5_small(
    tmp_path: Path, fake_st: type[FakeSentenceTransformer]
) -> None:
    path = _saved_index(tmp_path, SentenceTransformerEmbedder("e5-base"))
    saved = load_index(path)
    assert saved.embedder.model == E5_BASE_MODEL
    assert len(retriever_from_saved(saved, SentenceTransformerEmbedder(E5_BASE_MODEL))) == 2
    with pytest.raises(IndexFormatError, match="Rebuild the index"):
        retriever_from_saved(saved, SentenceTransformerEmbedder())


def test_cli_ingest_e5_base_alias(tmp_path: Path, fake_st: type[FakeSentenceTransformer]) -> None:
    from persian_rag.cli import main

    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("کتاب فارسی\n", encoding="utf-8")
    index = tmp_path / "index"
    args = ["ingest", str(docs), "--index", str(index), "--embedder", "sentence-transformers"]
    assert main([*args, "--model", "e5-base"]) == 0
    spec = load_index(index).embedder
    assert (spec.model, spec.query_prefix, spec.passage_prefix) == (
        E5_BASE_MODEL,
        "query: ",
        "passage: ",
    )
    assert all(text.startswith("passage: ") for _, _, texts in fake_st.calls for text in texts)


def test_health_reports_embedder(fake_st: type[FakeSentenceTransformer]) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from persian_rag.pipeline import RAGPipeline
    from persian_rag.server import create_app

    retriever = build_retriever(["حافظ"], SentenceTransformerEmbedder())
    health = TestClient(create_app(RAGPipeline(retriever))).get("/health").json()
    assert health["embedder"] == {
        "kind": "sentence-transformers",
        "model": DEFAULT_ST_MODEL,
        "dimension": 3,
        "query_prefix": "query: ",
        "passage_prefix": "passage: ",
    }
