from pathlib import Path

import pytest

from persian_rag.chunk import ChunkingConfig
from persian_rag.embeddings import HashingEmbedder
from persian_rag.errors import IndexFormatError, IndexNotFoundError
from persian_rag.persist import load_index, retriever_from_saved, save_index
from persian_rag.pipeline import RAGPipeline, build_retriever


def test_pipeline_calls_the_llm_with_retrieved_context() -> None:
    seen: list[str] = []

    def llm(prompt: str) -> str:
        seen.append(prompt)
        return "شیراز"

    retriever = build_retriever(
        [
            "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
            "فوتبال ورزشی گروهی با دو تیم است.",
        ],
        HashingEmbedder(dimension=64),
        document_ids=["hafez", "sport"],
        metadatas=[{"source": "hafez"}, {"source": "sport"}],
    )
    answer = RAGPipeline(retriever, llm).query("حافظ در کدام شهر زاده شد؟", k=2)
    assert answer.answer == "شیراز"
    assert "شیراز" in answer.prompt
    assert "حافظ" in seen[0]
    assert answer.contexts[0].document_id == "hafez"


def test_pipeline_without_llm_returns_context_and_empty_answer() -> None:
    retriever = build_retriever(["کتاب روی میز است."], HashingEmbedder(dimension=16))
    answer = RAGPipeline(retriever).query("کتاب کجاست؟")
    assert answer.answer == ""
    assert answer.contexts
    assert "کتاب" in answer.prompt


def test_llm_must_return_a_string() -> None:
    retriever = build_retriever(["سلام"], HashingEmbedder(dimension=8))

    def bad_llm(_prompt: str) -> str:
        return None  # type: ignore[return-value]

    with pytest.raises(TypeError):
        RAGPipeline(retriever, bad_llm).query("سلام")


def test_index_roundtrip_keeps_the_top_hit(tmp_path: Path) -> None:
    embedder = HashingEmbedder(dimension=32, seed=5)
    retriever = build_retriever(
        [
            "حافظ در شهر شیراز زاده شد.",
            "فوتبال ورزشی گروهی است.",
        ],
        embedder,
        document_ids=["hafez", "sport"],
        chunking=None,
    )
    before = retriever.retrieve("حافظ شیراز", k=1)[0].chunk_id
    target = save_index(tmp_path / "index", retriever, embedder.to_spec(), ChunkingConfig())
    saved = load_index(target)
    restored = retriever_from_saved(saved)
    after = restored.retrieve("حافظ شیراز", k=1)[0].chunk_id
    assert before == after == "hafez:0"
    assert "api_key" not in target.read_text(encoding="utf-8")


def test_missing_and_invalid_index(tmp_path: Path) -> None:
    with pytest.raises(IndexNotFoundError):
        load_index(tmp_path / "missing")
    bad = tmp_path / "index.json"
    bad.write_text('{"version": "9.9.9"}', encoding="utf-8")
    with pytest.raises(IndexFormatError):
        load_index(bad)
