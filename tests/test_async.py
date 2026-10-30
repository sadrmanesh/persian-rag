from __future__ import annotations

import asyncio

import pytest

from persian_rag.pipeline import RAGPipeline
from persian_rag.retrieval import HybridRetriever


def test_aretrieve_matches_retrieve(corpus_retriever: HybridRetriever) -> None:
    expected = corpus_retriever.retrieve("حافظ", k=2)
    assert asyncio.run(corpus_retriever.aretrieve("حافظ", k=2)) == expected


def test_aquery_with_async_llm(corpus_retriever: HybridRetriever) -> None:
    async def llm(prompt: str) -> str:
        await asyncio.sleep(0)
        return "شیراز"

    pipeline = RAGPipeline(corpus_retriever, llm)
    answer = asyncio.run(pipeline.aquery("حافظ کجا زاده شد؟", k=2))
    assert answer.answer == "شیراز"
    assert answer.contexts[0].document_id == "hafez"
    with pytest.raises(TypeError, match="aquery"):
        pipeline.query("حافظ")


def test_aquery_with_async_callable_object(corpus_retriever: HybridRetriever) -> None:
    class Model:
        async def __call__(self, prompt: str) -> str:
            return "ok"

    answer = asyncio.run(RAGPipeline(corpus_retriever, Model()).aquery("حافظ"))
    assert answer.answer == "ok"


def test_aquery_with_sync_llm_and_concurrency(corpus_retriever: HybridRetriever) -> None:
    pipeline = RAGPipeline(corpus_retriever, lambda prompt: "پاسخ")

    async def run() -> list[str]:
        answers = await asyncio.gather(
            *(pipeline.aquery(question) for question in ("حافظ", "فردوسی", "تهران"))
        )
        return [item.answer for item in answers]

    assert asyncio.run(run()) == ["پاسخ"] * 3
    no_llm = asyncio.run(RAGPipeline(corpus_retriever).aquery("حافظ"))
    assert no_llm.answer == ""


def test_aquery_rejects_non_string(corpus_retriever: HybridRetriever) -> None:
    async def llm(prompt: str) -> str:
        return 42  # type: ignore[return-value]

    with pytest.raises(TypeError):
        asyncio.run(RAGPipeline(corpus_retriever, llm).aquery("حافظ"))
