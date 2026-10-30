from __future__ import annotations

import asyncio

import pytest

from persian_rag.chat import (
    ChatSession,
    ConversationMemory,
    heuristic_standalone,
    needs_context,
)
from persian_rag.pipeline import RAGPipeline
from persian_rag.retrieval import HybridRetriever


def test_memory_keeps_last_turns() -> None:
    memory = ConversationMemory(max_turns=2, max_answer_chars=10)
    for index in range(3):
        memory.add(f"q{index}", "پاسخ " * 5)
    assert [turn.question for turn in memory.turns] == ["q1", "q2"]
    assert memory.format().splitlines()[1].endswith("...")
    assert memory.messages()[0] == {"role": "user", "content": "q1"}
    restored = ConversationMemory.from_dict(memory.to_dict())
    assert restored.turns == memory.turns
    with pytest.raises(ValueError):
        ConversationMemory(0)


def test_needs_context() -> None:
    assert needs_context("و پایتختش؟")
    assert needs_context("چرا؟")
    assert needs_context("درباره آن بیشتر توضیح بده")
    assert not needs_context("شاعر معروف شیراز در قرن هشتم چه کسی بود")


def test_heuristic_standalone_prefixes_previous_question() -> None:
    memory = ConversationMemory()
    assert heuristic_standalone("و پایتختش؟", memory) == "و پایتختش؟"
    memory.add("ایران کجاست", "در آسیا")
    assert heuristic_standalone("و پایتختش؟", memory) == "ایران کجاست و پایتختش؟"


def test_session_without_llm_uses_heuristic(corpus_retriever: HybridRetriever) -> None:
    session = ChatSession(RAGPipeline(corpus_retriever))
    first = session.ask("شاهنامه را چه کسی سرود", k=1)
    assert first.standalone_question == "شاهنامه را چه کسی سرود"
    assert first.answer.contexts[0].document_id == "ferdowsi"
    second = session.ask("داستان رستم؟", k=1)
    assert second.standalone_question.startswith("شاهنامه را چه کسی سرود")
    assert second.turn == 2
    assert "گفتگوی پیشین" in second.answer.prompt
    session.reset()
    assert len(session.memory) == 0


def test_session_condenses_with_llm(corpus_retriever: HybridRetriever) -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        if "پرسش مستقل:" in prompt:
            return "پرسش مستقل: پایتخت ایران کجاست؟"
        return "تهران [1]."

    session = ChatSession(RAGPipeline(corpus_retriever, llm, citations=True))
    session.ask("درباره ایران بگو", k=1)
    reply = session.ask("پایتختش کجاست؟", k=1)
    assert reply.standalone_question == "پایتخت ایران کجاست؟"
    assert reply.answer.contexts[0].document_id == "tehran"
    assert reply.answer.citations[0].document_id == "tehran"
    assert reply.to_dict()["standalone_question"] == "پایتخت ایران کجاست؟"


def test_condense_llm_requires_model(corpus_retriever: HybridRetriever) -> None:
    with pytest.raises(ValueError):
        ChatSession(RAGPipeline(corpus_retriever), condense="llm")
    with pytest.raises(ValueError):
        ChatSession(RAGPipeline(corpus_retriever), condense="other")  # type: ignore[arg-type]


def test_async_session_with_async_llm(corpus_retriever: HybridRetriever) -> None:
    async def llm(prompt: str) -> str:
        return "پاسخ"

    session = ChatSession(RAGPipeline(corpus_retriever, llm))

    async def run() -> None:
        await session.aask("فوتبال چیست", k=1)
        reply = await session.aask("چرا؟", k=1)
        assert reply.answer.answer == "پاسخ"
        assert reply.standalone_question == "فوتبال چیست چرا؟"

    asyncio.run(run())


def test_empty_message(corpus_retriever: HybridRetriever) -> None:
    with pytest.raises(ValueError):
        ChatSession(RAGPipeline(corpus_retriever)).ask("  ")
