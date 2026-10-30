from __future__ import annotations

import asyncio
from typing import Any

import pytest

from persian_rag.errors import PersianRagError
from persian_rag.pipeline import RAGPipeline
from persian_rag.rerank import (
    CrossEncoderReranker,
    LLMReranker,
    Reranker,
    arerank,
    parse_ranking,
)
from persian_rag.retrieval import HybridRetriever, RetrievalResult


def _result(chunk_id: str, text: str, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=chunk_id.split("#")[0],
        text=text,
        score=score,
        start=0,
        end=len(text),
        metadata={},
    )


class KeywordModel:
    """Scores a pair by how many query words appear in the passage."""

    def __init__(self) -> None:
        self.calls: list[list[tuple[str, str]]] = []

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.calls.append(sentences)
        return [
            float(sum(word in passage for word in query.split())) for query, passage in sentences
        ]


def test_cross_encoder_reorders_and_keeps_first_stage_score() -> None:
    results = [
        _result("a#0", "فوتبال ورزش است", 0.9),
        _result("b#0", "حافظ در شیراز زاده شد", 0.5),
        _result("c#0", "شیراز شهری زیباست", 0.4),
    ]
    model = KeywordModel()
    reranker = CrossEncoderReranker(model=model)
    assert isinstance(reranker, Reranker)
    ranked = reranker.rerank("حافظ شیراز", results, top_k=2)
    assert [item.chunk_id for item in ranked] == ["b#0", "c#0"]
    assert ranked[0].score == 2.0
    assert ranked[0].retrieval_score == 0.5
    assert ranked[0].to_dict()["retrieval_score"] == 0.5
    assert model.calls[0][0] == ("حافظ شیراز", "فوتبال ورزش است")


def test_cross_encoder_edge_cases() -> None:
    reranker = CrossEncoderReranker(model=KeywordModel())
    assert reranker.rerank("x", []) == []
    assert reranker.score("x", []) == []
    with pytest.raises(ValueError):
        reranker.rerank("x", [_result("a", "x", 1.0)], top_k=0)
    with pytest.raises(ValueError):
        CrossEncoderReranker(model=KeywordModel(), batch_size=0)

    class Broken:
        def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> list[float]:
            return [1.0]

    with pytest.raises(PersianRagError):
        CrossEncoderReranker(model=Broken()).rerank(
            "x", [_result("a", "x", 1), _result("b", "y", 1)]
        )


def test_cross_encoder_without_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("sentence_transformers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match=r"persian-rag\[rerank\]"):
        CrossEncoderReranker()


@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ("[2, 1, 3]", [1, 0, 2]),
        ("Ranking:\n```json\n[3,1]\n```", [2, 0, 1]),
        ("3 > 2 > 1", [2, 1, 0]),
        ("[2, 2, 9, 1]", [1, 0, 2]),
        ("no idea", [0, 1, 2]),
        ('["3", "1", "2"]', [2, 0, 1]),
    ],
)
def test_parse_ranking(reply: str, expected: list[int]) -> None:
    assert parse_ranking(reply, 3) == expected


def test_llm_reranker_listwise() -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "[3, 1]"

    results = [_result(f"d{n}", f"متن {n}", 1.0 / n) for n in range(1, 5)]
    reranker = LLMReranker(llm, max_candidates=3, max_passage_chars=10)
    ranked = reranker.rerank("پرسش", results)
    assert [item.chunk_id for item in ranked] == ["d3", "d1", "d2", "d4"]
    assert ranked[0].score > ranked[1].score > ranked[-1].score
    assert ranked[0].retrieval_score == pytest.approx(1 / 3)
    assert "[3] متن 3" in prompts[0]
    assert "d4" not in prompts[0] and "متن 4" not in prompts[0]
    assert "پرسش" in prompts[0]
    assert LLMReranker(llm).rerank("q", []) == []
    long = LLMReranker(llm, max_passage_chars=3).build_prompt("q", ["abcdef"])
    assert "abc..." in long


def test_llm_reranker_validation() -> None:
    with pytest.raises(ValueError):
        LLMReranker(lambda p: p, prompt_template="{query}")
    with pytest.raises(ValueError):
        LLMReranker(lambda p: p, max_candidates=0)

    def bad(prompt: str) -> Any:
        return 3

    with pytest.raises(TypeError):
        LLMReranker(bad).rerank("q", [_result("a", "x", 1.0)])


def test_pipeline_uses_reranker(corpus_retriever: HybridRetriever) -> None:
    seen: dict[str, int] = {}

    class Reverse:
        def rerank(
            self, query: str, results: Any, top_k: int | None = None
        ) -> list[RetrievalResult]:
            seen["pool"] = len(results)
            ordered = list(reversed(results))
            return ordered[:top_k] if top_k else ordered

    pipeline = RAGPipeline(corpus_retriever, reranker=Reverse(), rerank_candidates=3)
    answer = pipeline.query("حافظ شیراز", k=1)
    assert seen["pool"] == 3
    assert len(answer.contexts) == 1
    assert answer.contexts[0].document_id != "hafez"
    with pytest.raises(ValueError):
        RAGPipeline(corpus_retriever, rerank_candidates=0)


def test_async_rerank() -> None:
    reranker = CrossEncoderReranker(model=KeywordModel())
    results = [_result("a", "x", 1.0), _result("b", "شیراز", 0.1)]
    ranked = asyncio.run(arerank(reranker, "شیراز", results, top_k=1))
    assert ranked[0].chunk_id == "b"
