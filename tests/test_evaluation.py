from __future__ import annotations

import json
from pathlib import Path

import pytest

from persian_rag.errors import PersianRagError
from persian_rag.evaluation import (
    EvalExample,
    FaithfulnessScorer,
    LLMFaithfulnessJudge,
    TokenOverlapFaithfulness,
    evaluate_answers,
    evaluate_retrieval,
    load_eval_examples,
    parse_judge_score,
    retrieval_metrics,
    token_f1,
)
from persian_rag.pipeline import RAGPipeline
from persian_rag.retrieval import HybridRetriever


def test_retrieval_metrics_single_query() -> None:
    ranked = ["x", "a", "y", "b"]
    at_1 = retrieval_metrics(ranked, ["a", "b"], 1)
    assert at_1 == {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "ndcg": 0.0}
    at_4 = retrieval_metrics(ranked, ["a", "b"], 4)
    assert at_4["hit_rate"] == 1.0
    assert at_4["mrr"] == 0.5
    assert at_4["recall"] == 1.0
    assert 0 < at_4["ndcg"] < 1
    assert retrieval_metrics(["a"], ["a"], 3)["ndcg"] == 1.0
    assert retrieval_metrics(["a"], [], 3)["hit_rate"] == 0.0
    with pytest.raises(ValueError):
        retrieval_metrics(["a"], ["a"], 0)


def test_evaluate_retrieval_on_corpus(corpus_retriever: HybridRetriever) -> None:
    examples = [
        EvalExample("حافظ در کدام شهر زاده شد؟", ("hafez",)),
        EvalExample("شاهنامه را چه کسی سرود؟", ("ferdowsi",)),
        EvalExample("پایتخت ایران کجاست؟", ("tehran",)),
        EvalExample("بدون برچسب", ()),
    ]
    report = evaluate_retrieval(corpus_retriever, examples, k_values=(3, 1))
    assert report.k_values == (1, 3)
    assert report.num_queries == 3
    assert report.metrics[3]["hit_rate"] == 1.0
    assert report.metrics[1]["mrr"] <= report.metrics[3]["mrr"]
    payload = report.to_dict(include_queries=True)
    assert set(payload["metrics"]) == {"@1", "@3"}
    assert payload["queries"][0]["retrieved_ids"][0] == "hafez"
    assert "hit_rate" in report.format_table()

    pipeline_report = evaluate_retrieval(RAGPipeline(corpus_retriever), examples, k_values=[2])
    assert pipeline_report.metrics[2]["hit_rate"] == 1.0
    chunk_report = evaluate_retrieval(
        corpus_retriever, [EvalExample("حافظ", ("hafez:0",))], k_values=[1], match="chunk"
    )
    assert chunk_report.metrics[1]["hit_rate"] == 1.0


def test_evaluate_retrieval_validation(corpus_retriever: HybridRetriever) -> None:
    with pytest.raises(PersianRagError):
        evaluate_retrieval(corpus_retriever, [EvalExample("x")])
    with pytest.raises(ValueError):
        evaluate_retrieval(corpus_retriever, [EvalExample("x", ("a",))], k_values=[0])
    with pytest.raises(ValueError):
        evaluate_retrieval(
            corpus_retriever,
            [EvalExample("x", ("a",))],
            match="page",  # type: ignore[arg-type]
        )


def test_load_eval_examples(tmp_path: Path) -> None:
    path = tmp_path / "eval.jsonl"
    rows = [
        {"id": 1, "question": "حافظ کجا زاده شد؟", "relevant_ids": ["hafez"], "answer": "شیراز"},
        {"question": "شاهنامه", "relevant": "ferdowsi"},
    ]
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n\n", encoding="utf-8"
    )
    examples = load_eval_examples(path)
    assert examples[0] == EvalExample("حافظ کجا زاده شد؟", ("hafez",), "شیراز", "1")
    assert examples[1].relevant_ids == ("ferdowsi",)

    for bad in (
        "{not json}",
        "[1]",
        '{"question": ""}',
        '{"question": "x", "relevant_ids": 3}',
        "",
    ):
        path.write_text(bad, encoding="utf-8")
        with pytest.raises(PersianRagError):
            load_eval_examples(path)


def test_token_overlap_faithfulness() -> None:
    scorer = TokenOverlapFaithfulness()
    assert isinstance(scorer, FaithfulnessScorer)
    contexts = ["حافظ در شهر شیراز زاده شد."]
    supported = scorer.score("q", "حافظ در شیراز زاده شد", contexts)
    assert supported.score == 1.0
    partial = scorer.score("q", "حافظ در تبریز زاده شد", contexts)
    assert 0 < partial.score < 1
    assert partial.details["unsupported"] == ["تبریز"]
    # Arabic Kaf and Yeh are normalized before matching.
    assert scorer.score("q", "كتاب علي", ["کتاب علی"]).score == 1.0
    assert scorer.score("q", "و در", contexts).score == 1.0


def test_llm_judge_and_score_parsing() -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return 'Sure. {"score": 0.75, "reason": "mostly supported"}'

    judge = LLMFaithfulnessJudge(llm, max_context_chars=50)
    result = judge.score("پرسش", "پاسخ", ["متن " * 40])
    assert result.score == 0.75
    assert result.details == {"reason": "mostly supported"}
    assert "پرسش" in prompts[0] and prompts[0].count("متن") < 40
    assert parse_judge_score('{"score": "0.5"}') == (0.5, "")
    assert parse_judge_score("score: 7") == (1.0, "")
    assert parse_judge_score('{"score": true} 0.2') == (0.2, "")
    with pytest.raises(PersianRagError):
        parse_judge_score("no number")
    with pytest.raises(ValueError):
        LLMFaithfulnessJudge(llm, prompt_template="{context}")


def test_token_f1() -> None:
    assert token_f1("شیراز", "شیراز") == 1.0
    assert token_f1("در شیراز", "شهر شیراز") == pytest.approx(2 / 3)
    assert token_f1("تبریز", "شیراز") == 0.0
    assert token_f1("", "") == 1.0


def test_evaluate_answers(corpus_retriever: HybridRetriever) -> None:
    pipeline = RAGPipeline(corpus_retriever, lambda prompt: "حافظ در شیراز زاده شد")
    examples = [EvalExample("حافظ کجا زاده شد؟", ("hafez",), "شیراز")]
    report = evaluate_answers(pipeline, examples, k=2)
    assert report.mean_faithfulness == 1.0
    assert report.mean_reference_f1 is not None and report.mean_reference_f1 > 0
    assert report.to_dict(include_answers=True)["answers"][0]["answer"]

    silent = evaluate_answers(RAGPipeline(corpus_retriever), examples)
    assert silent.mean_faithfulness is None
    assert silent.mean_reference_f1 is None
