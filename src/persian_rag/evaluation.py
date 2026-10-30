"""Offline evaluation for retrieval and answers.

Retrieval metrics (computed at several cut-offs ``k``):

* ``hit_rate``: share of questions with at least one relevant result.
* ``mrr``: mean reciprocal rank of the first relevant result.
* ``recall``: share of the relevant ids that were retrieved.
* ``ndcg``: normalized discounted cumulative gain with binary relevance.

Answer quality is pluggable through :class:`FaithfulnessScorer`. Two scorers
ship with the package: :class:`TokenOverlapFaithfulness` (deterministic and
offline) and :class:`LLMFaithfulnessJudge` (any ``Callable[[str], str]``).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from persian_rag.errors import PersianRagError
from persian_rag.normalize import NormalizeConfig
from persian_rag.retrieval import RetrievalMode, RetrievalResult
from persian_rag.tokenize import tokenize

MatchOn = Literal["document", "chunk"]
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)


@dataclass(frozen=True, slots=True)
class EvalExample:
    """One labelled question."""

    question: str
    relevant_ids: tuple[str, ...] = ()
    reference_answer: str | None = None
    id: str | None = None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvalExample:
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise PersianRagError("Each example needs a non-empty 'question'")
        raw_ids = data.get("relevant_ids", data.get("relevant", []))
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids]
        if not isinstance(raw_ids, list) or not all(isinstance(item, str) for item in raw_ids):
            raise PersianRagError("'relevant_ids' must be a string or a list of strings")
        reference = data.get("reference_answer", data.get("answer"))
        example_id = data.get("id")
        return cls(
            question=question,
            relevant_ids=tuple(raw_ids),
            reference_answer=reference if isinstance(reference, str) else None,
            id=str(example_id) if example_id is not None else None,
        )


def load_eval_examples(path: str | Path) -> list[EvalExample]:
    """Read examples from a JSON Lines file (one object per line)."""

    source = Path(path)
    examples: list[EvalExample] = []
    for number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PersianRagError(f"{source}:{number}: invalid JSON ({exc.msg})") from exc
        if not isinstance(data, dict):
            raise PersianRagError(f"{source}:{number}: expected a JSON object")
        try:
            examples.append(EvalExample.from_dict(data))
        except PersianRagError as exc:
            raise PersianRagError(f"{source}:{number}: {exc}") from exc
    if not examples:
        raise PersianRagError(f"{source} contains no examples")
    return examples


class SupportsRetrieve(Protocol):
    """Anything with ``retrieve(question, k=..., mode=...)``."""

    def retrieve(
        self, question: str, /, *, k: int = ..., mode: RetrievalMode = ...
    ) -> list[RetrievalResult]: ...


def _ranked_ids(results: Sequence[RetrievalResult], match: MatchOn) -> list[str]:
    ranked: list[str] = []
    seen: set[str] = set()
    for result in results:
        key = result.document_id if match == "document" else result.chunk_id
        if key not in seen:
            seen.add(key)
            ranked.append(key)
    return ranked


def retrieval_metrics(
    ranked_ids: Sequence[str],
    relevant_ids: Iterable[str],
    k: int,
) -> dict[str, float]:
    """Metrics for one query at cut-off ``k``."""

    if k < 1:
        raise ValueError("k must be >= 1")
    relevant = set(relevant_ids)
    top = list(ranked_ids[:k])
    if not relevant:
        return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0, "ndcg": 0.0}
    first = next((rank for rank, item in enumerate(top, start=1) if item in relevant), None)
    found = sum(1 for item in top if item in relevant)
    dcg = sum(
        1.0 / math.log2(rank + 1) for rank, item in enumerate(top, start=1) if item in relevant
    )
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return {
        "hit_rate": 1.0 if first is not None else 0.0,
        "mrr": 1.0 / first if first is not None else 0.0,
        "recall": found / len(relevant),
        "ndcg": dcg / ideal if ideal else 0.0,
    }


@dataclass(frozen=True, slots=True)
class QueryEvaluation:
    question: str
    relevant_ids: tuple[str, ...]
    retrieved_ids: tuple[str, ...]
    metrics: dict[int, dict[str, float]]


@dataclass(frozen=True, slots=True)
class RetrievalReport:
    """Mean metrics per cut-off plus per-question details."""

    k_values: tuple[int, ...]
    metrics: dict[int, dict[str, float]]
    queries: tuple[QueryEvaluation, ...]
    match: MatchOn = "document"

    @property
    def num_queries(self) -> int:
        return len(self.queries)

    def to_dict(self, *, include_queries: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_queries": self.num_queries,
            "match": self.match,
            "metrics": {f"@{k}": dict(values) for k, values in self.metrics.items()},
        }
        if include_queries:
            payload["queries"] = [
                {
                    "question": item.question,
                    "relevant_ids": list(item.relevant_ids),
                    "retrieved_ids": list(item.retrieved_ids),
                    "metrics": {f"@{k}": dict(v) for k, v in item.metrics.items()},
                }
                for item in self.queries
            ]
        return payload

    def format_table(self) -> str:
        """A small plain-text table for terminals."""

        header = f"{'k':>4}  {'hit_rate':>8}  {'mrr':>8}  {'recall':>8}  {'ndcg':>8}"
        lines = [header]
        for k in self.k_values:
            row = self.metrics[k]
            lines.append(
                f"{k:>4}  {row['hit_rate']:>8.4f}  {row['mrr']:>8.4f}  "
                f"{row['recall']:>8.4f}  {row['ndcg']:>8.4f}"
            )
        return "\n".join(lines)


def evaluate_retrieval(
    retriever: SupportsRetrieve,
    examples: Sequence[EvalExample],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    mode: RetrievalMode = "hybrid",
    match: MatchOn = "document",
) -> RetrievalReport:
    """Score a retriever (or a :class:`~persian_rag.pipeline.RAGPipeline`).

    ``match="document"`` compares ``relevant_ids`` with document ids, so
    several chunks from one relevant file count once. ``match="chunk"``
    compares chunk ids. Examples without relevant ids are skipped.
    """

    ks = tuple(sorted(set(k_values)))
    if not ks or ks[0] < 1:
        raise ValueError("k_values must contain positive integers")
    if match not in ("document", "chunk"):
        raise ValueError("match must be 'document' or 'chunk'")
    labelled = [example for example in examples if example.relevant_ids]
    if not labelled:
        raise PersianRagError("No examples with relevant_ids to evaluate")
    depth = ks[-1]
    queries: list[QueryEvaluation] = []
    for example in labelled:
        results = retriever.retrieve(example.question, k=depth, mode=mode)
        ranked = _ranked_ids(results, match)
        per_k = {k: retrieval_metrics(ranked, example.relevant_ids, k) for k in ks}
        queries.append(
            QueryEvaluation(
                question=example.question,
                relevant_ids=example.relevant_ids,
                retrieved_ids=tuple(ranked),
                metrics=per_k,
            )
        )
    means = {
        k: {
            name: sum(query.metrics[k][name] for query in queries) / len(queries)
            for name in ("hit_rate", "mrr", "recall", "ndcg")
        }
        for k in ks
    }
    return RetrievalReport(k_values=ks, metrics=means, queries=tuple(queries), match=match)


# ---------------------------------------------------------------------------
# Answer faithfulness


@dataclass(frozen=True, slots=True)
class FaithfulnessResult:
    """A score in ``[0, 1]``; 1 means fully supported by the contexts."""

    score: float
    details: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class FaithfulnessScorer(Protocol):
    """Scores how well an answer is supported by its retrieved contexts."""

    def score(self, question: str, answer: str, contexts: Sequence[str]) -> FaithfulnessResult:
        """Return a :class:`FaithfulnessResult`."""


_STOPWORDS = frozenset(
    {
        # Persian
        "و",
        "در",
        "به",
        "از",
        "که",
        "این",
        "آن",
        "را",
        "با",
        "است",
        "برای",
        "یک",
        "تا",
        "بر",
        "هم",
        "نیز",
        "یا",
        "اما",
        "هر",
        "شد",
        "شده",
        "می",
        "بود",
        "کرد",
        "های",
        "ها",
        # English
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "with",
    }
)


def content_tokens(text: str, config: NormalizeConfig | None = None) -> list[str]:
    """Normalized tokens without common Persian and English stopwords."""

    return [token for token in tokenize(text, config=config) if token not in _STOPWORDS]


class TokenOverlapFaithfulness:
    """Share of the answer's content tokens that appear in the contexts.

    Cheap, deterministic, and language-aware through the Persian normalizer,
    so ``كتاب`` and ``کتاب`` match. It is a lexical proxy: paraphrases score
    lower than they should, and copied but wrong claims score high. Use it
    for regression tracking, and an LLM judge for absolute quality.
    """

    def __init__(
        self, config: NormalizeConfig | None = None, *, empty_answer_score: float = 1.0
    ) -> None:
        self.config = config
        self.empty_answer_score = empty_answer_score

    def score(self, question: str, answer: str, contexts: Sequence[str]) -> FaithfulnessResult:
        answer_tokens = content_tokens(answer, self.config)
        if not answer_tokens:
            return FaithfulnessResult(self.empty_answer_score, {"answer_tokens": 0})
        support: set[str] = set()
        for context in contexts:
            support.update(content_tokens(context, self.config))
        unsupported = [token for token in answer_tokens if token not in support]
        value = 1.0 - len(unsupported) / len(answer_tokens)
        return FaithfulnessResult(
            value,
            {
                "answer_tokens": len(answer_tokens),
                "unsupported": sorted(set(unsupported)),
            },
        )


DEFAULT_FAITHFULNESS_PROMPT = """\
You check whether an answer is supported by the given context. The texts may be
in Persian or English.

Context:
{context}

Question: {question}

Answer: {answer}

Rate how much of the answer is directly supported by the context, from 0
(unsupported or contradicted) to 1 (fully supported). Reply with JSON only:
{{"score": <number between 0 and 1>, "reason": "<one short sentence>"}}"""

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_judge_score(text: str) -> tuple[float, str]:
    """Extract ``(score, reason)`` from a judge reply; the score is clamped to [0, 1]."""

    reason = ""
    score: float | None = None
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            raw = data.get("score")
            if isinstance(raw, bool):
                raw = None
            if isinstance(raw, (int, float)) or (
                isinstance(raw, str) and _NUMBER_RE.fullmatch(raw.strip())
            ):
                score = float(raw)
            if isinstance(data.get("reason"), str):
                reason = data["reason"]
    if score is None:
        number = _NUMBER_RE.search(text)
        if number is None:
            raise PersianRagError(f"Could not parse a score from the judge reply: {text[:200]!r}")
        score = float(number.group(0))
    return min(1.0, max(0.0, score)), reason


class LLMFaithfulnessJudge:
    """Ask a language model to rate faithfulness on a 0 to 1 scale."""

    def __init__(
        self,
        llm: Callable[[str], str],
        *,
        prompt_template: str = DEFAULT_FAITHFULNESS_PROMPT,
        max_context_chars: int = 6000,
    ) -> None:
        for placeholder in ("{context}", "{question}", "{answer}"):
            if placeholder not in prompt_template:
                raise ValueError(f"prompt_template must contain {placeholder}")
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_context_chars = max_context_chars

    def build_prompt(self, question: str, answer: str, contexts: Sequence[str]) -> str:
        context = "\n\n".join(f"[{n}] {text}" for n, text in enumerate(contexts, start=1))
        if len(context) > self.max_context_chars:
            context = context[: self.max_context_chars].rstrip() + "..."
        return self.prompt_template.format(context=context, question=question, answer=answer)

    def score(self, question: str, answer: str, contexts: Sequence[str]) -> FaithfulnessResult:
        reply = self.llm(self.build_prompt(question, answer, contexts))
        if not isinstance(reply, str):
            raise TypeError("LLM callable must return a string")
        value, reason = parse_judge_score(reply)
        return FaithfulnessResult(value, {"reason": reason})


def token_f1(prediction: str, reference: str, config: NormalizeConfig | None = None) -> float:
    """Token-level F1 between an answer and a reference answer."""

    pred = content_tokens(prediction, config)
    ref = content_tokens(reference, config)
    if not pred or not ref:
        return 1.0 if pred == ref else 0.0
    remaining = list(ref)
    common = 0
    for token in pred:
        if token in remaining:
            remaining.remove(token)
            common += 1
    if common == 0:
        return 0.0
    precision = common / len(pred)
    recall = common / len(ref)
    return 2 * precision * recall / (precision + recall)


class SupportsQuery(Protocol):
    def query(self, question: str, /, *, k: int = ..., mode: RetrievalMode = ...) -> Any: ...


@dataclass(frozen=True, slots=True)
class AnswerEvaluation:
    question: str
    answer: str
    faithfulness: float | None
    reference_f1: float | None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PipelineReport:
    answers: tuple[AnswerEvaluation, ...]
    mean_faithfulness: float | None
    mean_reference_f1: float | None

    def to_dict(self, *, include_answers: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "num_questions": len(self.answers),
            "mean_faithfulness": self.mean_faithfulness,
            "mean_reference_f1": self.mean_reference_f1,
        }
        if include_answers:
            payload["answers"] = [
                {
                    "question": item.question,
                    "answer": item.answer,
                    "faithfulness": item.faithfulness,
                    "reference_f1": item.reference_f1,
                    "details": item.details,
                }
                for item in self.answers
            ]
        return payload


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def evaluate_answers(
    pipeline: SupportsQuery,
    examples: Sequence[EvalExample],
    *,
    scorer: FaithfulnessScorer | None = None,
    k: int = 5,
    mode: RetrievalMode = "hybrid",
) -> PipelineReport:
    """Run the pipeline on every question and score the answers.

    ``scorer`` defaults to :class:`TokenOverlapFaithfulness`. When an example
    has a ``reference_answer``, token F1 against it is reported too. Answers
    from a pipeline without an LLM are empty and are not scored.
    """

    judge = scorer or TokenOverlapFaithfulness()
    rows: list[AnswerEvaluation] = []
    for example in examples:
        result = pipeline.query(example.question, k=k, mode=mode)
        answer = str(getattr(result, "answer", ""))
        contexts = [str(item.text) for item in getattr(result, "contexts", ())]
        faithfulness: float | None = None
        details: dict[str, Any] = {}
        if answer.strip():
            verdict = judge.score(example.question, answer, contexts)
            faithfulness = verdict.score
            details = dict(verdict.details)
        reference = (
            token_f1(answer, example.reference_answer)
            if example.reference_answer is not None and answer.strip()
            else None
        )
        rows.append(AnswerEvaluation(example.question, answer, faithfulness, reference, details))
    return PipelineReport(
        answers=tuple(rows),
        mean_faithfulness=_mean([row.faithfulness for row in rows if row.faithfulness is not None]),
        mean_reference_f1=_mean([row.reference_f1 for row in rows if row.reference_f1 is not None]),
    )
