"""Second-stage rerankers.

A reranker reorders the candidates returned by a first-stage retriever with a
more expensive relevance model. Two implementations ship with the package:

* :class:`CrossEncoderReranker` scores (query, passage) pairs with a
  sentence-transformers cross-encoder. Install ``persian-rag[rerank]``.
* :class:`LLMReranker` asks any text-completion callable to order the
  passages (listwise reranking). It needs no extra dependency.

Both return new :class:`~persian_rag.retrieval.RetrievalResult` objects whose
``score`` is the reranker score and whose ``retrieval_score`` keeps the
first-stage score.
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any, Protocol, runtime_checkable

from persian_rag.errors import PersianRagError
from persian_rag.retrieval import RetrievalResult

DEFAULT_CROSS_ENCODER = "BAAI/bge-reranker-v2-m3"


@runtime_checkable
class Reranker(Protocol):
    """Reorders retrieval results for a query."""

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Return ``results`` best first, truncated to ``top_k`` when given."""


class CrossEncoderModel(Protocol):
    """The part of ``sentence_transformers.CrossEncoder`` used here."""

    def predict(self, sentences: list[tuple[str, str]], **kwargs: Any) -> Any:
        """Return one relevance score per pair."""


def _check_top_k(top_k: int | None) -> None:
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1")


def _apply_scores(
    results: Sequence[RetrievalResult],
    scores: Sequence[float],
    top_k: int | None,
) -> list[RetrievalResult]:
    if len(scores) != len(results):
        raise PersianRagError(f"Reranker returned {len(scores)} scores for {len(results)} results")
    order = sorted(
        range(len(results)),
        key=lambda index: (-scores[index], index),
    )
    ranked = [
        replace(
            results[index],
            score=float(scores[index]),
            retrieval_score=_first_stage(results[index]),
        )
        for index in order
    ]
    return ranked[:top_k] if top_k is not None else ranked


def _first_stage(result: RetrievalResult) -> float:
    return result.retrieval_score if result.retrieval_score is not None else result.score


class CrossEncoderReranker:
    """Rerank with a cross-encoder such as ``BAAI/bge-reranker-v2-m3``.

    The default model is multilingual and handles Persian. Pass ``model`` to
    reuse an already loaded encoder or to inject a stub in tests.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_CROSS_ENCODER,
        *,
        model: CrossEncoderModel | None = None,
        device: str | None = None,
        batch_size: int = 16,
        max_length: int | None = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self.model_name = model_name
        self.batch_size = batch_size
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise ImportError(
                    "sentence-transformers is not installed. "
                    "Install it with: pip install 'persian-rag[rerank]'"
                ) from exc
            model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._model: CrossEncoderModel = model

    def score(self, query: str, passages: Sequence[str]) -> list[float]:
        """Return one relevance score per passage."""

        if not passages:
            return []
        raw = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return [float(value) for value in raw]

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        _check_top_k(top_k)
        if not results:
            return []
        return _apply_scores(results, self.score(query, [r.text for r in results]), top_k)


LLMCallable = Callable[[str], str]

DEFAULT_RERANK_PROMPT = """\
You rank passages by how well they answer a search query. The query and the
passages may be in Persian or English.

Query: {query}

Passages:
{passages}

Return a JSON array with the passage numbers ordered from most to least
relevant, for example [3, 1, 2]. Include every passage number exactly once.
Return only the array."""

_INT_RE = re.compile(r"\d+")


def parse_ranking(text: str, count: int) -> list[int]:
    """Parse an LLM ranking reply into zero-based indexes.

    Accepts a JSON array, an array inside surrounding prose or code fences, or
    a plain list of numbers. Unknown and repeated numbers are ignored, and
    passages the model left out keep their original relative order at the end.
    """

    numbers: list[int] = []
    match = re.search(r"\[[^\[\]]*\]", text)
    if match:
        try:
            decoded = json.loads(match.group(0))
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, list):
            numbers = [
                int(item)
                for item in decoded
                if isinstance(item, (int, str)) and str(item).isdigit()
            ]
    if not numbers:
        numbers = [int(item) for item in _INT_RE.findall(text)]
    seen: set[int] = set()
    order: list[int] = []
    for number in numbers:
        index = number - 1
        if 0 <= index < count and index not in seen:
            seen.add(index)
            order.append(index)
    order.extend(index for index in range(count) if index not in seen)
    return order


class LLMReranker:
    """Listwise reranking with any ``Callable[[str], str]`` language model.

    The model sees the query and numbered passages and returns an ordering.
    Scores are assigned from the final rank (``1.0`` for the best passage,
    falling linearly), so they are comparable only within one query.
    """

    def __init__(
        self,
        llm: LLMCallable,
        *,
        prompt_template: str = DEFAULT_RERANK_PROMPT,
        max_passage_chars: int = 1200,
        max_candidates: int = 20,
    ) -> None:
        if "{query}" not in prompt_template or "{passages}" not in prompt_template:
            raise ValueError("prompt_template must contain {query} and {passages}")
        if max_passage_chars < 1 or max_candidates < 1:
            raise ValueError("max_passage_chars and max_candidates must be >= 1")
        self.llm = llm
        self.prompt_template = prompt_template
        self.max_passage_chars = max_passage_chars
        self.max_candidates = max_candidates

    def build_prompt(self, query: str, passages: Sequence[str]) -> str:
        blocks = []
        for number, passage in enumerate(passages, start=1):
            text = " ".join(passage.split())
            if len(text) > self.max_passage_chars:
                text = text[: self.max_passage_chars].rstrip() + "..."
            blocks.append(f"[{number}] {text}")
        return self.prompt_template.replace("{passages}", "\n".join(blocks)).replace(
            "{query}", query
        )

    def rerank(
        self,
        query: str,
        results: Sequence[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        _check_top_k(top_k)
        if not results:
            return []
        head = list(results[: self.max_candidates])
        tail = list(results[self.max_candidates :])
        reply = self.llm(self.build_prompt(query, [item.text for item in head]))
        if not isinstance(reply, str):
            raise TypeError("LLM callable must return a string")
        order = parse_ranking(reply, len(head)) + [len(head) + i for i in range(len(tail))]
        total = len(order)
        scores = [0.0] * total
        for rank, index in enumerate(order):
            scores[index] = 1.0 - rank / total
        return _apply_scores(head + tail, scores, top_k)


async def arerank(
    reranker: Reranker,
    query: str,
    results: Sequence[RetrievalResult],
    top_k: int | None = None,
) -> list[RetrievalResult]:
    """Run a synchronous reranker in a worker thread."""

    return await asyncio.to_thread(reranker.rerank, query, results, top_k)
