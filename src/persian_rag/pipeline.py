"""A small retrieve-then-generate pipeline.

The language model is any callable that accepts the filled prompt and returns
a string. persian-rag does not ship a model.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from persian_rag.chunk import Chunk, ChunkingConfig, chunk_text
from persian_rag.embeddings import Embedder
from persian_rag.normalize import NormalizeConfig, normalize
from persian_rag.retrieval import HybridRetriever, RetrievalMode, RetrievalResult
from persian_rag.store import VectorStore

if TYPE_CHECKING:
    from persian_rag.citations import Citation
    from persian_rag.rerank import Reranker
    from persian_rag.rewrite import QueryRewriter

LLM = Callable[[str], str]
AsyncLLM = Callable[[str], Awaitable[str]]

DEFAULT_PROMPT_FA = (
    "تنها با تکیه بر متن‌های زیر به پرسش پاسخ بده. "
    "اگر پاسخ در متن‌ها نیست، صریح بگو که نمی‌دانی.\n\n"
    "متن‌ها:\n{context}\n\n"
    "پرسش: {question}\n\n"
    "پاسخ:"
)

DEFAULT_PROMPT_EN = (
    "Answer the question using only the context below. "
    "If the answer is not contained in the context, say you do not know.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


@dataclass(frozen=True, slots=True)
class RAGAnswer:
    """Result of :meth:`RAGPipeline.query`."""

    question: str
    answer: str
    prompt: str
    contexts: tuple[RetrievalResult, ...]
    citations: tuple[Citation, ...] = ()
    """Resolved ``[n]`` markers, filled when the pipeline has ``citations=True``."""
    uncited: tuple[str, ...] = ()
    """Answer sentences without a citation marker (``citations=True`` only)."""

    def to_dict(self, *, include_prompt: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "question": self.question,
            "answer": self.answer,
            "contexts": [item.to_dict() for item in self.contexts],
            "citations": [item.to_dict() for item in self.citations],
            "uncited": list(self.uncited),
        }
        if include_prompt:
            payload["prompt"] = self.prompt
        return payload


class RAGPipeline:
    """Retrieve context and, when an LLM is set, generate an answer.

    With a ``reranker``, the retriever first fetches ``rerank_candidates``
    chunks (at least ``k``) and the reranker picks the final ``k``.
    ``llm`` may be a plain function or an ``async`` function; async models
    are supported by :meth:`aquery`.

    With ``citations=True`` the default prompt asks the model to cite the
    numbered contexts and :attr:`RAGAnswer.citations` maps each marker to a
    source span (see :mod:`persian_rag.citations`). With a
    ``query_rewriter`` every query variant is retrieved and the rankings are
    fused (see :mod:`persian_rag.rewrite`).
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        llm: LLM | AsyncLLM | None = None,
        *,
        prompt_template: str | None = None,
        context_template: str = "[{n}] {text}",
        separator: str = "\n\n",
        reranker: Reranker | None = None,
        rerank_candidates: int = 20,
        citations: bool = False,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        if prompt_template is None:
            if citations:
                from persian_rag.citations import CITATION_PROMPT_FA

                prompt_template = CITATION_PROMPT_FA
            else:
                prompt_template = DEFAULT_PROMPT_FA
        _require_placeholders(prompt_template)
        if "{text}" not in context_template or "{n}" not in context_template:
            raise ValueError("context_template must contain {n} and {text}")
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template
        self.context_template = context_template
        self.separator = separator
        if rerank_candidates < 1:
            raise ValueError("rerank_candidates must be >= 1")
        self.reranker = reranker
        self.rerank_candidates = rerank_candidates
        self.citations = citations
        self.query_rewriter = query_rewriter

    def build_prompt(self, question: str, contexts: Sequence[RetrievalResult]) -> str:
        """Fill the prompt template. Placeholders are replaced literally."""

        if contexts:
            blocks = [
                self.context_template.format(n=index, text=item.text, id=item.chunk_id)
                for index, item in enumerate(contexts, start=1)
            ]
            context = self.separator.join(blocks)
        else:
            context = "(no retrieved context)"
        # Replace placeholders via sentinels so a context that itself contains
        # "{question}" is left unchanged.
        context_token = "\x00persian_rag_context\x00"
        question_token = "\x00persian_rag_question\x00"
        prepared = self.prompt_template.replace("{context}", context_token).replace(
            "{question}", question_token
        )
        return prepared.replace(context_token, context).replace(question_token, question)

    def query(
        self,
        question: str,
        *,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
    ) -> RAGAnswer:
        """Retrieve ``k`` chunks and call the LLM when one was provided.

        Without an LLM, ``answer`` is an empty string and ``prompt`` still
        contains the retrieved context.
        """

        contexts = self.retrieve(question, k=k, mode=mode)
        prompt = self.build_prompt(question, contexts)
        answer = self.generate(prompt)
        return self.make_answer(question, answer, prompt, contexts)

    def make_answer(
        self,
        question: str,
        answer: str,
        prompt: str,
        contexts: Sequence[RetrievalResult],
    ) -> RAGAnswer:
        """Package a generated answer, resolving citations when enabled."""

        citations: tuple[Citation, ...] = ()
        uncited: tuple[str, ...] = ()
        if self.citations and answer:
            from persian_rag.citations import extract_citations

            report = extract_citations(answer, contexts)
            citations, uncited = report.citations, report.uncited
        return RAGAnswer(
            question=question,
            answer=answer,
            prompt=prompt,
            contexts=tuple(contexts),
            citations=citations,
            uncited=uncited,
        )

    def retrieve(
        self,
        question: str,
        *,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        """Retrieve ``k`` chunks, reranking a larger pool when configured."""

        if self.query_rewriter is not None:
            from persian_rag.rewrite import RewritingRetriever

            search: Any = RewritingRetriever(self.retriever, self.query_rewriter)
        else:
            search = self.retriever
        if self.reranker is None:
            return list(search.retrieve(question, k=k, mode=mode))
        pool = search.retrieve(question, k=max(k, self.rerank_candidates), mode=mode)
        return self.reranker.rerank(question, pool, top_k=k)

    def generate(self, prompt: str) -> str:
        """Call the synchronous LLM; return ``""`` when none is configured."""

        if self.llm is None:
            return ""
        generated = self.llm(prompt)
        if inspect.isawaitable(generated):
            _close_awaitable(generated)
            raise TypeError("The LLM is asynchronous; use aquery() instead of query()")
        return _require_text(generated)

    async def aretrieve(
        self,
        question: str,
        *,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        """Async variant of :meth:`retrieve`."""

        return await asyncio.to_thread(self.retrieve, question, k=k, mode=mode)

    async def aquery(
        self,
        question: str,
        *,
        k: int = 5,
        mode: RetrievalMode = "hybrid",
    ) -> RAGAnswer:
        """Async variant of :meth:`query`.

        Retrieval and reranking run in a worker thread. An ``async`` LLM is
        awaited directly; a synchronous one runs in a worker thread so the
        event loop is never blocked.
        """

        contexts = await self.aretrieve(question, k=k, mode=mode)
        prompt = self.build_prompt(question, contexts)
        if self.llm is None:
            answer = ""
        elif _is_async_callable(self.llm):
            answer = _require_text(await self.llm(prompt))  # type: ignore[misc]
        else:
            generated = await asyncio.to_thread(self.llm, prompt)
            if inspect.isawaitable(generated):
                generated = await generated
            answer = _require_text(generated)
        return self.make_answer(question, answer, prompt, contexts)


def _is_async_callable(func: object) -> bool:
    if inspect.iscoroutinefunction(func):
        return True
    call = getattr(func, "__call__", None)  # noqa: B004
    return inspect.iscoroutinefunction(call)


def _close_awaitable(value: object) -> None:
    close = getattr(value, "close", None)
    if callable(close):
        close()


def _require_text(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("LLM callable must return a string")
    return value


def build_retriever(
    texts: Sequence[str],
    embedder: Embedder,
    *,
    document_ids: Sequence[str] | None = None,
    metadatas: Sequence[Mapping[str, object]] | None = None,
    normalize_config: NormalizeConfig | None = None,
    chunking: ChunkingConfig | None = None,
    store: VectorStore | None = None,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    rrf_k: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> HybridRetriever:
    """Normalize, chunk, and index ``texts``.

    Chunk offsets refer to the normalized text, not the original string.
    """

    if document_ids is None:
        ids = [f"doc-{index}" for index in range(len(texts))]
    else:
        ids = list(document_ids)
    if len(ids) != len(texts):
        raise ValueError("document_ids must have one id per text")
    if len(set(ids)) != len(ids):
        raise ValueError("document ids must be unique")
    if metadatas is not None and len(metadatas) != len(texts):
        raise ValueError("metadatas must have one mapping per text")

    options = normalize_config or NormalizeConfig()
    chunk_options = chunking or ChunkingConfig()
    retriever = HybridRetriever(
        embedder,
        store=store,
        normalize_config=options,
        bm25_k1=bm25_k1,
        bm25_b=bm25_b,
        rrf_k=rrf_k,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
    )
    chunks: list[Chunk] = []
    for index, text in enumerate(texts):
        raw_meta = metadatas[index] if metadatas is not None else {}
        meta: dict[str, Any] = {str(key): value for key, value in raw_meta.items()}
        normalized = normalize(text, options)
        chunks.extend(
            chunk_text(
                normalized,
                chunk_options,
                document_id=ids[index],
                metadata=meta,
            )
        )
    retriever.add_chunks(chunks)
    return retriever


def _require_placeholders(template: str) -> None:
    if "{context}" not in template or "{question}" not in template:
        raise ValueError("prompt template must contain {context} and {question}")
