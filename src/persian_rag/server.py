"""HTTP API for a :class:`~persian_rag.pipeline.RAGPipeline`.

Install with ``pip install 'persian-rag[server]'`` and run either
``persian-rag serve --index .persian-rag`` or, from code::

    import uvicorn
    from persian_rag.server import create_app

    uvicorn.run(create_app(pipeline), host="127.0.0.1", port=8000)

Endpoints:

* ``GET /health``: liveness, index size, and the embedding model in use.
* ``POST /retrieve``: ranked chunks for a query.
* ``POST /query``: retrieved chunks plus a generated answer when the
  pipeline has an LLM, with citations when the pipeline has
  ``citations=True``.
* ``POST /chat``: one turn of a conversation. Send ``session_id`` from the
  previous reply to continue it. Sessions live in memory; the least
  recently used are dropped beyond ``max_sessions``.

Handlers are ``async`` and use :meth:`RAGPipeline.aquery`, so a slow model
call does not block other requests.
"""

from __future__ import annotations

import secrets
from collections import OrderedDict
from typing import Any, Literal

from persian_rag.chat import ChatSession, ConversationMemory
from persian_rag.embeddings import EmbedderSpec
from persian_rag.errors import MissingDependencyError, PersianRagError
from persian_rag.pipeline import RAGPipeline
from persian_rag.version import __version__

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise MissingDependencyError(
        "FastAPI is not installed. Install it with: pip install 'persian-rag[server]'"
    ) from exc


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    k: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    include_prompt: bool = False


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)
    k: int = Field(default=5, ge=1, le=100)
    mode: Literal["hybrid", "dense", "sparse"] = "hybrid"
    reset: bool = False


class Context(BaseModel):
    """One retrieved chunk; same shape as ``RetrievalResult.to_dict()``."""

    id: str
    document_id: str
    text: str
    score: float
    start: int
    end: int
    metadata: dict[str, Any]
    dense_rank: int | None = None
    sparse_rank: int | None = None
    retrieval_score: float | None = None


class RetrieveResponse(BaseModel):
    query: str
    results: list[Context]


class CitationModel(BaseModel):
    marker: int
    chunk_id: str
    document_id: str
    quote: str
    start: int
    end: int
    chunk_start: int
    chunk_end: int
    score: float
    claim: str
    answer_start: int
    answer_end: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[Context]
    citations: list[CitationModel] = []
    uncited: list[str] = []
    prompt: str | None = None


class ChatResponse(QueryResponse):
    session_id: str
    standalone_question: str
    turn: int


class EmbedderInfo(BaseModel):
    kind: str | None = None
    model: str | None = None
    dimension: int | None = None
    query_prefix: str | None = None
    passage_prefix: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks: int
    llm: bool
    reranker: bool
    embedder: EmbedderInfo


def create_app(
    pipeline: RAGPipeline,
    *,
    title: str = "persian-rag",
    max_k: int = 50,
    max_sessions: int = 1000,
    max_turns: int = 6,
) -> FastAPI:
    """Build a FastAPI application that serves ``pipeline``."""

    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    if max_sessions < 1:
        raise ValueError("max_sessions must be >= 1")
    sessions: OrderedDict[str, ChatSession] = OrderedDict()

    def session_for(session_id: str | None) -> tuple[str, ChatSession]:
        if session_id and session_id in sessions:
            sessions.move_to_end(session_id)
            return session_id, sessions[session_id]
        new_id = session_id or secrets.token_urlsafe(12)
        session = ChatSession(pipeline, memory=ConversationMemory(max_turns))
        sessions[new_id] = session
        while len(sessions) > max_sessions:
            sessions.popitem(last=False)
        return new_id, session

    app = FastAPI(title=title, version=__version__)
    embedder_info = _embedder_info(pipeline.retriever.embedder)

    def check_k(k: int) -> None:
        if k > max_k:
            raise HTTPException(status_code=422, detail=f"k must be <= {max_k}")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            version=__version__,
            chunks=len(pipeline.retriever),
            llm=pipeline.llm is not None,
            reranker=pipeline.reranker is not None,
            embedder=embedder_info,
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
        check_k(request.k)
        results = await pipeline.aretrieve(request.query, k=request.k, mode=request.mode)
        return RetrieveResponse(
            query=request.query,
            results=[Context(**item.to_dict()) for item in results],
        )

    @app.post("/query", response_model=QueryResponse)
    async def query(request: QueryRequest) -> QueryResponse:
        check_k(request.k)
        answer = await pipeline.aquery(request.question, k=request.k, mode=request.mode)
        return QueryResponse(
            question=answer.question,
            answer=answer.answer,
            contexts=[Context(**item.to_dict()) for item in answer.contexts],
            citations=[CitationModel(**item.to_dict()) for item in answer.citations],
            uncited=list(answer.uncited),
            prompt=answer.prompt if request.include_prompt else None,
        )

    @app.post("/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        check_k(request.k)
        session_id, session = session_for(request.session_id)
        if request.reset:
            session.reset()
        reply = await session.aask(request.message, k=request.k, mode=request.mode)
        answer = reply.answer
        return ChatResponse(
            session_id=session_id,
            standalone_question=reply.standalone_question,
            turn=reply.turn,
            question=answer.question,
            answer=answer.answer,
            contexts=[Context(**item.to_dict()) for item in answer.contexts],
            citations=[CitationModel(**item.to_dict()) for item in answer.citations],
            uncited=list(answer.uncited),
        )

    @app.delete("/chat/{session_id}")
    async def end_chat(session_id: str) -> dict[str, bool]:
        return {"deleted": sessions.pop(session_id, None) is not None}

    return app


def _embedder_info(embedder: object) -> EmbedderInfo:
    """Describe the query embedder for ``/health``. Never includes API keys."""

    to_spec = getattr(embedder, "to_spec", None)
    if not callable(to_spec):
        return EmbedderInfo(kind=type(embedder).__name__)
    try:
        spec = to_spec()
    except (AttributeError, TypeError, ValueError, PersianRagError):
        return EmbedderInfo(kind=type(embedder).__name__)
    if not isinstance(spec, EmbedderSpec):
        return EmbedderInfo(kind=type(embedder).__name__)
    return EmbedderInfo(
        kind=spec.kind,
        model=spec.model,
        dimension=spec.dimension,
        query_prefix=spec.query_prefix,
        passage_prefix=spec.passage_prefix,
    )
