"""HTTP API for a :class:`~persian_rag.pipeline.RAGPipeline`.

Install with ``pip install 'persian-rag[server]'`` and run either
``persian-rag serve --index .persian-rag`` or, from code::

    import uvicorn
    from persian_rag.server import create_app

    uvicorn.run(create_app(pipeline), host="127.0.0.1", port=8000)

Endpoints:

* ``GET /health``: liveness and index size.
* ``POST /retrieve``: ranked chunks for a query.
* ``POST /query``: retrieved chunks plus a generated answer when the
  pipeline has an LLM.

Handlers are ``async`` and use :meth:`RAGPipeline.aquery`, so a slow model
call does not block other requests.
"""

from __future__ import annotations

from typing import Any, Literal

from persian_rag.errors import MissingDependencyError
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


class QueryResponse(BaseModel):
    question: str
    answer: str
    contexts: list[Context]
    prompt: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    chunks: int
    llm: bool
    reranker: bool


def create_app(
    pipeline: RAGPipeline,
    *,
    title: str = "persian-rag",
    max_k: int = 50,
) -> FastAPI:
    """Build a FastAPI application that serves ``pipeline``."""

    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    app = FastAPI(title=title, version=__version__)

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
            prompt=answer.prompt if request.include_prompt else None,
        )

    return app
