from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient

from persian_rag.pipeline import RAGPipeline
from persian_rag.retrieval import HybridRetriever
from persian_rag.server import create_app


def test_health_retrieve_and_query(corpus_retriever: HybridRetriever) -> None:
    async def llm(prompt: str) -> str:
        return "شیراز"

    client = TestClient(create_app(RAGPipeline(corpus_retriever, llm), max_k=10))
    health = client.get("/health").json()
    assert health["status"] == "ok"
    assert health["chunks"] == 4
    assert health["llm"] is True and health["reranker"] is False

    found = client.post("/retrieve", json={"query": "حافظ شیراز", "k": 2}).json()
    assert found["results"][0]["document_id"] == "hafez"
    assert len(found["results"]) == 2

    answer = client.post(
        "/query", json={"question": "حافظ کجا زاده شد؟", "k": 1, "include_prompt": True}
    ).json()
    assert answer["answer"] == "شیراز"
    assert "حافظ" in answer["prompt"]
    assert client.post("/query", json={"question": "حافظ"}).json()["prompt"] is None


def test_validation_errors(corpus_retriever: HybridRetriever) -> None:
    client = TestClient(create_app(RAGPipeline(corpus_retriever), max_k=3))
    assert client.post("/retrieve", json={"query": ""}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "k": 5}).status_code == 422
    assert client.post("/retrieve", json={"query": "x", "mode": "fuzzy"}).status_code == 422
    assert client.post("/query", json={"question": "حافظ", "k": 2}).json()["answer"] == ""
    with pytest.raises(ValueError):
        create_app(RAGPipeline(corpus_retriever), max_k=0)
