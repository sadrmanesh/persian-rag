"""Integration tests against real Elasticsearch and OpenSearch servers.

Run with ``pytest -m search`` after setting ``PERSIAN_RAG_ES_URL`` and/or
``PERSIAN_RAG_OPENSEARCH_URL`` (security disabled, for example the CI
services).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Iterator

import pytest

from persian_rag.embeddings import HashingEmbedder
from persian_rag.pipeline import build_retriever
from persian_rag.retrieval import HybridRetriever
from persian_rag.vectorstores import ElasticsearchStore, OpenSearchStore

pytestmark = pytest.mark.search

CORPUS = {
    "hafez": "حافظ شیرازی در شهر شیراز زاده شد و دیوان غزلیات او مشهور است.",
    "ferdowsi": "فردوسی شاهنامه را در توس سرود و داستان رستم را روایت کرد.",
    "tehran": "تهران پایتخت ایران است و کوه دماوند در نزدیکی آن قرار دارد.",
}

SERVERS = [
    pytest.param((ElasticsearchStore, "PERSIAN_RAG_ES_URL"), id="elasticsearch"),
    pytest.param((OpenSearchStore, "PERSIAN_RAG_OPENSEARCH_URL"), id="opensearch"),
]


@pytest.fixture(params=SERVERS)
def store(request: pytest.FixtureRequest) -> Iterator[ElasticsearchStore]:
    cls, env = request.param
    url = os.environ.get(env)
    if not url:
        pytest.skip(f"{env} is not set")
    instance = cls(url, f"persian_rag_test_{uuid.uuid4().hex[:8]}", refresh="true")
    yield instance
    instance.drop_index()


def test_dense_sparse_hybrid_and_delete(store: ElasticsearchStore) -> None:
    embedder = HashingEmbedder(dimension=64)
    retriever = build_retriever(
        list(CORPUS.values()), embedder, document_ids=list(CORPUS), store=store
    )
    assert len(store) == 3

    remote = HybridRetriever(embedder, store=store, sparse_backend=store)
    sparse = remote.retrieve("دیوان حافظ", k=1, mode="sparse")
    assert sparse[0].document_id == "hafez"
    dense = remote.retrieve(CORPUS["ferdowsi"], k=1, mode="dense")
    assert dense[0].document_id == "ferdowsi"
    hybrid = remote.retrieve("پایتخت ایران", k=3)
    assert hybrid[0].document_id == "tehran"
    assert hybrid[0].end > hybrid[0].start

    with pytest.raises(ValueError, match="Duplicate"):
        store.add(["hafez:0"], [embedder.embed_query("x")], ["x"])
    retriever.delete_document("hafez")
    assert len(store) == 2
    assert all(item.document_id != "hafez" for item in remote.retrieve("حافظ", k=3))

    reopened = type(store)(store.url, store.index)
    assert reopened.dimension is None
    assert reopened.index_exists()
    assert reopened.dimension == 64
