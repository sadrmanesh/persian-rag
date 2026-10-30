from __future__ import annotations

import pytest

from fake_search import FakeSearchCluster
from persian_rag.embeddings import HashingEmbedder
from persian_rag.errors import PersianRagError
from persian_rag.pipeline import build_retriever
from persian_rag.retrieval import HybridRetriever
from persian_rag.vectorstores import ElasticsearchStore, OpenSearchStore
from persian_rag.vectorstores.elasticsearch import store_from_url

CORPUS = {
    "hafez": "حافظ شیرازی در شهر شیراز زاده شد و دیوان غزلیات او مشهور است.",
    "ferdowsi": "فردوسی شاهنامه را در توس سرود و داستان رستم را روایت کرد.",
    "tehran": "تهران پایتخت ایران است و کوه دماوند در نزدیکی آن قرار دارد.",
}


def _retriever(store: ElasticsearchStore) -> HybridRetriever:
    retriever = build_retriever(
        list(CORPUS.values()),
        HashingEmbedder(dimension=64),
        document_ids=list(CORPUS),
        store=store,
    )
    retriever.sparse_backend = store
    return retriever


def test_creates_index_with_persian_analyzer_and_dense_vector() -> None:
    cluster = FakeSearchCluster()
    store = ElasticsearchStore("http://es:9200", "docs", transport=cluster)
    _retriever(store)
    mapping = cluster.indexes["docs"]["body"]["mappings"]["properties"]
    assert mapping["text"]["analyzer"] == "persian"
    assert mapping["embedding"] == {
        "type": "dense_vector",
        "dims": 64,
        "index": True,
        "similarity": "cosine",
    }
    assert len(store) == 3
    assert store.dimension == 64


def test_opensearch_mapping_and_knn_query() -> None:
    cluster = FakeSearchCluster()
    store = OpenSearchStore("http://os:9200", "docs", transport=cluster)
    retriever = _retriever(store)
    body = cluster.indexes["docs"]["body"]
    assert body["settings"] == {"index": {"knn": True}}
    assert body["mappings"]["properties"]["embedding"]["type"] == "knn_vector"
    retriever.retrieve("شاهنامه", k=1, mode="dense")
    searches = [payload for method, path, payload in cluster.calls if path == "docs/_search"]
    assert "knn" in searches[-1]["query"]


def test_hybrid_search_with_server_side_bm25() -> None:
    store = ElasticsearchStore("http://es:9200", "docs", transport=FakeSearchCluster())
    retriever = _retriever(store)
    top = retriever.retrieve("دیوان حافظ", k=1, mode="sparse")[0]
    assert top.document_id == "hafez"
    assert top.text == normalize_text(CORPUS["hafez"])


def normalize_text(text: str) -> str:
    from persian_rag.normalize import normalize

    return normalize(text)


def test_remote_only_retriever_rebuilds_chunks_from_hits() -> None:
    cluster = FakeSearchCluster()
    writer = ElasticsearchStore("http://es:9200", "docs", transport=cluster)
    _retriever(writer)
    reader = ElasticsearchStore("http://es:9200", "docs", transport=cluster)
    retriever = HybridRetriever(HashingEmbedder(dimension=64), store=reader, sparse_backend=reader)
    assert len(retriever) == 0
    results = retriever.retrieve("پایتخت ایران", k=2)
    assert results[0].document_id == "tehran"
    assert results[0].start == 0
    assert not any(key.startswith("_") for key in results[0].metadata)
    assert reader.dimension == 64


def test_duplicate_ids_are_rejected_and_upsert_replaces() -> None:
    cluster = FakeSearchCluster()
    store = ElasticsearchStore("http://es:9200", "docs", transport=cluster)
    store.add(["a"], [[1.0, 0.0]], ["متن"])
    with pytest.raises(ValueError, match="Duplicate"):
        store.add(["a"], [[0.0, 1.0]], ["دیگر"])
    store.upsert(["a"], [[0.0, 1.0]], ["دیگر"])
    assert cluster.indexes["docs"]["docs"]["a"]["text"] == "دیگر"
    with pytest.raises(ValueError, match="dimension"):
        store.upsert(["b"], [[1.0, 0.0, 0.0]], ["x"])


def test_delete_document_removes_remote_chunks() -> None:
    cluster = FakeSearchCluster()
    store = ElasticsearchStore("http://es:9200", "docs", transport=cluster)
    retriever = _retriever(store)
    assert retriever.delete_document("ferdowsi") == 1
    assert "ferdowsi:0" not in cluster.indexes["docs"]["docs"]
    assert all(r.document_id != "ferdowsi" for r in retriever.retrieve("شاهنامه رستم", k=3))


def test_auth_headers(monkeypatch: pytest.MonkeyPatch) -> None:
    cluster = FakeSearchCluster()
    ElasticsearchStore("http://es", "x", api_key="secret", transport=cluster).index_exists()
    assert cluster.headers[-1]["Authorization"] == "ApiKey secret"
    monkeypatch.setenv("OPENSEARCH_PASSWORD", "pw")
    OpenSearchStore("http://os", "x", username="admin", transport=cluster).index_exists()
    assert cluster.headers[-1]["Authorization"].startswith("Basic ")


def test_http_errors_are_reported() -> None:
    def failing(method: str, url: str, body: bytes | None, headers: object) -> object:
        from persian_rag.vectorstores.elasticsearch import HttpResponse

        return HttpResponse(500, {"error": {"reason": "boom"}})

    store = ElasticsearchStore("http://es", "x", transport=failing)  # type: ignore[arg-type]
    with pytest.raises(PersianRagError, match="boom"):
        store.index_exists()


def test_missing_index_without_create() -> None:
    store = ElasticsearchStore("http://es", "x", create_index=False, transport=FakeSearchCluster())
    assert store.search([1.0], k=1) == []
    assert store.search_text("حافظ") == []
    assert len(store) == 0
    with pytest.raises(PersianRagError, match="does not exist"):
        store.add(["a"], [[1.0]], ["t"])


def test_store_from_url() -> None:
    store = store_from_url("opensearch+https://search.local:9200", "docs")
    assert isinstance(store, OpenSearchStore)
    assert store.url == "https://search.local:9200"
    es = store_from_url("elasticsearch://es.local:9200", "docs")
    assert type(es) is ElasticsearchStore
    assert es.url == "http://es.local:9200"


def test_invalid_index_name() -> None:
    with pytest.raises(ValueError):
        ElasticsearchStore("http://es", "Docs")
