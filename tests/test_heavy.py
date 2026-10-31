"""Model tests.

These stay skipped unless ``PERSIAN_RAG_RUN_HEAVY=1`` is set. They download
weights and are not part of the default suite.
"""

import os

import pytest

RUN_HEAVY = os.environ.get("PERSIAN_RAG_RUN_HEAVY") == "1"


@pytest.mark.heavy
@pytest.mark.skipif(not RUN_HEAVY, reason="Set PERSIAN_RAG_RUN_HEAVY=1 to run heavy model tests")
def test_sentence_transformer_embedder_returns_vectors() -> None:
    pytest.importorskip("sentence_transformers")
    from persian_rag.embeddings import SentenceTransformerEmbedder

    embedder = SentenceTransformerEmbedder()
    assert embedder.model_name == "intfloat/multilingual-e5-small"
    assert (embedder.query_prefix, embedder.passage_prefix) == ("query: ", "passage: ")
    vector = embedder.embed_query("سلام، حال شما چطور است؟")
    batch = embedder.embed_documents(["سلام", "کتاب فارسی"])
    assert len(vector) == embedder.dimension
    assert len(batch) == 2
    assert len(batch[0]) == embedder.dimension


def test_faiss_store_when_extra_is_installed() -> None:
    pytest.importorskip("faiss")
    pytest.importorskip("numpy")
    from persian_rag.store import FaissVectorStore

    store = FaissVectorStore(dimension=2, metric="cosine")
    store.add(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], ["الف", "ب"])
    hits = store.search([0.9, 0.1], k=1)
    assert hits[0].id == "a"
    assert len(store) == 2
