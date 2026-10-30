import pytest

from persian_rag.bm25 import BM25Index
from persian_rag.chunk import Chunk
from persian_rag.embeddings import HashingEmbedder
from persian_rag.normalize import NormalizeConfig
from persian_rag.retrieval import HybridRetriever, reciprocal_rank_fusion
from persian_rag.store import InMemoryVectorStore


def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(id=chunk_id, text=text, start=0, end=len(text), index=0, document_id=chunk_id)


def test_bm25_ranks_the_document_that_contains_the_term() -> None:
    index = BM25Index()
    index.add(
        ["cats", "dogs"],
        [["گربه", "گربه", "گربه"], ["گربه", "سگ"]],
    )
    dogs = index.search(["سگ"], k=2)
    assert dogs[0][0] == "dogs"
    cats = index.search(["گربه"], k=2)
    assert cats[0][0] == "cats"
    assert index.search(["فیل"], k=2) == []
    assert index.search([], k=1) == []


def test_reciprocal_rank_fusion_scores_and_tie_break() -> None:
    fused = reciprocal_rank_fusion([["d1", "d2"], ["d2", "d1"]], k=60)
    assert fused[0][0] == "d1"
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    weighted = reciprocal_rank_fusion([["a"], ["b", "a"]], k=60, weights=[0.0, 1.0])
    assert [item[0] for item in weighted] == ["b", "a"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)


def test_memory_store_uses_cosine_and_rejects_duplicates() -> None:
    store = InMemoryVectorStore()
    store.add(["b", "a"], [[0.0, 1.0], [1.0, 0.0]], ["bee", "ay"])
    hits = store.search([1.0, 0.0], k=2)
    assert [hit.id for hit in hits] == ["a", "b"]
    assert hits[0].score == pytest.approx(1.0)
    tied = InMemoryVectorStore()
    tied.add(["b", "a"], [[1.0, 0.0], [1.0, 0.0]], ["b", "a"])
    assert [hit.id for hit in tied.search([1.0, 0.0], k=2)] == ["a", "b"]
    with pytest.raises(ValueError):
        store.add(["a"], [[1.0, 0.0]], ["again"])
    assert store.search([0.0, 0.0], k=1)[0].score == 0.0
    assert InMemoryVectorStore().search([1.0], k=1) == []


def test_hybrid_retrieval_finds_the_persian_passage() -> None:
    hafez = "حافظ در شهر شیراز زاده شد و دیوان او مشهور است."
    sport = "فوتبال ورزشی گروهی با دو تیم است."
    retriever = HybridRetriever(HashingEmbedder(dimension=64))
    retriever.add_chunks([_chunk("hafez", hafez), _chunk("sport", sport)])
    hybrid = retriever.retrieve("حافظ در کدام شهر زاده شد؟", k=2)
    sparse = retriever.retrieve("حافظ", k=1, mode="sparse")
    dense = retriever.retrieve(hafez, k=1, mode="dense")
    assert hybrid[0].chunk_id == "hafez"
    assert hybrid[0].sparse_rank == 1
    assert sparse[0].chunk_id == "hafez"
    assert dense[0].chunk_id == "hafez"
    assert dense[0].score == pytest.approx(1.0)


def test_arabic_yeh_query_matches_persian_text() -> None:
    retriever = HybridRetriever(
        HashingEmbedder(dimension=32),
        normalize_config=NormalizeConfig(),
    )
    retriever.add_chunks([_chunk("book", "یک کتاب فارسی روی میز است.")])
    hits = retriever.retrieve("يك كتاب", k=1, mode="sparse")
    assert hits[0].chunk_id == "book"


def test_duplicate_chunk_id_is_rejected() -> None:
    retriever = HybridRetriever(HashingEmbedder(dimension=8))
    chunk = _chunk("same", "سلام")
    retriever.add_chunks([chunk])
    with pytest.raises(ValueError):
        retriever.add_chunks([chunk])
    assert retriever.retrieve("سلام", k=1)[0].chunk_id == "same"
