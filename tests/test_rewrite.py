from __future__ import annotations

import pytest

from persian_rag.embeddings import HashingEmbedder
from persian_rag.pipeline import RAGPipeline, build_retriever
from persian_rag.retrieval import HybridRetriever, RetrievalResult
from persian_rag.rewrite import (
    QueryRewriter,
    RewriteConfig,
    RewritingRetriever,
    affix_variants,
    orthographic_variants,
)

ZWNJ = "\u200c"


def test_original_query_comes_first() -> None:
    variants = QueryRewriter().variants("  اطاق   خواب ")
    assert variants[0] == "اطاق خواب"
    assert "اتاق خواب" in variants


def test_spelling_groups_are_symmetric() -> None:
    rewriter = QueryRewriter()
    assert "طهران قدیم" in rewriter.variants("تهران قدیم")
    assert "تهران قدیم" in rewriter.variants("طهران قدیم")
    assert set(rewriter.variants("هیئت")) >= {"هیات", "هیأت"}


def test_spelling_matches_whole_words_only() -> None:
    assert QueryRewriter().variants("اتاقک") == ["اتاقک"]


def test_multi_word_entries_win() -> None:
    variants = QueryRewriter().variants("رئیس جمهور ایران")
    assert "رییس جمهور ایران" in variants


def test_orthographic_and_affix_variants() -> None:
    assert "پاییز" in orthographic_variants("پائیز")
    assert "مسوول" in orthographic_variants("مسؤول")
    spaced, joined, attached = affix_variants("کتاب ها را می خوانم")
    assert spaced == "کتاب ها را می خوانم"
    assert joined == f"کتاب{ZWNJ}ها را می{ZWNJ}خوانم"
    assert attached == "کتابها را میخوانم"
    assert affix_variants("سلام دنیا") == []


def test_arabic_variants_are_optional() -> None:
    assert "كتاب" not in " ".join(QueryRewriter().variants("کتاب"))
    rewriter = QueryRewriter(RewriteConfig(arabic_variants=True))
    assert "كتاب" in rewriter.variants("کتاب")


def test_max_variants_and_extra_groups() -> None:
    rewriter = QueryRewriter(RewriteConfig(max_variants=2), extra_groups=[("پایتون", "Python")])
    assert rewriter.variants("آموزش پایتون") == ["آموزش پایتون", "آموزش Python"]
    with pytest.raises(ValueError):
        RewriteConfig(max_variants=0)


def test_llm_rewrites_are_parsed_and_failures_ignored() -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "1. بهترین کتاب های حافظ\n- «دیوان حافظ»\n\n"

    variants = QueryRewriter(llm=llm).variants("کتاب حافظ")
    assert "بهترین کتاب های حافظ" in variants
    assert "دیوان حافظ" in variants
    assert "کتاب حافظ" in prompts[0]

    def broken(prompt: str) -> str:
        raise RuntimeError("down")

    assert QueryRewriter(llm=broken).variants("کتاب حافظ")[0] == "کتاب حافظ"


@pytest.fixture
def spelling_retriever() -> HybridRetriever:
    texts = {
        "room": "اطاق نشیمن خانه قدیمی رو به حیاط باز می شد.",
        "board": "هیات مدیره شرکت جلسه سالانه را برگزار کرد.",
        "sport": "فوتبال ورزشی گروهی است.",
    }
    return build_retriever(
        list(texts.values()), HashingEmbedder(dimension=128), document_ids=list(texts)
    )


def test_rewriting_retriever_finds_alternative_spellings(
    spelling_retriever: HybridRetriever,
) -> None:
    plain = spelling_retriever.retrieve("اتاق نشیمن", k=1, mode="sparse")
    assert plain and plain[0].document_id == "room"
    assert spelling_retriever.retrieve("اتاق", k=3, mode="sparse") == []
    rewriting = RewritingRetriever(spelling_retriever)
    results = rewriting.retrieve("اتاق", k=1, mode="sparse")
    assert results[0].document_id == "room"
    assert "اطاق" in results[0].metadata["matched_queries"]
    assert len(rewriting) == len(spelling_retriever)


def test_pipeline_uses_query_rewriter(spelling_retriever: HybridRetriever) -> None:
    pipeline = RAGPipeline(spelling_retriever, query_rewriter=QueryRewriter())
    assert pipeline.retrieve("هیئت مدیره", k=1, mode="sparse")[0].document_id == "board"


class _FixedRewriter(QueryRewriter):
    def __init__(self, variants: list[str]) -> None:
        super().__init__()
        self._fixed = variants

    def variants(self, query: str) -> list[str]:
        return list(self._fixed)


class _RankingRetriever:
    """Return chunk ``exact`` first for the original query, ``other`` first otherwise."""

    def __init__(self, original: str) -> None:
        self.original = original

    def __len__(self) -> int:
        return 2

    def retrieve(self, query: str, k: int = 5, *, mode: str = "hybrid") -> list[RetrievalResult]:
        order = ["exact", "other"] if query == self.original else ["other", "exact"]
        return [
            RetrievalResult(
                chunk_id=cid, document_id=cid, text=cid, score=1.0, start=0, end=len(cid)
            )
            for cid in order[:k]
        ]


MANY_VARIANTS = ["q", "q1", "q2", "q3", "q4", "q5"]


def test_many_variants_do_not_outvote_the_original_query() -> None:
    retriever = RewritingRetriever(_RankingRetriever("q"), _FixedRewriter(MANY_VARIANTS))
    results = retriever.retrieve("q", k=2)
    assert [item.chunk_id for item in results] == ["exact", "other"]


def test_variant_family_weight_is_fixed() -> None:
    retriever = RewritingRetriever(_RankingRetriever("q"), _FixedRewriter(MANY_VARIANTS))
    weights = retriever.weights(6)
    assert weights[0] == 2.0
    assert sum(weights[1:]) == pytest.approx(1.0)
    assert sum(retriever.weights(3)[1:]) == pytest.approx(1.0)
    assert retriever.weights(1) == [2.0]
    with pytest.raises(ValueError):
        RewritingRetriever(_RankingRetriever("q"), variant_weight=-1.0)


def test_query_contributions_add_up_to_the_score() -> None:
    retriever = RewritingRetriever(_RankingRetriever("q"), _FixedRewriter(MANY_VARIANTS))
    for item in retriever.retrieve("q", k=2):
        contributions = item.metadata["query_contributions"]
        assert set(contributions) == set(item.metadata["matched_queries"])
        assert sum(contributions.values()) == pytest.approx(item.score)
    exact = retriever.retrieve("q", k=1)[0]
    assert exact.metadata["query_contributions"]["q"] == pytest.approx(2.0 / 61)
    assert exact.metadata["query_contributions"]["q1"] == pytest.approx(0.2 / 62)


def test_async_path_uses_the_same_weights() -> None:
    import asyncio

    retriever = RewritingRetriever(_RankingRetriever("q"), _FixedRewriter(MANY_VARIANTS))
    results = asyncio.run(retriever.aretrieve("q", k=2))
    assert results == retriever.retrieve("q", k=2)
    assert results[0].chunk_id == "exact"
