from __future__ import annotations

from persian_rag.citations import (
    CITATION_PROMPT_EN,
    CITATION_PROMPT_FA,
    best_supporting_span,
    extract_citations,
    format_sources,
    parse_markers,
    strip_markers,
)
from persian_rag.pipeline import RAGPipeline
from persian_rag.retrieval import HybridRetriever, RetrievalResult

CONTEXTS = [
    RetrievalResult(
        "hafez:0", "hafez", "حافظ شاعر قرن هشتم بود. او در شیراز زندگی می کرد.", 1.0, 100, 150
    ),
    RetrievalResult(
        "ferdowsi:0", "ferdowsi", "فردوسی شاهنامه را سرود. شاهنامه حماسه ملی است.", 0.9, 0, 47
    ),
]


def test_parse_and_strip_markers() -> None:
    assert parse_markers("الف [1] ب [۲، 3] ج [٤]") == [1, 2, 3, 4]
    assert strip_markers("حافظ شاعر بود [1].") == "حافظ شاعر بود."


def test_markers_resolve_to_document_offsets() -> None:
    report = extract_citations("فردوسی شاهنامه را سرود [2].", CONTEXTS)
    (citation,) = report.citations
    assert citation.chunk_id == "ferdowsi:0"
    assert citation.quote == "فردوسی شاهنامه را سرود."
    assert (citation.start, citation.end) == (0, len(citation.quote))
    assert citation.score == 1.0
    assert report.grounded


def test_offsets_include_chunk_start() -> None:
    report = extract_citations("حافظ در شیراز زندگی می کرد [۱].", CONTEXTS)
    citation = report.citations[0]
    assert citation.start == 100 + citation.chunk_start
    assert CONTEXTS[0].text[citation.chunk_start : citation.chunk_end] == citation.quote
    assert "شیراز" in citation.quote


def test_marker_after_full_stop_belongs_to_previous_sentence() -> None:
    report = extract_citations("فردوسی شاهنامه را سرود. [2] حافظ شاعر بود [1].", CONTEXTS)
    assert [c.marker for c in report.citations] == [2, 1]
    assert report.citations[0].claim == "فردوسی شاهنامه را سرود."


def test_uncited_and_invalid_markers() -> None:
    report = extract_citations("حافظ شاعر بود [1]. این جمله منبع ندارد. غلط است [7].", CONTEXTS)
    assert report.uncited == ("این جمله منبع ندارد.",)
    assert report.invalid_markers == (7,)
    assert not report.grounded
    assert report.cited_chunk_ids == ["hafez:0"]


def test_multiple_markers_in_one_sentence() -> None:
    report = extract_citations("حافظ و فردوسی شاعر بودند [1, 2].", CONTEXTS)
    assert [c.document_id for c in report.citations] == ["hafez", "ferdowsi"]


def test_best_span_without_overlap() -> None:
    assert best_supporting_span("چیزی دیگر", "متن کاملا بی ربط.")[2] == 0.0


def test_format_sources() -> None:
    report = extract_citations("فردوسی شاهنامه را سرود [2].", CONTEXTS)
    assert format_sources(report) == "[2] ferdowsi (0-23): فردوسی شاهنامه را سرود."


def test_prompts_have_placeholders_and_no_zwnj() -> None:
    for prompt in (CITATION_PROMPT_FA, CITATION_PROMPT_EN):
        assert "{context}" in prompt and "{question}" in prompt
        assert "\u200c" not in prompt


def test_pipeline_with_citations(corpus_retriever: HybridRetriever) -> None:
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return "فردوسی شاهنامه را در توس سرود [1]."

    pipeline = RAGPipeline(corpus_retriever, llm, citations=True)
    answer = pipeline.query("شاهنامه را چه کسی سرود؟", k=2)
    assert "کروشه" in prompts[0]
    assert answer.citations[0].document_id == "ferdowsi"
    assert answer.uncited == ()
    payload = answer.to_dict()
    assert payload["citations"][0]["marker"] == 1


def test_pipeline_without_citations_leaves_fields_empty(
    corpus_retriever: HybridRetriever,
) -> None:
    pipeline = RAGPipeline(corpus_retriever, lambda prompt: "پاسخ [1].")
    answer = pipeline.query("شاهنامه", k=1)
    assert answer.citations == ()
