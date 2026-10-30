"""Citation-grounded answers.

The citation prompts ask the model to mark every claim with the number of
the context block it came from, for example ``[2]`` or ``[۱، ۳]``.
:func:`extract_citations` then maps each marker back to the retrieved chunk
and finds the sentence in that chunk that best supports the claim, with
character offsets in the source document (the normalized text that was
indexed). Sentences without a marker, and markers that point to a context
that does not exist, are reported so callers can flag unsupported claims.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from persian_rag.retrieval import RetrievalResult
from persian_rag.tokenize import split_sentence_spans, tokenize

CITATION_PROMPT_FA = (
    "فقط با تکیه بر متن های شماره دار زیر به پرسش پاسخ بده. "
    "بعد از هر جمله شماره متنی را که آن جمله از آن آمده در کروشه بنویس، "
    "مانند [1] یا [1، 3]. "
    "اگر پاسخ در متن ها نیست، صریح بگو که نمی دانی و شماره ای ننویس.\n\n"
    "متن ها:\n{context}\n\n"
    "پرسش: {question}\n\n"
    "پاسخ:"
)

CITATION_PROMPT_EN = (
    "Answer the question using only the numbered context below. "
    "After every sentence, cite the context it came from in square brackets, "
    "for example [1] or [1, 3]. "
    "If the answer is not in the context, say you do not know and cite nothing.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)

_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")
_NUM = r"[0-9۰-۹٠-٩]+"
_MARKER_RE = re.compile(rf"\[\s*({_NUM}(?:\s*[,،;؛]\s*{_NUM})*)\s*\]")
_LEADING_MARKERS_RE = re.compile(rf"^\s*(?:{_MARKER_RE.pattern}\s*)+")
_STOPWORD_TEXT = (
    "و در به از که این آن را با است برای یک تا هم یا اما نیز بر شد شده می هست بود "
    "the a an of to in and or is are was were be for on with by as at it this that"
)
_STOPWORDS = frozenset(_STOPWORD_TEXT.split())


@dataclass(frozen=True, slots=True)
class Citation:
    """A marker in the answer resolved to a supporting span of a chunk.

    ``start`` and ``end`` are offsets in the indexed (normalized) document
    text; ``chunk_start`` and ``chunk_end`` are offsets inside the chunk.
    ``score`` is the share of the claim's content words found in ``quote``
    (0 to 1); a low score means the cited span does not obviously support
    the claim.
    """

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "quote": self.quote,
            "start": self.start,
            "end": self.end,
            "chunk_start": self.chunk_start,
            "chunk_end": self.chunk_end,
            "score": self.score,
            "claim": self.claim,
            "answer_start": self.answer_start,
            "answer_end": self.answer_end,
        }


@dataclass(frozen=True, slots=True)
class CitationReport:
    """Citations found in an answer and the claims left without one."""

    citations: tuple[Citation, ...]
    uncited: tuple[str, ...]
    invalid_markers: tuple[int, ...]

    @property
    def cited_chunk_ids(self) -> list[str]:
        return list(dict.fromkeys(citation.chunk_id for citation in self.citations))

    @property
    def grounded(self) -> bool:
        """True when every sentence cites a valid context."""

        return bool(self.citations) and not self.uncited and not self.invalid_markers


@dataclass(frozen=True, slots=True)
class _Claim:
    text: str
    start: int
    end: int
    markers: tuple[int, ...]


def parse_markers(text: str) -> list[int]:
    """Return marker numbers in ``text`` in order (Persian digits accepted)."""

    numbers: list[int] = []
    for match in _MARKER_RE.finditer(text):
        for part in re.split(r"[,،;؛]", match.group(1)):
            numbers.append(int(part.strip().translate(_DIGITS)))
    return numbers


def strip_markers(text: str) -> str:
    """Remove citation markers and tidy the spaces they leave."""

    cleaned = _MARKER_RE.sub("", text)
    cleaned = re.sub(r"[ \t]+([.!?؟،,؛;:])", r"\1", cleaned)
    return re.sub(r"[ \t]{2,}", " ", cleaned).strip()


def _claims(answer: str) -> list[_Claim]:
    spans = split_sentence_spans(answer)
    claims: list[_Claim] = []
    for start, end in spans:
        piece = answer[start:end]
        leading = _LEADING_MARKERS_RE.match(piece)
        if leading and claims:
            # Markers placed after the full stop belong to the sentence before.
            previous = claims[-1]
            extra = tuple(parse_markers(leading.group(0)))
            claims[-1] = _Claim(
                previous.text, previous.start, start + leading.end(), previous.markers + extra
            )
            piece = piece[leading.end() :]
            start += leading.end()
        text = strip_markers(piece)
        if not text:
            continue
        claims.append(_Claim(text, start, end, tuple(parse_markers(piece))))
    return claims


def _content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in _STOPWORDS and len(token) > 1}


def best_supporting_span(claim: str, text: str) -> tuple[int, int, float]:
    """Return ``(start, end, score)`` of the sentence in ``text`` closest to ``claim``.

    Adjacent sentence pairs are also considered, for claims that merge two
    sentences. Without any overlap the whole text is returned with score 0.
    """

    wanted = _content_tokens(claim)
    spans = split_sentence_spans(text) or [(0, len(text))]
    if not wanted:
        return spans[0][0], spans[0][1], 0.0
    candidates = list(spans)
    candidates.extend((spans[i][0], spans[i + 1][1]) for i in range(len(spans) - 1))
    best = (0, len(text), 0.0)
    best_length = len(text) + 1
    for start, end in candidates:
        found = _content_tokens(text[start:end])
        score = len(wanted & found) / len(wanted)
        length = end - start
        if score > best[2] or (score == best[2] and score > 0 and length < best_length):
            best = (start, end, score)
            best_length = length
    return best


def extract_citations(answer: str, contexts: Sequence[RetrievalResult]) -> CitationReport:
    """Resolve ``[n]`` markers in ``answer`` against the numbered ``contexts``."""

    citations: list[Citation] = []
    uncited: list[str] = []
    invalid: list[int] = []
    for claim in _claims(answer):
        if not claim.markers:
            if _content_tokens(claim.text):
                uncited.append(claim.text)
            continue
        for marker in dict.fromkeys(claim.markers):
            if not 1 <= marker <= len(contexts):
                invalid.append(marker)
                continue
            context = contexts[marker - 1]
            local_start, local_end, score = best_supporting_span(claim.text, context.text)
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=context.chunk_id,
                    document_id=context.document_id,
                    quote=context.text[local_start:local_end],
                    start=context.start + local_start,
                    end=context.start + local_end,
                    chunk_start=local_start,
                    chunk_end=local_end,
                    score=round(score, 4),
                    claim=claim.text,
                    answer_start=claim.start,
                    answer_end=claim.end,
                )
            )
    return CitationReport(
        citations=tuple(citations),
        uncited=tuple(uncited),
        invalid_markers=tuple(dict.fromkeys(invalid)),
    )


def format_sources(report: CitationReport, *, max_quote: int = 160) -> str:
    """Render cited spans as a numbered source list for display."""

    lines: list[str] = []
    seen: set[tuple[int, int, str]] = set()
    for citation in report.citations:
        key = (citation.marker, citation.start, citation.chunk_id)
        if key in seen:
            continue
        seen.add(key)
        quote = " ".join(citation.quote.split())
        if len(quote) > max_quote:
            quote = quote[: max_quote - 3].rstrip() + "..."
        lines.append(
            f"[{citation.marker}] {citation.document_id} ({citation.start}-{citation.end}): {quote}"
        )
    return "\n".join(lines)
