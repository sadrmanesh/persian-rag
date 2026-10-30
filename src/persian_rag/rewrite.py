"""Persian query rewriting and expansion.

Persian text in the wild mixes spellings that normalization alone does not
unify: ``اطاق`` and ``اتاق``, ``هیئت`` and ``هیات``, ``مسئله`` and ``مساله``,
``آیین`` and ``آئین``, ``می شود`` with a space, a half-space, or no space.
Search servers that index raw text also need the Arabic ``ي`` and ``ك``
forms that :func:`persian_rag.normalize.normalize` folds away locally.

:class:`QueryRewriter` produces a short, ordered list of query variants:

1. the original query;
2. spelling variants from a built-in table (extendable);
3. orthographic variants: hamza seat, alef madda, ``ئی``/``یی``, and the
   spacing of ``می``, ``ها``, ``تر``, and ``ترین``;
4. optionally, Arabic character variants for external indexes;
5. optionally, rewrites from a language model.

:class:`RewritingRetriever` runs every variant and fuses the rankings with
reciprocal rank fusion, so a document that uses any spelling can be found.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from persian_rag.retrieval import RetrievalMode, RetrievalResult, reciprocal_rank_fusion

LLM = Callable[[str], str]

#: Groups of interchangeable spellings. Each group is searched as a whole.
DEFAULT_SPELLING_GROUPS: tuple[tuple[str, ...], ...] = (
    ("اتاق", "اطاق"),
    ("تهران", "طهران"),
    ("هیئت", "هیات", "هیأت"),
    ("مسئله", "مساله", "مسأله"),
    ("مسئول", "مسوول", "مسؤول"),
    ("مسئولیت", "مسوولیت", "مسؤولیت"),
    ("رئیس", "رییس"),
    ("رئیس جمهور", "رییس جمهور", "رئیس جمهوری"),
    ("آیین", "آئین"),
    ("آیینه", "آئینه", "آینه"),
    ("پاییز", "پائیز"),
    ("پایین", "پائین"),
    ("جزئی", "جزیی"),
    ("جزئیات", "جزییات"),
    ("ابتدایی", "ابتدائی"),
    ("نهایی", "نهائی"),
    ("دایره", "دائره"),
    ("دایم", "دائم"),
    ("سئوال", "سوال", "سؤال"),
    ("مؤسسه", "موسسه"),
    ("مؤثر", "موثر"),
    ("مؤلف", "مولف"),
    ("تأثیر", "تاثیر"),
    ("تأمین", "تامین"),
    ("تأیید", "تایید"),
    ("مأموریت", "ماموریت"),
    ("شأن", "شان"),
    ("قرآن", "قران"),
    ("ذغال", "زغال"),
    ("باطری", "باتری"),
    ("بلیط", "بلیت"),
    ("غلطیدن", "غلتیدن"),
    ("اسطبل", "اصطبل"),
    ("امپراتور", "امپراطور"),
    ("امپراتوری", "امپراطوری"),
    ("تپش", "طپش"),
    ("ایمیل", "ای میل", "رایانامه"),
    ("کامپیوتر", "رایانه"),
)

_PREFIXES = ("نمی", "می")
_SUFFIXES = ("هایی", "های", "ها", "ترین", "تر")
_WORD_RE = re.compile(r"[\u0600-\u06ff\u200c]+|[^\u0600-\u06ff\u200c\s]+")
_ARABIC_VARIANTS = str.maketrans({"ی": "ي", "ک": "ك"})
_ZWNJ = "\u200c"

REWRITE_PROMPT = (
    "You rewrite search queries for a Persian document collection.\n"
    "Write up to {n} alternative Persian queries that ask for the same thing using "
    "synonyms or common alternative spellings. Keep names and numbers unchanged. "
    "Reply with one query per line and nothing else.\n\n"
    "Query: {query}"
)

_LIST_MARK_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)]|[۰-۹]+[.)])\s*")


@dataclass(frozen=True, slots=True)
class RewriteConfig:
    """Options for :class:`QueryRewriter`.

    ``max_variants`` counts the original query. ``arabic_variants`` adds a
    copy with Arabic ``ي`` and ``ك``; enable it only for indexes that store
    un-normalized text (local retrieval normalizes both sides).
    """

    max_variants: int = 6
    spelling: bool = True
    orthography: bool = True
    affix_spacing: bool = True
    arabic_variants: bool = False
    llm_variants: int = 3

    def __post_init__(self) -> None:
        if self.max_variants < 1:
            raise ValueError("max_variants must be >= 1")
        if self.llm_variants < 0:
            raise ValueError("llm_variants must be >= 0")


@dataclass(slots=True)
class QueryRewriter:
    """Generate spelling and orthographic variants of a Persian query.

    ``extra_groups`` adds domain spellings (brand names, transliterations).
    ``llm`` is any ``Callable[[str], str]``; its lines are appended after the
    rule-based variants. Failures of the model are ignored so search never
    breaks because of the rewriter.
    """

    config: RewriteConfig = field(default_factory=RewriteConfig)
    extra_groups: Sequence[Sequence[str]] = ()
    llm: LLM | None = None
    prompt_template: str = REWRITE_PROMPT
    _lookup: dict[str, tuple[str, ...]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if "{query}" not in self.prompt_template:
            raise ValueError("prompt_template must contain {query}")
        self._lookup = {}
        groups: Iterable[Sequence[str]] = (
            (*DEFAULT_SPELLING_GROUPS, *self.extra_groups)
            if self.config.spelling
            else self.extra_groups
        )
        for group in groups:
            members = tuple(dict.fromkeys(item.strip() for item in group if item.strip()))
            for member in members:
                existing = self._lookup.get(member, ())
                self._lookup[member] = tuple(dict.fromkeys((*existing, *members)))

    def variants(self, query: str) -> list[str]:
        """Return the original query followed by distinct rewrites."""

        original = " ".join(query.split())
        if not original:
            return []
        found: list[str] = [original]
        limit = self.config.max_variants

        def push(candidate: str) -> None:
            candidate = " ".join(candidate.split())
            if candidate and candidate not in found and len(found) < limit:
                found.append(candidate)

        if self._lookup:
            for candidate in self._spelling_variants(original):
                push(candidate)
        if self.config.orthography:
            for candidate in orthographic_variants(original):
                push(candidate)
        if self.config.affix_spacing:
            for candidate in affix_variants(original):
                push(candidate)
        if self.config.arabic_variants:
            push(original.translate(_ARABIC_VARIANTS))
        if self.llm is not None and self.config.llm_variants:
            for candidate in self._llm_variants(original):
                push(candidate)
        return found

    __call__ = variants

    def _spelling_variants(self, query: str) -> list[str]:
        out: list[str] = []
        # Longest keys first so multi-word entries win over their parts.
        for key in sorted(self._lookup, key=len, reverse=True):
            pattern = rf"(?<![\w\u200c]){re.escape(key)}(?![\w\u200c])"
            if not re.search(pattern, query):
                continue
            for alternative in self._lookup[key]:
                if alternative != key:
                    out.append(re.sub(pattern, alternative, query))
        return out

    def _llm_variants(self, query: str) -> list[str]:
        assert self.llm is not None
        prompt = self.prompt_template.replace("{n}", str(self.config.llm_variants)).replace(
            "{query}", query
        )
        try:
            reply = self.llm(prompt)
        except Exception:
            return []
        if not isinstance(reply, str):
            return []
        lines = []
        for line in reply.splitlines():
            cleaned = _LIST_MARK_RE.sub("", line).strip().strip("\"'«»")
            if cleaned and len(cleaned) <= 4 * len(query) + 40:
                lines.append(cleaned)
        return lines[: self.config.llm_variants]


def orthographic_variants(query: str) -> list[str]:
    """Hamza, alef madda, and ``ئی``/``یی`` variants of ``query``.

    A lone hamza seat (``ئ`` before a letter other than ``ی``) is left alone
    because replacing it usually produces a misspelling; the spelling table
    covers the common cases such as ``هیئت`` and ``مسئله``.
    """

    rules: tuple[tuple[str, str], ...] = (
        ("ئی", "یی"),
        ("یی", "ئی"),
        ("أ", "ا"),
        ("ؤ", "و"),
        ("آ", "ا"),
        ("ۀ", "ه"),
        ("ة", "ه"),
    )
    out: list[str] = []
    for source, target in rules:
        if source in query:
            out.append(query.replace(source, target))
    return out


def affix_variants(query: str) -> list[str]:
    """Variants with ``می``/``نمی`` and plural or comparative suffixes spaced,
    joined with a half-space, or attached."""

    words = query.split()
    spaced, joined, attached = [], [], []
    changed = False
    index = 0
    while index < len(words):
        word = words[index]
        nxt = words[index + 1] if index + 1 < len(words) else None
        if word in _PREFIXES and nxt is not None and _is_persian(nxt):
            spaced.append(f"{word} {nxt}")
            joined.append(f"{word}{_ZWNJ}{nxt}")
            attached.append(f"{word}{nxt}")
            changed = True
            index += 2
            continue
        if nxt in _SUFFIXES and _is_persian(word):
            spaced.append(f"{word} {nxt}")
            joined.append(f"{word}{_ZWNJ}{nxt}")
            attached.append(f"{word}{nxt}")
            changed = True
            index += 2
            continue
        if _ZWNJ in word:
            spaced.append(word.replace(_ZWNJ, " "))
            joined.append(word)
            attached.append(word.replace(_ZWNJ, ""))
            changed = True
            index += 1
            continue
        spaced.append(word)
        joined.append(word)
        attached.append(word)
        index += 1
    if not changed:
        return []
    return [" ".join(spaced), " ".join(joined), " ".join(attached)]


def _is_persian(word: str) -> bool:
    return bool(word) and "\u0600" <= word[0] <= "\u06ff"


class RewritingRetriever:
    """Retrieve with every query variant and fuse the rankings.

    Wraps any object with ``retrieve(query, k, mode=...)`` such as
    :class:`~persian_rag.retrieval.HybridRetriever`. The original query gets
    ``original_weight`` in the fusion; variants get weight 1. Results keep
    the text and position of the underlying chunks; ``score`` is the fused
    RRF score and ``metadata["matched_queries"]`` lists the variants that
    found the chunk.
    """

    def __init__(
        self,
        retriever: object,
        rewriter: QueryRewriter | None = None,
        *,
        rrf_k: int = 60,
        original_weight: float = 2.0,
        candidate_multiplier: int = 2,
    ) -> None:
        if not callable(getattr(retriever, "retrieve", None)):
            raise TypeError("retriever must have a retrieve() method")
        if original_weight <= 0:
            raise ValueError("original_weight must be > 0")
        if candidate_multiplier < 1:
            raise ValueError("candidate_multiplier must be >= 1")
        self.retriever = retriever
        self.rewriter = rewriter or QueryRewriter()
        self.rrf_k = rrf_k
        self.original_weight = original_weight
        self.candidate_multiplier = candidate_multiplier

    def __len__(self) -> int:
        return len(self.retriever)  # type: ignore[arg-type]

    def __getattr__(self, name: str) -> object:
        # Delegate attributes such as ``chunks`` or ``normalize_config``.
        if name == "retriever":
            raise AttributeError(name)
        return getattr(self.retriever, name)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        variants = self.rewriter.variants(query) or [query]
        retrieve = self.retriever.retrieve  # type: ignore[attr-defined]
        if len(variants) == 1:
            return list(retrieve(query, k, mode=mode))
        pool = k * self.candidate_multiplier
        rankings: list[list[str]] = []
        by_id: dict[str, RetrievalResult] = {}
        matched: dict[str, list[str]] = {}
        for variant in variants:
            results: list[RetrievalResult] = retrieve(variant, pool, mode=mode)
            rankings.append([item.chunk_id for item in results])
            for item in results:
                by_id.setdefault(item.chunk_id, item)
                matched.setdefault(item.chunk_id, []).append(variant)
        weights = [self.original_weight] + [1.0] * (len(variants) - 1)
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=weights)
        output: list[RetrievalResult] = []
        for chunk_id, score in fused[:k]:
            item = by_id[chunk_id]
            metadata = {**item.metadata, "matched_queries": matched[chunk_id]}
            output.append(
                RetrievalResult(
                    chunk_id=item.chunk_id,
                    document_id=item.document_id,
                    text=item.text,
                    score=score,
                    start=item.start,
                    end=item.end,
                    metadata=metadata,
                    dense_rank=item.dense_rank,
                    sparse_rank=item.sparse_rank,
                    retrieval_score=item.retrieval_score,
                )
            )
        return output

    async def aretrieve(
        self,
        query: str,
        k: int = 5,
        *,
        mode: RetrievalMode = "hybrid",
    ) -> list[RetrievalResult]:
        import asyncio

        return await asyncio.to_thread(self.retrieve, query, k, mode=mode)


def spelling_groups_from_mapping(mapping: Mapping[str, Sequence[str]]) -> list[tuple[str, ...]]:
    """Turn ``{"word": ["variant", ...]}`` into groups for ``extra_groups``."""

    return [(key, *values) for key, values in mapping.items()]
