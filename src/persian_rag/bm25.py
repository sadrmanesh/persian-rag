"""Okapi BM25 over in-memory tokenized documents.

The IDF term follows the non-negative Lucene variant:

``log(1 + (N - df + 0.5) / (df + 0.5))``
"""

from __future__ import annotations

import math
from collections.abc import Sequence


class BM25Index:
    """Sparse lexical index. Duplicate document ids are rejected."""

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be > 0")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be between 0 and 1")
        self.k1 = k1
        self.b = b
        self._ids: list[str] = []
        self._positions: dict[str, int] = {}
        self._term_freqs: list[dict[str, int]] = []
        self._lengths: list[int] = []
        self._df: dict[str, int] = {}
        self._avgdl = 0.0

    def __len__(self) -> int:
        return len(self._ids)

    def add(self, ids: Sequence[str], documents: Sequence[Sequence[str]]) -> None:
        """Index tokenized documents aligned with ``ids``."""

        if len(ids) != len(documents):
            raise ValueError("ids and documents must have the same length")
        if len(set(ids)) != len(list(ids)):
            raise ValueError("ids must be unique")
        for doc_id, tokens in zip(ids, documents, strict=True):
            if doc_id in self._positions:
                raise ValueError(f"Duplicate document id: {doc_id}")
            frequencies: dict[str, int] = {}
            for token in tokens:
                frequencies[token] = frequencies.get(token, 0) + 1
            position = len(self._ids)
            self._positions[doc_id] = position
            self._ids.append(doc_id)
            self._term_freqs.append(frequencies)
            self._lengths.append(len(list(tokens)))
            for token in frequencies:
                self._df[token] = self._df.get(token, 0) + 1
        self._recompute_avgdl()

    def search(self, query_tokens: Sequence[str], k: int) -> list[tuple[str, float]]:
        """Return up to ``k`` ``(doc_id, score)`` pairs, best score first."""

        if k < 1:
            raise ValueError("k must be >= 1")
        if not self._ids or not query_tokens:
            return []

        total = len(self._ids)
        scores = [0.0] * total
        average = self._avgdl or 1.0
        for token in dict.fromkeys(query_tokens):
            df = self._df.get(token, 0)
            if df == 0:
                continue
            idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
            for position, frequencies in enumerate(self._term_freqs):
                freq = frequencies.get(token, 0)
                if freq == 0:
                    continue
                length_norm = 1.0 - self.b + self.b * self._lengths[position] / average
                denominator = freq + self.k1 * length_norm
                scores[position] += idf * (freq * (self.k1 + 1.0)) / denominator

        ranked = [
            (self._ids[position], scores[position])
            for position in range(total)
            if scores[position] > 0.0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:k]

    def _recompute_avgdl(self) -> None:
        if not self._lengths:
            self._avgdl = 0.0
            return
        self._avgdl = sum(self._lengths) / len(self._lengths)
