"""Persian-aware tokenization and sentence splitting."""

from __future__ import annotations

import re

from persian_rag.normalize import NormalizeConfig, normalize

# Letters only. Digits, diacritics, and Arabic punctuation stay out of words
# so "سلام،" does not become a single token.
_TOKEN_RE = re.compile(
    r"[0-9۰-۹٠-٩]+(?:[.\u066b][0-9۰-۹٠-٩]+)?|"
    r"[A-Za-z]+(?:'[A-Za-z]+)?|"
    r"[\u0621-\u063A\u0641-\u064A\u066E\u0671-\u06D3\u06FA-\u06FF\u200c]+"
)

_SENTENCE_END = frozenset(".!?؟…۔")
_CLOSERS = frozenset("\"'»”’)】」›»)]}>")
_KNOWN_ABBREVIATIONS = frozenset(
    {
        "دکتر",
        "مهندس",
        "استاد",
        "آقا",
        "خانم",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "vs",
    }
)


def tokenize(
    text: str,
    *,
    normalize_text: bool = True,
    config: NormalizeConfig | None = None,
) -> list[str]:
    """Split ``text`` into word and number tokens.

    Half-spaces (ZWNJ) stay inside a token, so ``می‌روم`` is one token.
    Latin tokens are lowercased. Persian characters are unchanged.
    """

    prepared = normalize(text, config) if normalize_text else text
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(prepared):
        token = match.strip("\u200c")
        if not token:
            continue
        if token.isascii():
            token = token.lower()
        tokens.append(token)
    return tokens


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences.

    Offsets are not returned. Use :func:`split_sentence_spans` when the
    sentences must line up with the original string. Callers that need
    normalized text should normalize before splitting; this function does
    not rewrite characters, so spans stay valid.
    """

    return [text[start:end] for start, end in split_sentence_spans(text)]


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` spans of sentences in ``text``."""

    if not text or not text.strip():
        return []

    folded, index_map = _fold_newlines(text)
    spans: list[tuple[int, int]] = []
    for start, end in _sentence_spans(folded):
        original_start = index_map[start]
        original_end = index_map[end - 1] + 1
        spans.append((original_start, original_end))
    return spans


def _fold_newlines(text: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    position = 0
    length = len(text)
    while position < length:
        if text[position] == "\r":
            chars.append("\n")
            index_map.append(position)
            position += 2 if position + 1 < length and text[position + 1] == "\n" else 1
            continue
        chars.append(text[position])
        index_map.append(position)
        position += 1
    return "".join(chars), index_map


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    length = len(text)
    while index < length:
        char = text[index]
        if char == "\n":
            paragraph = _paragraph_break(text, index)
            if paragraph is not None:
                _append_span(text, start, index, spans)
                index = paragraph
                start = paragraph
                continue
        if char in _SENTENCE_END and _should_split(text, index):
            end = index + 1
            while end < length and text[end] in _SENTENCE_END:
                end += 1
            while end < length and text[end] in _CLOSERS:
                end += 1
            if end >= length or text[end].isspace():
                _append_span(text, start, end, spans)
                end = _skip_space_keep_paragraph(text, end)
                start = end
                index = end
                continue
        index += 1
    _append_span(text, start, length, spans)
    return spans


def _paragraph_break(text: str, index: int) -> int | None:
    position = index + 1
    length = len(text)
    while position < length and text[position] in " \t":
        position += 1
    if position < length and text[position] == "\n":
        position += 1
        while position < length and text[position] in " \t\n":
            position += 1
        return position
    return None


def _skip_space_keep_paragraph(text: str, index: int) -> int:
    length = len(text)
    while index < length and text[index] in " \t":
        index += 1
    if index < length and text[index] == "\n" and _paragraph_break(text, index) is None:
        index += 1
        while index < length and text[index] in " \t":
            index += 1
    return index


def _append_span(text: str, start: int, end: int, spans: list[tuple[int, int]]) -> None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))


def _should_split(text: str, index: int) -> bool:
    if text[index] != ".":
        return True
    previous = text[index - 1] if index > 0 else ""
    nxt = text[index + 1] if index + 1 < len(text) else ""
    decimal = previous.isdigit() and nxt.isdigit()
    abbreviation = _is_single_letter_abbreviation(text, index) or (
        _word_before(text, index) in _KNOWN_ABBREVIATIONS
    )
    return not (decimal or abbreviation)


def _is_single_letter_abbreviation(text: str, dot_index: int) -> bool:
    if dot_index < 1 or not text[dot_index - 1].isalpha():
        return False
    before = text[dot_index - 2] if dot_index >= 2 else ""
    return before in ("", " ", "\n", "\t", ".", "\u200c")


def _word_before(text: str, dot_index: int) -> str:
    position = dot_index - 1
    while position >= 0 and (text[position].isalnum() or text[position] == "\u200c"):
        position -= 1
    word = text[position + 1 : dot_index]
    return word.lower() if word.isascii() else word
