"""Normalize Persian and Arabic-script text for retrieval."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

DigitStyle = Literal["persian", "ascii", "keep"]
ZwnjMode = Literal["normalize", "remove", "space"]
PunctuationStyle = Literal["persian", "ascii", "keep"]

# Arabic yeh / kaf and common presentation forms are folded after NFKC.
_ARABIC_TO_PERSIAN = str.maketrans(
    {
        "ي": "ی",  # Arabic yeh
        "ى": "ی",  # alef maksura
        "ك": "ک",  # Arabic kaf
        "أ": "ا",
        "إ": "ا",
        "ٱ": "ا",
        "ؤ": "و",
        "ة": "ه",
    }
)

_INVISIBLE = dict.fromkeys(
    map(
        ord,
        [
            "\u200b",  # zero-width space
            "\u200d",  # zero-width joiner
            "\u200e",  # left-to-right mark
            "\u200f",  # right-to-left mark
            "\ufeff",  # BOM / zero-width no-break space
            "\u2060",  # word joiner
            "\u202a",
            "\u202b",
            "\u202c",
            "\u202d",
            "\u202e",
            "\u2066",
            "\u2067",
            "\u2068",
            "\u2069",
        ],
    )
)

_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = "\u0640"
_ZWNJ = "\u200c"

_ASCII_DIGITS = "0123456789"
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"

_TO_PERSIAN_PUNCT = str.maketrans({",": "،", ";": "؛", "?": "؟", "%": "٪"})
_TO_ASCII_PUNCT = str.maketrans({"،": ",", "؛": ";", "؟": "?", "٪": "%"})
_ORNAMENTS_RE = re.compile(r"[•●▪▸►◆★☆♦◦∙·]")
_PUNCT_REPEAT_RE = re.compile(r"([!?؟،؛:٪\.])\1+")
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([،؛؟!\.٪…])")
_SPACE_AFTER_PUNCT_RE = re.compile(r"([،؛؟٪])(?=[^\s\n])")

# Affixes commonly written with a half-space in Persian.
_PREFIX_SPACE_RE = re.compile(r"(?<![\w\u200c])(نمی|می|بی)[ \t]+(?=[\u0600-\u06FF])")
_PREFIX_HYPHEN_RE = re.compile(r"(?<![\w\u200c])(نمی|می|بی)[ \t]*[-‐‑–—−][ \t]*(?=[\u0600-\u06FF])")
_SUFFIX_SPACE_RE = re.compile(r"(?<=[\u0600-\u06FF])[ \t]+(هایی|های|ها|ترین|تر)(?![\w\u200c])")


@dataclass(frozen=True, slots=True)
class NormalizeConfig:
    """Options for :func:`normalize`.

    ``remove_diacritics`` is off by default. Turn it on when vowels marks
    should not affect matching.
    """

    unify_arabic_chars: bool = True
    compatibility_forms: bool = True
    zwnj: ZwnjMode = "normalize"
    correct_half_spaces: bool = True
    digits: DigitStyle = "persian"
    remove_diacritics: bool = False
    remove_tatweel: bool = True
    clean_punctuation: bool = True
    punctuation: PunctuationStyle = "persian"

    def __post_init__(self) -> None:
        if self.zwnj not in ("normalize", "remove", "space"):
            raise ValueError("zwnj must be 'normalize', 'remove', or 'space'")
        if self.digits not in ("persian", "ascii", "keep"):
            raise ValueError("digits must be 'persian', 'ascii', or 'keep'")
        if self.punctuation not in ("persian", "ascii", "keep"):
            raise ValueError("punctuation must be 'persian', 'ascii', or 'keep'")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> NormalizeConfig:
        known = {item.name for item in fields(cls)}
        kwargs = {key: data[key] for key in known if key in data}
        return cls(**kwargs)


def normalize(text: str, config: NormalizeConfig | None = None) -> str:
    """Return a canonical form of ``text`` for indexing and queries.

    The same config must be used for documents and queries. The function is
    idempotent for a fixed config: ``normalize(normalize(text)) == normalize(text)``.
    """

    options = config or NormalizeConfig()
    if not text:
        return ""

    if options.compatibility_forms:
        text = unicodedata.normalize("NFKC", text)

    text = text.translate(_INVISIBLE)

    if options.unify_arabic_chars:
        text = text.translate(_ARABIC_TO_PERSIAN)

    if options.remove_tatweel:
        text = text.replace(_TATWEEL, "")

    if options.remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)

    text = _map_digits(text, options.digits)

    if options.clean_punctuation:
        text = _clean_punctuation(text, options.punctuation)

    text = _apply_zwnj_policy(text, options)
    text = _normalize_whitespace(text)
    return text


def _map_digits(text: str, style: DigitStyle) -> str:
    if style == "keep":
        return text
    if style == "persian":
        table = str.maketrans(
            _ASCII_DIGITS + _ARABIC_INDIC_DIGITS,
            _PERSIAN_DIGITS + _PERSIAN_DIGITS,
        )
        return text.translate(table)
    table = str.maketrans(
        _PERSIAN_DIGITS + _ARABIC_INDIC_DIGITS,
        _ASCII_DIGITS + _ASCII_DIGITS,
    )
    return text.translate(table)


def _clean_punctuation(text: str, style: PunctuationStyle) -> str:
    if style == "persian":
        text = text.translate(_TO_PERSIAN_PUNCT)
    elif style == "ascii":
        text = text.translate(_TO_ASCII_PUNCT)
    text = _ORNAMENTS_RE.sub(" ", text)
    text = _PUNCT_REPEAT_RE.sub(_collapse_punctuation, text)
    text = _SPACE_BEFORE_PUNCT_RE.sub(r"\1", text)
    text = _SPACE_AFTER_PUNCT_RE.sub(r"\1 ", text)
    return text


def _collapse_punctuation(match: re.Match[str]) -> str:
    char = match.group(1)
    if char == ".":
        return "..."
    return char


def _apply_zwnj_policy(text: str, options: NormalizeConfig) -> str:
    if options.zwnj == "remove":
        return text.replace(_ZWNJ, "")
    if options.zwnj == "space":
        return text.replace(_ZWNJ, " ")
    text = _normalize_zwnj(text)
    if options.correct_half_spaces:
        text = _correct_half_spaces(text)
        text = _normalize_zwnj(text)
    return text


def _normalize_zwnj(text: str) -> str:
    text = re.sub(rf"{_ZWNJ}{{2,}}", _ZWNJ, text)
    text = re.sub(rf"[ \t]*{_ZWNJ}[ \t]*", _ZWNJ, text)
    text = re.sub(rf"{_ZWNJ}+(?=[^\w\u0600-\u06FF])", "", text)
    text = re.sub(rf"(?<=[^\w\u0600-\u06FF]){_ZWNJ}+", "", text)
    return text.strip(_ZWNJ)


def _correct_half_spaces(text: str) -> str:
    text = _PREFIX_HYPHEN_RE.sub(lambda match: match.group(1) + _ZWNJ, text)
    text = _PREFIX_SPACE_RE.sub(lambda match: match.group(1) + _ZWNJ, text)
    text = _SUFFIX_SPACE_RE.sub(lambda match: _ZWNJ + match.group(1), text)
    return text


def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
