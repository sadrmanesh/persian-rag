"""Load plain text and Markdown for ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persian_rag.errors import PersianRagError

MAX_FILE_BYTES = 10 * 1024 * 1024
SUPPORTED_SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown"})

_CODE_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_TILDE_FENCE_RE = re.compile(r"~~~.*?~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`]*)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HEADING_RE = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_QUOTE_RE = re.compile(r"^[ \t]{0,3}>[ \t]?", re.MULTILINE)
_BULLET_RE = re.compile(r"^[ \t]*[-*+][ \t]+", re.MULTILINE)
_NUMBER_RE = re.compile(r"^[ \t]*\d+[.)][ \t]+", re.MULTILINE)
_BOLD_RE = re.compile(r"(\*\*|__)(.+?)\1")
_STAR_EM_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)")
_UNDER_EM_RE = re.compile(r"(?<!_)_(?!_)(.+?)_(?!_)")
_RULE_RE = re.compile(r"^[ \t]*([-*_])\1{2,}[ \t]*$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """One UTF-8 source file."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def strip_markdown(text: str) -> str:
    """Remove common Markdown markup and keep the visible words.

    This is intentionally small. It covers headings, emphasis, links,
    lists, quotes, and fenced code. It is not a full Markdown parser.
    """

    text = _CODE_FENCE_RE.sub("\n", text)
    text = _TILDE_FENCE_RE.sub("\n", text)
    text = _INLINE_CODE_RE.sub(r"\1", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _HEADING_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = _BULLET_RE.sub("", text)
    text = _NUMBER_RE.sub("", text)
    text = _BOLD_RE.sub(r"\2", text)
    text = _STAR_EM_RE.sub(r"\1", text)
    text = _UNDER_EM_RE.sub(r"\1", text)
    text = _RULE_RE.sub("", text)
    return text


def load_documents(path: str | Path) -> list[SourceDocument]:
    """Load a UTF-8 text file or a directory of ``.txt`` and ``.md`` files.

    Hidden files and directories are skipped. Markdown files are passed
    through :func:`strip_markdown`. Document ids are paths relative to the
    directory (or the file name when ``path`` is a file).
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [_read_file(root, document_id=root.name)]
    if not root.is_dir():
        raise PersianRagError(f"Not a file or directory: {root}")

    files = [
        item
        for item in sorted(root.rglob("*"))
        if item.is_file() and not item.is_symlink() and _is_supported(item, root)
    ]
    if not files:
        suffixes = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise PersianRagError(f"No supported files ({suffixes}) found in {root}")
    return [_read_file(item, document_id=item.relative_to(root).as_posix()) for item in files]


def _is_supported(path: Path, root: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    relative = path.relative_to(root)
    return not any(part.startswith(".") for part in relative.parts)


def _read_file(path: Path, *, document_id: str) -> SourceDocument:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise PersianRagError(f"Unsupported file type {path.suffix!r}. Supported: {supported}")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        raise PersianRagError(f"{path} is larger than the {MAX_FILE_BYTES}-byte limit")
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PersianRagError(f"{path} is not valid UTF-8") from exc
    text = strip_markdown(raw) if path.suffix.lower() in {".md", ".markdown"} else raw
    return SourceDocument(
        id=document_id,
        text=text,
        metadata={"source": document_id, "filename": path.name},
    )
