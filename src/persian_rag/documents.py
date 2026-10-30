"""Load plain text, Markdown, PDF, and DOCX files for ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from persian_rag.errors import PersianRagError

if TYPE_CHECKING:
    from persian_rag.loaders import RtlFix

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BINARY_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown"})
BINARY_SUFFIXES = frozenset({".pdf", ".docx"})
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | BINARY_SUFFIXES

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


def load_documents(
    path: str | Path,
    *,
    rtl_fix: RtlFix = "auto",
    split_pages: bool = False,
) -> list[SourceDocument]:
    """Load one file or a directory of supported files.

    Supported: ``.txt``, ``.md``, and with the optional extras ``.pdf``
    (``persian-rag[pdf]``) and ``.docx`` (``persian-rag[docx]``). Hidden files
    and directories are skipped. Markdown is passed through
    :func:`strip_markdown`; PDF and DOCX text through
    :func:`persian_rag.loaders.clean_extracted_text`. Document ids are paths
    relative to the directory (or the file name when ``path`` is a file).
    ``rtl_fix`` and ``split_pages`` apply to PDFs.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return _read_file(root, document_id=root.name, rtl_fix=rtl_fix, split_pages=split_pages)
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
    documents: list[SourceDocument] = []
    for item in files:
        documents.extend(
            _read_file(
                item,
                document_id=item.relative_to(root).as_posix(),
                rtl_fix=rtl_fix,
                split_pages=split_pages,
            )
        )
    return documents


def _is_supported(path: Path, root: Path) -> bool:
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    relative = path.relative_to(root)
    return not any(part.startswith(".") for part in relative.parts)


def _read_file(
    path: Path,
    *,
    document_id: str,
    rtl_fix: RtlFix = "auto",
    split_pages: bool = False,
) -> list[SourceDocument]:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        supported = ", ".join(sorted(SUPPORTED_SUFFIXES))
        raise PersianRagError(f"Unsupported file type {path.suffix!r}. Supported: {supported}")
    size = path.stat().st_size
    limit = MAX_BINARY_BYTES if suffix in BINARY_SUFFIXES else MAX_FILE_BYTES
    if size > limit:
        raise PersianRagError(f"{path} is larger than the {limit}-byte limit")
    if suffix == ".pdf":
        from persian_rag.loaders import load_pdf

        return load_pdf(path, document_id=document_id, rtl_fix=rtl_fix, split_pages=split_pages)
    if suffix == ".docx":
        from persian_rag.loaders import load_docx

        return load_docx(path, document_id=document_id)
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PersianRagError(f"{path} is not valid UTF-8") from exc
    text = strip_markdown(raw) if suffix in {".md", ".markdown"} else raw
    return [
        SourceDocument(
            id=document_id,
            text=text,
            metadata={"source": document_id, "filename": path.name},
        )
    ]
