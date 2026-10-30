"""Load plain text, Markdown, HTML, PDF, and DOCX files for ingestion."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from persian_rag.errors import PersianRagError

if TYPE_CHECKING:
    from persian_rag.loaders import RtlFix

MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BINARY_BYTES = 50 * 1024 * 1024
TEXT_SUFFIXES = frozenset({".txt", ".text", ".md", ".markdown", ".html", ".htm"})
MARKDOWN_SUFFIXES = frozenset({".md", ".markdown"})
HTML_SUFFIXES = frozenset({".html", ".htm"})
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
_FRONT_MATTER_RE = re.compile(
    r"\A(?:\ufeff)?---[ \t]*\n(.*?)\n(?:---|\.\.\.)[ \t]*(?:\n|\Z)", re.DOTALL
)
_ATX_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$")
_FENCE_LINE_RE = re.compile(r"^[ \t]*(```|~~~)")


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


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split simple YAML front matter (``key: value`` lines) from Markdown.

    Returns ``(fields, body)``. Nested YAML is not interpreted; such lines
    are skipped. Quotes around values are removed.
    """

    match = _FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text
    fields: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if not line.strip() or line.startswith((" ", "\t", "#", "-")):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key.strip() and value:
            fields[key.strip()] = value
    return fields, text[match.end() :]


def load_markdown_text(text: str) -> tuple[str, dict[str, str]]:
    """Strip front matter and markup; return ``(plain_text, front_matter)``.

    When the front matter has no ``title``, the first level-one heading is
    used.
    """

    fields, body = parse_front_matter(text)
    if "title" not in fields:
        for line in body.splitlines():
            heading = _ATX_RE.match(line)
            if heading and len(heading.group(1)) == 1:
                fields["title"] = heading.group(2).strip()
                break
    return strip_markdown(body), fields


@dataclass(frozen=True, slots=True)
class MarkdownSection:
    """A heading and the Markdown under it, up to the next heading of any level."""

    heading: str
    level: int
    path: tuple[str, ...]
    body: str


def split_markdown_sections(text: str) -> list[MarkdownSection]:
    """Split Markdown at ATX headings, ignoring ``#`` lines inside code fences.

    Text before the first heading becomes a section with an empty heading.
    ``path`` holds the enclosing headings, for example ``("Guide", "Install")``.
    """

    sections: list[MarkdownSection] = []
    trail: list[tuple[int, str]] = []
    heading, level = "", 0
    lines: list[str] = []
    in_fence = False

    def close() -> None:
        body = "\n".join(lines).strip()
        if heading or body:
            path = tuple(title for _, title in trail) if heading else ()
            sections.append(MarkdownSection(heading, level, path, body))

    for line in text.splitlines():
        if _FENCE_LINE_RE.match(line):
            in_fence = not in_fence
        match = None if in_fence else _ATX_RE.match(line)
        if match is None:
            lines.append(line)
            continue
        close()
        level = len(match.group(1))
        heading = match.group(2).strip()
        while trail and trail[-1][0] >= level:
            trail.pop()
        trail.append((level, heading))
        lines = []
    close()
    return sections


def _slug(text: str) -> str:
    words = re.findall(r"[\w\u0600-\u06ff]+", text.lower())
    return "-".join(words) or "section"


def load_markdown(
    path: str | Path,
    *,
    document_id: str | None = None,
    split_sections: bool = False,
) -> list[SourceDocument]:
    """Load a Markdown file with its front matter as metadata.

    With ``split_sections`` every heading starts a new document whose id ends
    with ``#slug`` and whose metadata holds ``section`` (the heading path
    joined with `` > ``). Section text starts with its heading so the topic
    stays searchable.
    """

    source = Path(path)
    doc_id = document_id or source.name
    raw = source.read_text(encoding="utf-8-sig")
    fields, body = parse_front_matter(raw)
    base: dict[str, Any] = {"source": doc_id, "filename": source.name, "format": "markdown"}
    base.update(fields)
    if not split_sections:
        text, found = load_markdown_text(raw)
        base.update(found)
        return [SourceDocument(id=doc_id, text=text, metadata=base)] if text.strip() else []
    documents: list[SourceDocument] = []
    used: dict[str, int] = {}
    for section in split_markdown_sections(body):
        content = strip_markdown(section.body).strip()
        if not content:
            continue
        slug = _slug(section.heading) if section.heading else "intro"
        used[slug] = used.get(slug, 0) + 1
        if used[slug] > 1:
            slug = f"{slug}-{used[slug]}"
        meta = dict(base)
        if section.heading:
            meta["section"] = " > ".join(section.path)
            content = f"{section.heading}\n\n{content}"
        documents.append(SourceDocument(id=f"{doc_id}#{slug}", text=content, metadata=meta))
    return documents


def load_documents(
    path: str | Path,
    *,
    rtl_fix: RtlFix = "auto",
    split_pages: bool = False,
    split_sections: bool = False,
    html_extract: Literal["main", "all"] = "main",
) -> list[SourceDocument]:
    """Load one file or a directory of supported files.

    Supported: ``.txt``, ``.md``, ``.html`` (main content only by default,
    see :func:`persian_rag.web.extract_main_content`), and with the optional extras ``.pdf``
    (``persian-rag[pdf]``) and ``.docx`` (``persian-rag[docx]``). Hidden files
    and directories are skipped. Markdown is passed through
    :func:`strip_markdown`; PDF and DOCX text through
    :func:`persian_rag.loaders.clean_extracted_text`. Document ids are paths
    relative to the directory (or the file name when ``path`` is a file).
    ``rtl_fix`` and ``split_pages`` apply to PDFs, ``split_sections`` to
    Markdown, and ``html_extract`` to HTML.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return _read_file(
            root,
            document_id=root.name,
            rtl_fix=rtl_fix,
            split_pages=split_pages,
            split_sections=split_sections,
            html_extract=html_extract,
        )
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
                split_sections=split_sections,
                html_extract=html_extract,
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
    split_sections: bool = False,
    html_extract: Literal["main", "all"] = "main",
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
    if suffix in HTML_SUFFIXES:
        from persian_rag.web import load_html

        return load_html(path, document_id=document_id, extract=html_extract)
    if suffix in MARKDOWN_SUFFIXES:
        try:
            return load_markdown(path, document_id=document_id, split_sections=split_sections)
        except UnicodeDecodeError as exc:
            raise PersianRagError(f"{path} is not valid UTF-8") from exc
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise PersianRagError(f"{path} is not valid UTF-8") from exc
    return [
        SourceDocument(
            id=document_id,
            text=text,
            metadata={"source": document_id, "filename": path.name},
        )
    ]
