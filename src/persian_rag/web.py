"""HTML and web page loaders with main-content extraction.

:func:`extract_main_content` finds the article body of a page the way
reading-mode tools do: it drops scripts, navigation, and boilerplate, scores
the remaining blocks by text length, punctuation (Persian and Latin commas
count), class and id hints, and link density, then keeps the best container
and the sibling blocks that look like part of it. Everything uses the
standard library, so no extra package is needed.

>>> page = extract_main_content(html)
>>> page.title, page.text
"""

from __future__ import annotations

import html as html_lib
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Literal

from persian_rag.documents import SourceDocument, load_markdown_text, strip_markdown
from persian_rag.errors import PersianRagError

ExtractMode = Literal["main", "all"]

MAX_PAGE_BYTES = 10 * 1024 * 1024
DEFAULT_USER_AGENT = "persian-rag (+https://github.com/sadrmanesh/persian-rag)"

_VOID = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
_DROP = frozenset(
    {
        "script",
        "style",
        "noscript",
        "template",
        "svg",
        "canvas",
        "iframe",
        "object",
        "form",
        "button",
        "select",
        "textarea",
        "input",
        "nav",
        "aside",
        "footer",
        "dialog",
        "head",
    }
)
_INVISIBLE = frozenset({"script", "style", "head", "noscript", "template", "svg"})
_BLOCK = frozenset(
    {
        "address",
        "article",
        "blockquote",
        "body",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "tr",
        "ul",
    }
)
_PARAGRAPH_TAGS = frozenset({"p", "pre", "td", "blockquote", "li", "dd"})
_HEADINGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_IMPLIED_END = {
    "p": _BLOCK - {"li", "td", "th", "tr", "dd", "dt"},
    "li": frozenset({"li"}),
    "dt": frozenset({"dt", "dd"}),
    "dd": frozenset({"dt", "dd"}),
    "tr": frozenset({"tr"}),
    "td": frozenset({"td", "th", "tr"}),
    "th": frozenset({"td", "th", "tr"}),
    "option": frozenset({"option"}),
}

_NEGATIVE_RE = re.compile(
    r"comment|sidebar|side-bar|footer|footnote|masthead|menu|navbar|nav-|breadcrumb|"
    r"share|social|sharing|related|recommend|promo|sponsor|advert|banner|\bads?\b|"
    r"cookie|consent|popup|modal|subscribe|newsletter|pagination|pager|widget|tags?\b|"
    r"author-box|byline|meta-|login|signup",
    re.IGNORECASE,
)
_POSITIVE_RE = re.compile(
    r"article|body|content|entry|main|page|post|story|text|blog|matn|khabar|news-?body",
    re.IGNORECASE,
)
_UNLIKELY_ROLES = frozenset(
    {"navigation", "banner", "contentinfo", "complementary", "menu", "dialog", "search"}
)
_COMMA_RE = re.compile("[,،]")
_SPACE_RE = re.compile(r"[ \t\r\f\v\u00a0]+")
_CHARSET_RE = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-]+)""", re.IGNORECASE)


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: dict[str, str]
    parent: _Node | None = None
    children: list[_Node | str] = field(default_factory=list)
    score: float = 0.0
    scored: bool = False

    def iter_nodes(self) -> Iterator[_Node]:
        yield self
        for child in self.children:
            if isinstance(child, _Node):
                yield from child.iter_nodes()

    def text(self) -> str:
        parts: list[str] = []
        for child in self.children:
            parts.append(child if isinstance(child, str) else child.text())
        return "".join(parts)

    def link_text_length(self) -> int:
        return sum(len(_squash(node.text())) for node in self.iter_nodes() if node.tag == "a")

    def hint(self) -> str:
        return f"{self.attrs.get('class', '')} {self.attrs.get('id', '')}".strip()


class _TreeBuilder(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", {})
        self.stack: list[_Node] = [self.root]
        self.title = ""
        self.lang = ""
        self.meta: dict[str, str] = {}
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        values = {key.lower(): value or "" for key, value in attrs}
        if tag == "html" and values.get("lang"):
            self.lang = values["lang"]
        if tag == "meta":
            name = (values.get("property") or values.get("name") or "").lower()
            if name and values.get("content"):
                self.meta.setdefault(name, values["content"])
        if tag == "title":
            self._in_title = True
        closes = _IMPLIED_END.get(self.stack[-1].tag)
        if closes is not None and tag in closes and len(self.stack) > 1:
            self.stack.pop()
        node = _Node(tag, values, parent=self.stack[-1])
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag.lower() not in _VOID and len(self.stack) > 1 and self.stack[-1].tag == tag.lower():
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "title":
            self._in_title = False
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title += data
            return
        self.stack[-1].children.append(data)


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """The readable part of an HTML page."""

    title: str
    text: str
    lang: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


def _squash(text: str) -> str:
    return _SPACE_RE.sub(" ", text.replace("\n", " ")).strip()


def _is_hidden(node: _Node) -> bool:
    if "hidden" in node.attrs or node.attrs.get("aria-hidden") == "true":
        return True
    style = node.attrs.get("style", "").replace(" ", "").lower()
    return "display:none" in style or "visibility:hidden" in style


def _is_unlikely(node: _Node) -> bool:
    if node.tag in _DROP or _is_hidden(node):
        return True
    if node.attrs.get("role", "").lower() in _UNLIKELY_ROLES:
        return True
    if node.tag in ("body", "html", "article", "main", "#root"):
        return False
    hint = node.hint()
    return bool(hint) and bool(_NEGATIVE_RE.search(hint)) and not _POSITIVE_RE.search(hint)


def _prune(node: _Node) -> None:
    kept: list[_Node | str] = []
    for child in node.children:
        if isinstance(child, _Node):
            if _is_unlikely(child):
                continue
            _prune(child)
        kept.append(child)
    node.children = kept


def _class_weight(node: _Node) -> float:
    hint = node.hint()
    if not hint:
        return 0.0
    weight = 0.0
    if _NEGATIVE_RE.search(hint):
        weight -= 25.0
    if _POSITIVE_RE.search(hint):
        weight += 25.0
    return weight


def _initial_score(node: _Node) -> float:
    base = {
        "article": 10.0,
        "main": 10.0,
        "section": 3.0,
        "div": 5.0,
        "pre": 3.0,
        "td": 3.0,
        "blockquote": 3.0,
        "form": -3.0,
        "ol": -3.0,
        "ul": -3.0,
        "dl": -3.0,
        "li": -3.0,
        "th": -5.0,
        "header": -5.0,
        "address": -3.0,
    }.get(node.tag, 0.0)
    if node.tag in _HEADINGS:
        base = -5.0
    return base + _class_weight(node)


def _link_density(node: _Node) -> float:
    total = len(_squash(node.text()))
    if total == 0:
        return 0.0
    return node.link_text_length() / total


def _score_tree(root: _Node) -> list[_Node]:
    candidates: list[_Node] = []
    for node in root.iter_nodes():
        if node.tag not in _PARAGRAPH_TAGS:
            continue
        text = _squash(node.text())
        if len(text) < 25:
            continue
        points = 1.0 + len(_COMMA_RE.findall(text)) + min(len(text) / 100.0, 3.0)
        ancestor = node.parent
        for level in range(3):
            if ancestor is None or ancestor.tag == "#root":
                break
            if not ancestor.scored:
                ancestor.score = _initial_score(ancestor)
                ancestor.scored = True
                candidates.append(ancestor)
            ancestor.score += points / (1.0 if level == 0 else 2.0 if level == 1 else 3.0)
            ancestor = ancestor.parent
    for node in candidates:
        node.score *= 1.0 - _link_density(node)
    return candidates


def _render(node: _Node | str, lines: list[str], buffer: list[str]) -> None:
    if isinstance(node, str):
        buffer.append(node)
        return
    if node.tag == "br":
        buffer.append("\n")
        return
    if node.tag == "tr":
        _flush(lines, buffer)
        cells = [
            _squash(child.text())
            for child in node.children
            if isinstance(child, _Node) and child.tag in ("td", "th")
        ]
        row = " | ".join(cell for cell in cells if cell)
        if row:
            lines.append(row)
        return
    block = node.tag in _BLOCK
    if block:
        _flush(lines, buffer)
    for child in node.children:
        _render(child, lines, buffer)
    if block:
        _flush(lines, buffer)


def _flush(lines: list[str], buffer: list[str]) -> None:
    if not buffer:
        return
    raw = "".join(buffer)
    buffer.clear()
    # Line breaks from <br> stay inside the block; blocks are separated by
    # blank lines in _render_text.
    parts = [text for text in (_squash(part) for part in raw.split("\n")) if text]
    if parts:
        lines.append("\n".join(parts))


def _render_text(nodes: Sequence[_Node | str]) -> str:
    lines: list[str] = []
    buffer: list[str] = []
    for node in nodes:
        _render(node, lines, buffer)
    _flush(lines, buffer)
    deduped: list[str] = []
    for line in lines:
        if not deduped or deduped[-1] != line:
            deduped.append(line)
    return "\n\n".join(deduped)


def _body(root: _Node) -> _Node:
    for node in root.iter_nodes():
        if node.tag == "body":
            return node
    return root


def _select_content(root: _Node) -> list[_Node]:
    candidates = _score_tree(root)
    if not candidates:
        return [_body(root)]
    top = max(candidates, key=lambda node: node.score)
    parent = top.parent
    if parent is None or parent.tag == "#root":
        return [top]
    threshold = max(10.0, top.score * 0.2)
    selected: list[_Node] = []
    for sibling in parent.children:
        if not isinstance(sibling, _Node):
            continue
        if sibling is top:
            selected.append(sibling)
            continue
        bonus = top.score * 0.2 if top.hint() and sibling.hint() == top.hint() else 0.0
        if sibling.scored and sibling.score + bonus >= threshold:
            selected.append(sibling)
        elif sibling.tag == "p":
            text = _squash(sibling.text())
            density = _link_density(sibling)
            if (len(text) > 80 and density < 0.25) or (
                0 < len(text) <= 80 and density == 0 and re.search(r"[.!؟?]$", text)
            ):
                selected.append(sibling)
    return selected


def _strip_title(text: str, title: str) -> str:
    if not title:
        return text
    head, _, rest = text.partition("\n\n")
    if _squash(head) == _squash(title):
        return rest
    return text


def _clean_title(raw: str, h1: str) -> str:
    title = _squash(html_lib.unescape(raw))
    if h1 and h1 in title:
        return h1
    for separator in (" | ", " - ", " :: ", " « ", " » ", " – "):
        if separator in title:
            head = title.split(separator)[0].strip()
            if len(head.split()) >= 2:
                return head
    return title or h1


def parse_html(markup: str) -> tuple[_Node, _TreeBuilder]:
    builder = _TreeBuilder()
    builder.feed(markup)
    builder.close()
    return builder.root, builder


def extract_main_content(
    markup: str,
    *,
    url: str | None = None,
    min_text_length: int = 200,
) -> ExtractedPage:
    """Return the title and readable main text of an HTML document.

    When the best candidate holds fewer than ``min_text_length`` characters
    (very short pages, unusual layouts), the whole visible body text is
    returned instead, so content is never silently lost.
    """

    root, builder = parse_html(markup)
    _prune(root)
    h1 = next((_squash(node.text()) for node in root.iter_nodes() if node.tag == "h1"), "")
    raw_title = builder.meta.get("og:title") or builder.title
    title = _clean_title(raw_title, h1) if raw_title else h1
    text = _render_text(_select_content(root))
    if len(text) < min_text_length:
        fallback = _render_text([_body(root)])
        if len(fallback) > len(text):
            text = fallback
    text = _strip_title(text, title)
    metadata = _page_metadata(builder, url)
    return ExtractedPage(title=title, text=text, lang=builder.lang, metadata=metadata)


def html_to_text(markup: str) -> str:
    """Visible text of an HTML document without boilerplate removal.

    Scripts, styles, and hidden elements are still removed.
    """

    root, _ = parse_html(markup)
    for node in root.iter_nodes():
        node.children = [
            child
            for child in node.children
            if not (isinstance(child, _Node) and (child.tag in _INVISIBLE or _is_hidden(child)))
        ]
    return _render_text([_body(root)])


def _page_metadata(builder: _TreeBuilder, url: str | None) -> dict[str, str]:
    metadata: dict[str, str] = {}
    pairs = (
        ("description", ("og:description", "description")),
        ("published", ("article:published_time", "date", "pubdate")),
        ("author", ("author", "article:author")),
        ("site_name", ("og:site_name",)),
    )
    for key, names in pairs:
        for name in names:
            value = builder.meta.get(name)
            if value:
                metadata[key] = _squash(value)
                break
    if url:
        metadata["url"] = url
    return metadata


def _document(
    document_id: str,
    markup: str,
    *,
    extract: ExtractMode,
    url: str | None,
    extra: dict[str, Any],
) -> list[SourceDocument]:
    if extract not in ("main", "all"):
        raise ValueError("extract must be 'main' or 'all'")
    page = extract_main_content(markup, url=url)
    text = page.text if extract == "main" else html_to_text(markup)
    if not text.strip():
        return []
    metadata: dict[str, Any] = {"source": document_id, "format": "html", **extra}
    if page.title:
        metadata["title"] = page.title
    if page.lang:
        metadata["lang"] = page.lang
    metadata.update(page.metadata)
    return [SourceDocument(id=document_id, text=text, metadata=metadata)]


def load_html(
    path: str | Path,
    *,
    document_id: str | None = None,
    extract: ExtractMode = "main",
    encoding: str | None = None,
) -> list[SourceDocument]:
    """Load a saved HTML file.

    ``extract="main"`` keeps only the article body (see
    :func:`extract_main_content`); ``"all"`` keeps every visible line.
    """

    source = Path(path)
    raw = source.read_bytes()
    markup = decode_html(raw, encoding)
    return _document(
        document_id or source.name,
        markup,
        extract=extract,
        url=None,
        extra={"filename": source.name},
    )


def decode_html(raw: bytes, declared: str | None = None) -> str:
    """Decode HTML bytes using the declared, ``<meta>``, or detected charset.

    Falls back from UTF-8 to Windows-1256, which older Persian sites use.
    """

    candidates: list[str] = []
    if declared:
        candidates.append(declared)
    match = _CHARSET_RE.search(raw[:4096])
    if match:
        candidates.append(match.group(1).decode("ascii", errors="ignore"))
    candidates.extend(["utf-8", "cp1256"])
    for name in candidates:
        try:
            return raw.decode(name.strip().lower().replace("windows-", "cp"))
        except (LookupError, UnicodeDecodeError):
            continue
    return raw.decode("utf-8", errors="replace")


@dataclass(frozen=True, slots=True)
class FetchedPage:
    """A downloaded page."""

    url: str
    content: bytes
    content_type: str = ""
    charset: str | None = None


Fetcher = Callable[[str], FetchedPage]


def fetch_url(
    url: str,
    *,
    timeout: float = 20.0,
    max_bytes: int = MAX_PAGE_BYTES,
    user_agent: str = DEFAULT_USER_AGENT,
) -> FetchedPage:
    """Download ``url`` with the standard library (http and https only)."""

    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise PersianRagError(f"Only http and https URLs are supported: {url}")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": user_agent,
            "Accept": "text/html,application/xhtml+xml,text/plain,text/markdown;q=0.9,*/*;q=0.5",
            "Accept-Language": "fa,en;q=0.8",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            content = response.read(max_bytes + 1)
            headers = response.headers
            final_url = response.geturl()
    except urllib.error.HTTPError as exc:
        raise PersianRagError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise PersianRagError(f"Could not fetch {url}: {exc.reason}") from exc
    if len(content) > max_bytes:
        raise PersianRagError(f"{url} is larger than the {max_bytes}-byte limit")
    return FetchedPage(
        url=final_url,
        content=content,
        content_type=headers.get_content_type(),
        charset=headers.get_content_charset(),
    )


def load_url(
    url: str,
    *,
    extract: ExtractMode = "main",
    fetcher: Fetcher | None = None,
) -> list[SourceDocument]:
    """Fetch a web page and return its main content as a document.

    HTML is passed through :func:`extract_main_content`; Markdown and plain
    text responses are loaded as they are. The document id is the URL.
    ``fetcher`` replaces the HTTP call (for tests, caching, or a headless
    browser).
    """

    page = (fetcher or fetch_url)(url)
    content_type = page.content_type.lower()
    final = page.url or url
    if content_type in ("text/markdown", "text/x-markdown") or (
        content_type == "text/plain" and urllib.parse.urlsplit(final).path.endswith(".md")
    ):
        text = page.content.decode(page.charset or "utf-8", errors="replace")
        body, front = load_markdown_text(text)
        meta: dict[str, Any] = {"source": url, "url": final, "format": "markdown", **front}
        return [SourceDocument(id=url, text=body, metadata=meta)] if body.strip() else []
    if content_type == "text/plain":
        text = page.content.decode(page.charset or "utf-8", errors="replace")
        meta = {"source": url, "url": final, "format": "text"}
        return [SourceDocument(id=url, text=text, metadata=meta)] if text.strip() else []
    if content_type and "html" not in content_type and "xml" not in content_type:
        raise PersianRagError(f"Unsupported content type {content_type!r} at {url}")
    markup = decode_html(page.content, page.charset)
    return _document(url, markup, extract=extract, url=final, extra={})


def load_urls(
    urls: Sequence[str],
    *,
    extract: ExtractMode = "main",
    fetcher: Fetcher | None = None,
) -> list[SourceDocument]:
    """Load several URLs in order, skipping duplicates."""

    documents: list[SourceDocument] = []
    for url in dict.fromkeys(urls):
        documents.extend(load_url(url, extract=extract, fetcher=fetcher))
    return documents


__all__ = [
    "ExtractedPage",
    "FetchedPage",
    "decode_html",
    "extract_main_content",
    "fetch_url",
    "html_to_text",
    "load_html",
    "load_url",
    "load_urls",
    "strip_markdown",
]
