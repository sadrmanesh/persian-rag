# Web pages and Markdown

New in 0.3.0. Everything here uses the standard library only.

## HTML files and URLs

```python
from persian_rag import extract_main_content, html_to_text, load_html, load_url, load_urls

docs = load_url("https://fa.wikipedia.org/wiki/حافظ")  # a list with one SourceDocument
docs = load_urls(["https://example.com/a", "https://example.com/b"])
docs = load_html("saved/article.html")
```

`load_url` downloads the page (following redirects, honouring the declared charset or `<meta charset>`), extracts the main content, and returns a document whose id is the URL. The metadata has `source`, `url` (after redirects), `format`, and, when the page declares them, `title`, `lang`, `description`, `published`, `author`, and `site_name`. Markdown and plain text responses are loaded as they are. Only `http` and `https` URLs are accepted, and pages larger than 10 MiB are refused.

Pass `fetcher=` to supply your own download function (for example one with authentication, a proxy, or a cache). It receives the URL and returns a `FetchedPage(url, content, content_type, charset)`.

### Main-content extraction

`extract="main"` (the default) works like a readability filter:

1. Scripts, styles, forms, navigation, headers, footers, asides, and elements hidden with `hidden`, `aria-hidden="true"`, or `display:none` are removed.
2. `<article>` and `<main>` are preferred when they hold enough text.
3. Otherwise each block is scored by its text length, punctuation (Persian and Latin commas and full stops), and link density, and the best block plus its strong siblings is kept. Class and id names such as `content`, `article`, `post`, `matn` raise a score; `comment`, `sidebar`, `footer`, `share`, `related`, `ads` lower it.

If the best candidate is shorter than 200 characters, the whole visible body is used instead, so short pages are not lost. `extract="all"` always keeps every visible block. Headings, paragraphs, and list items become separate paragraphs, `<br>` becomes a line break, and table rows become `cell | cell` lines.

```python
page = extract_main_content(html)  # ExtractedPage(title, text, lang, metadata)
everything = html_to_text(html)  # every visible block as text
```

`load_documents("./site")` now reads `.html` and `.htm` files with the same extractor; choose the mode with `html_extract=`.

## Markdown

```python
from persian_rag import load_markdown, load_markdown_text, split_markdown_sections

doc = load_markdown("guide.md")
sections = load_markdown("guide.md", split_sections=True)
```

* Front matter (a `---` block with `key: value` lines) is added to the metadata. Quotes around values are removed; nested YAML is skipped.
* The first level 1 heading becomes the `title` when the front matter has none.
* Markdown syntax is reduced to visible text: links keep their label, images keep their alt text, emphasis, list marks, and quotes are removed, and fenced code blocks are dropped.
* With `split_sections=True` each heading starts a new document. Ids look like `guide.md#نصب`, duplicate headings get `-2`, `-3`, and so on, and `metadata["section"]` holds the heading path (`"Install > Linux"`). Headings inside code fences are ignored.

`load_documents(path, split_sections=True)` applies the same split to every Markdown file in a folder.

## Command line

```bash
persian-rag ingest ./docs https://example.com/blog/post --split-sections --html-extract main
```

`ingest` accepts any mix of files, folders, and URLs.
