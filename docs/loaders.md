# Loading documents

```python
from persian_rag import load_documents, load_docx, load_pdf

documents = load_documents("./library")  # .txt, .text, .md, .markdown, .pdf, .docx
```

`load_documents` accepts a file or a directory. Hidden files and folders are skipped, Markdown is reduced to its visible text, and document ids are paths relative to the directory. Text files may be up to 10 MiB and PDF or Word files up to 50 MiB.

## PDF

```bash
pip install "persian-rag[pdf]"
```

```python
whole = load_pdf("book.pdf")  # one document
pages = load_pdf("book.pdf", split_pages=True)  # ids like "book.pdf#page=3"
```

Text extracted from Persian PDFs has three common problems, and the loader fixes all of them:

1. **Presentation forms.** Many PDFs store contextual glyphs (for example U+FEE3) instead of letters. They are folded into ordinary letters.
2. **Visual order.** Some generators write each line right to left as drawn, so extraction yields reversed words such as `باتک` for `کتاب`. With `rtl_fix="auto"` every page is scored on two signals: which glyph forms open and close each word, and whether frequent words like `از` and `است` appear forwards or reversed. Reversed pages are flipped line by line, brackets are mirrored, and numbers and Latin words keep their left to right order.
3. **Invisible characters.** Bidi embedding and isolate marks, soft hyphens, and stray control characters are removed.

Set `rtl_fix="reverse"` to always flip lines with Persian letters, or `rtl_fix="none"` to keep the extracted order. Encrypted files take `password=`. To use a different extractor, pass `reader_factory`, a callable that returns an object with a `pages` list whose items have `extract_text()`.

The same cleanup is available on its own:

```python
from persian_rag import clean_extracted_text

clean_extracted_text(raw_text, rtl_fix="auto")
```

Scanned PDFs contain images, not text. Run OCR first; the loader returns no document for pages without text.

## Word

```bash
pip install "persian-rag[docx]"
```

```python
load_docx("report.docx")
```

Paragraphs and tables are read in body order. Each table row becomes one `cell | cell` line, and merged cells are not repeated. Word stores text in logical order, so no RTL repair is applied by default.

## Errors

A missing extra raises `MissingDependencyError` (a subclass of both `PersianRagError` and `ImportError`) with the install command. A corrupt file raises `PersianRagError` with the parser message.
