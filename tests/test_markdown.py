from __future__ import annotations

from pathlib import Path

from persian_rag.documents import (
    load_documents,
    load_markdown,
    load_markdown_text,
    parse_front_matter,
    split_markdown_sections,
)

GUIDE = """---
title: "راهنمای نصب"
lang: fa
tags:
  - nested
---
# راهنما

مقدمه کوتاه.

## نصب

دستور `pip install persian-rag` را اجرا کنید.

```bash
# not a heading
pip install persian-rag
```

## نصب

بخش تکراری.

### پیکربندی

متغیر محیطی را تنظیم کنید.
"""


def test_front_matter_is_parsed_and_removed() -> None:
    fields, body = parse_front_matter(GUIDE)
    assert fields == {"title": "راهنمای نصب", "lang": "fa"}
    assert body.startswith("# راهنما")
    text, meta = load_markdown_text(GUIDE)
    assert meta["title"] == "راهنمای نصب"
    assert "---" not in text
    assert "lang: fa" not in text


def test_title_falls_back_to_first_h1() -> None:
    _, meta = load_markdown_text("مقدمه\n\n# عنوان اصلی\n\n## فرعی\n")
    assert meta == {"title": "عنوان اصلی"}


def test_no_front_matter() -> None:
    assert parse_front_matter("# x\n---\n") == ({}, "# x\n---\n")


def test_sections_ignore_code_fences() -> None:
    sections = split_markdown_sections(parse_front_matter(GUIDE)[1])
    headings = [section.heading for section in sections]
    assert headings == ["راهنما", "نصب", "نصب", "پیکربندی"]
    assert sections[-1].path == ("راهنما", "نصب", "پیکربندی")
    assert "# not a heading" in sections[1].body


def test_load_markdown_split_sections(tmp_path: Path) -> None:
    path = tmp_path / "guide.md"
    path.write_text(GUIDE, encoding="utf-8")
    whole = load_markdown(path)
    assert len(whole) == 1
    assert whole[0].metadata["title"] == "راهنمای نصب"
    parts = load_markdown(path, split_sections=True)
    ids = [doc.id for doc in parts]
    assert ids == ["guide.md#راهنما", "guide.md#نصب", "guide.md#نصب-2", "guide.md#پیکربندی"]
    assert parts[3].metadata["section"] == "راهنما > نصب > پیکربندی"
    assert parts[3].text.startswith("پیکربندی")
    via_loader = load_documents(tmp_path, split_sections=True)
    assert [doc.id for doc in via_loader] == ids
