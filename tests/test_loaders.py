from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from persian_rag.documents import load_documents
from persian_rag.errors import MissingDependencyError, PersianRagError
from persian_rag.loaders import (
    clean_extracted_text,
    fold_presentation_forms,
    load_docx,
    load_pdf,
    looks_visual_order,
    reverse_rtl_line,
    visual_order_score,
)

LOGICAL = "این کتاب در سال 2026 برای (ما) است"
# "سلام" written with presentation forms: initial seen, medial lam, final alef, isolated meem.
SALAM_FORMS = "\ufeb3\ufee0\ufe8e\ufee1"


def test_reverse_rtl_line_keeps_numbers_and_mirrors_brackets() -> None:
    visual = reverse_rtl_line(LOGICAL)
    assert visual.startswith("تسا")
    assert "2026" in visual
    assert reverse_rtl_line(visual) == LOGICAL
    assert reverse_rtl_line("ABC v1.2 لاس") == "سال v1.2 ABC"


def test_visual_order_detection() -> None:
    assert looks_visual_order(reverse_rtl_line(LOGICAL))
    assert not looks_visual_order(LOGICAL)
    assert visual_order_score(SALAM_FORMS) < 0
    assert visual_order_score(SALAM_FORMS[::-1]) > 0
    assert visual_order_score("hello world") == 0


def test_fold_presentation_forms() -> None:
    assert fold_presentation_forms(SALAM_FORMS) == "سلام"
    assert fold_presentation_forms("\ufefb") == "لا"
    assert fold_presentation_forms("ABC ۱۲") == "ABC ۱۲"


def test_clean_extracted_text_auto_repairs_visual_pages() -> None:
    page = "\u202b" + reverse_rtl_line(LOGICAL) + "\u202c\r\n\n\n\nPage 1\x0c"
    assert clean_extracted_text(page) == LOGICAL + "\n\nPage 1"
    assert clean_extracted_text(SALAM_FORMS[::-1]) == "سلام"
    assert clean_extracted_text(LOGICAL) == LOGICAL
    assert clean_extracted_text(LOGICAL, rtl_fix="reverse") == reverse_rtl_line(LOGICAL)
    visual = reverse_rtl_line(LOGICAL)
    assert clean_extracted_text(visual, rtl_fix="none") == visual
    assert clean_extracted_text("soft\u00adhyphen\u200f") == "softhyphen"
    with pytest.raises(ValueError):
        clean_extracted_text("x", rtl_fix="maybe")  # type: ignore[arg-type]


class FakePage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


def _reader(pages: list[str | None]) -> Any:
    def factory(path: Path, password: str | None) -> Any:
        return SimpleNamespace(pages=[FakePage(text) for text in pages])

    return factory


def test_load_pdf_with_injected_reader(tmp_path: Path) -> None:
    pdf = tmp_path / "book.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    pages = [reverse_rtl_line(LOGICAL), None, "فصل دوم درباره شیراز است."]
    whole = load_pdf(pdf, reader_factory=_reader(pages))
    assert len(whole) == 1
    assert whole[0].text == LOGICAL + "\n\nفصل دوم درباره شیراز است."
    assert whole[0].metadata == {
        "source": "book.pdf",
        "filename": "book.pdf",
        "format": "pdf",
        "page_count": 3,
    }
    split = load_pdf(pdf, split_pages=True, reader_factory=_reader(pages))
    assert [doc.id for doc in split] == ["book.pdf#page=1", "book.pdf#page=3"]
    assert split[1].metadata["page"] == 3
    assert load_pdf(pdf, reader_factory=_reader([None, " "])) == []


def test_load_pdf_wraps_reader_errors(tmp_path: Path) -> None:
    pdf = tmp_path / "bad.pdf"
    pdf.write_bytes(b"junk")

    def broken(path: Path, password: str | None) -> Any:
        raise RuntimeError("boom")

    with pytest.raises(PersianRagError, match="boom"):
        load_pdf(pdf, reader_factory=broken)


def _minimal_pdf(lines: list[str]) -> bytes:
    stream = "BT /F1 12 Tf 72 720 Td 14 TL " + " ".join(f"({line}) Tj T*" for line in lines) + " ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        "/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode("latin-1")
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin-1")
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode("latin-1")
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    ).encode("latin-1")
    return out


def test_load_pdf_with_pypdf(tmp_path: Path) -> None:
    pytest.importorskip("pypdf")
    pdf = tmp_path / "report.pdf"
    pdf.write_bytes(_minimal_pdf(["Persian RAG report", "Version 0.2.0"]))
    documents = load_documents(pdf)
    assert len(documents) == 1
    assert "Persian RAG report" in documents[0].text
    assert "0.2.0" in documents[0].text
    bad = tmp_path / "bad.pdf"
    bad.write_text("نه", encoding="utf-8")
    with pytest.raises(PersianRagError):
        load_documents(bad)


def test_missing_pdf_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {"pypdf", "docx"}:
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF")
    with pytest.raises(MissingDependencyError, match=r"persian-rag\[pdf\]"):
        load_pdf(pdf)
    docx_file = tmp_path / "x.docx"
    docx_file.write_bytes(b"PK")
    with pytest.raises(MissingDependencyError, match=r"persian-rag\[docx\]"):
        load_docx(docx_file)


def test_load_docx_with_injected_document(tmp_path: Path) -> None:
    path = tmp_path / "notes.docx"
    path.write_bytes(b"PK")
    cell = SimpleNamespace
    table = SimpleNamespace(
        rows=[
            SimpleNamespace(cells=[cell(text="شهر"), cell(text="استان")]),
            SimpleNamespace(cells=[cell(text="شیراز"), cell(text="شیراز"), cell(text="فارس")]),
        ]
    )
    document = SimpleNamespace(
        paragraphs=[SimpleNamespace(text="گزارش سفر"), SimpleNamespace(text="")],
        tables=[table],
    )
    loaded = load_docx(path, document_factory=lambda _: document)
    assert loaded[0].text == "گزارش سفر\n\nشهر | استان\nشیراز | فارس"
    assert loaded[0].metadata["format"] == "docx"
    assert (
        load_docx(path, document_factory=lambda _: SimpleNamespace(paragraphs=[], tables=[])) == []
    )

    def broken(_: Path) -> Any:
        raise ValueError("not a zip")

    with pytest.raises(PersianRagError, match="not a zip"):
        load_docx(path, document_factory=broken)


def test_load_docx_with_python_docx(tmp_path: Path) -> None:
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("حافظ در شیراز زاده شد.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "دیوان"
    table.rows[0].cells[1].text = "غزل"
    document.add_paragraph("پایان")
    folder = tmp_path / "docs"
    folder.mkdir()
    document.save(str(folder / "hafez.docx"))
    (folder / "note.txt").write_text("یادداشت", encoding="utf-8")
    loaded = load_documents(folder)
    assert [doc.id for doc in loaded] == ["hafez.docx", "note.txt"]
    assert loaded[0].text == "حافظ در شیراز زاده شد.\nدیوان | غزل\nپایان"
