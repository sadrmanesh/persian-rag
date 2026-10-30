from pathlib import Path

import pytest

from persian_rag.documents import MAX_FILE_BYTES, load_documents, strip_markdown
from persian_rag.errors import PersianRagError

FIXTURES = Path(__file__).parent / "fixtures"


def test_strip_markdown_keeps_persian_words() -> None:
    text = (FIXTURES / "sample.md").read_text(encoding="utf-8")
    plain = strip_markdown(text)
    assert "عنوان" in plain
    assert "متن" in plain
    assert "مورد اول" in plain
    assert "پیوند" in plain
    assert "نقل از متن" in plain
    assert "secret" not in plain
    assert "https://example.com" not in plain
    assert "**" not in plain


def test_load_directory_skips_hidden_files_and_reads_nested_text() -> None:
    documents = load_documents(FIXTURES)
    ids = {document.id for document in documents}
    assert "sample.txt" in ids
    assert "sample.md" in ids
    assert "nested/note.txt" in ids
    assert ".hidden.txt" not in ids
    joined = "\n".join(document.text for document in documents)
    assert "پنهان" not in joined
    assert "شیراز" in joined
    assert all(document.metadata["filename"] for document in documents)


def test_load_rejects_unsupported_and_oversized_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pdf = tmp_path / "notes.rtf"
    pdf.write_text("نه", encoding="utf-8")
    with pytest.raises(PersianRagError):
        load_documents(pdf)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(PersianRagError):
        load_documents(empty)
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        load_documents(missing)
    big = tmp_path / "big.txt"
    big.write_text("سلام", encoding="utf-8")
    monkeypatch.setattr("persian_rag.documents.MAX_FILE_BYTES", 1)
    assert MAX_FILE_BYTES > 1
    with pytest.raises(PersianRagError):
        load_documents(big)
    broken = tmp_path / "bad.txt"
    broken.write_bytes(b"\xff\xfe")
    monkeypatch.setattr("persian_rag.documents.MAX_FILE_BYTES", MAX_FILE_BYTES)
    with pytest.raises(PersianRagError):
        load_documents(broken)
