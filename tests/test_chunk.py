import pytest

from persian_rag.chunk import ChunkingConfig, chunk_text


def _assert_slices(text: str, chunks: list[object]) -> None:
    covered = [False] * len(text)
    for chunk in chunks:
        assert text[chunk.start : chunk.end] == chunk.text  # type: ignore[attr-defined]
        assert chunk.text == chunk.text.strip()  # type: ignore[attr-defined]
        for index in range(chunk.start, chunk.end):  # type: ignore[attr-defined]
            covered[index] = True
    for index, char in enumerate(text):
        if not char.isspace():
            assert covered[index], f"character {char!r} at {index} was dropped"


def test_fixed_chunks_overlap_and_cover_the_text() -> None:
    text = "یک دو سه چهار پنج شش هفت هشت نه ده"
    chunks = chunk_text(text, ChunkingConfig(strategy="fixed", chunk_size=12, overlap=4))
    assert len(chunks) > 1
    _assert_slices(text, chunks)
    assert chunks[0].index == 0
    assert chunks[1].start < chunks[0].end


def test_fixed_chunk_without_spaces_still_covers_characters() -> None:
    text = "ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی"
    chunks = chunk_text(text, ChunkingConfig(strategy="fixed", chunk_size=8, overlap=2))
    _assert_slices(text, chunks)


def test_sentence_chunks_pack_and_can_overlap() -> None:
    text = "جمله اول کوتاه است. جمله دوم هم کوتاه است. جمله سوم ادامه دارد."
    chunks = chunk_text(
        text,
        ChunkingConfig(strategy="sentence", chunk_size=40, overlap=0, sentence_overlap=1),
    )
    assert len(chunks) >= 2
    _assert_slices(text, chunks)
    assert "جمله" in chunks[0].text


def test_long_sentence_is_split_on_characters() -> None:
    text = "کلمه " * 30
    chunks = chunk_text(text.strip(), ChunkingConfig(strategy="sentence", chunk_size=20, overlap=0))
    assert len(chunks) > 1
    _assert_slices(text.strip(), chunks)


def test_recursive_chunks_prefer_paragraphs() -> None:
    first = "پاراگراف اول درباره حافظ است."
    second = "پاراگراف دوم درباره فوتبال است."
    text = f"{first}\n\n{second}"
    together = chunk_text(text, ChunkingConfig(strategy="recursive", chunk_size=500, overlap=20))
    assert len(together) == 1
    apart = chunk_text(text, ChunkingConfig(strategy="recursive", chunk_size=40, overlap=0))
    assert len(apart) >= 2
    _assert_slices(text, apart)
    assert any("حافظ" in chunk.text for chunk in apart)
    assert any("فوتبال" in chunk.text for chunk in apart)


def test_chunk_ids_follow_the_document() -> None:
    chunks = chunk_text("سلام دنیا.", document_id="docs/note.txt", metadata={"source": "note"})
    assert chunks[0].id == "docs/note.txt:0"
    assert chunks[0].document_id == "docs/note.txt"
    assert chunks[0].metadata == {"source": "note"}
    chunks[0].metadata["source"] = "changed"
    again = chunk_text("سلام دنیا.", document_id="docs/note.txt", metadata={"source": "note"})
    assert again[0].metadata["source"] == "note"


def test_invalid_chunk_config() -> None:
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=10, overlap=10)
    with pytest.raises(ValueError):
        ChunkingConfig(chunk_size=0, overlap=0)
    assert chunk_text("   ") == []
