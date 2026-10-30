from persian_rag.tokenize import split_sentences, tokenize


def test_tokenize_keeps_zwnj_and_splits_punctuation() -> None:
    tokens = tokenize("می‌روم به مدرسه، Hello!")
    assert tokens == ["می‌روم", "به", "مدرسه", "hello"]


def test_tokenize_keeps_decimals() -> None:
    assert tokenize("نسخه ۱.۲ و 3.14") == ["نسخه", "۱.۲", "و", "۳.۱۴"]


def test_tokenize_can_skip_normalization() -> None:
    raw = tokenize("يك", normalize_text=False)
    assert raw == ["يك"]


def test_sentence_splitter_handles_persian_punctuation() -> None:
    text = "سلام. خوبی؟ بله! ه.ش. در تقویم است. دکتر. علی آمد."
    sentences = split_sentences(text)
    assert sentences[0] == "سلام."
    assert sentences[1] == "خوبی؟"
    assert sentences[2] == "بله!"
    assert sentences[3] == "ه.ش. در تقویم است."
    assert sentences[4] == "دکتر. علی آمد."


def test_sentence_splitter_keeps_decimals_and_quotes() -> None:
    text = "نسخه 1.2 منتشر شد. گفت: «سلام.» او رفت."
    assert split_sentences(text) == ["نسخه 1.2 منتشر شد.", "گفت: «سلام.»", "او رفت."]


def test_paragraphs_and_blank_input() -> None:
    assert split_sentences("اول.\n\nدوم.") == ["اول.", "دوم."]
    assert split_sentences("   ") == []
    assert split_sentences("سلام.\r\nخوبی؟") == ["سلام.", "خوبی؟"]
