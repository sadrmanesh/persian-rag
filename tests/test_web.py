from __future__ import annotations

from pathlib import Path

import pytest

from persian_rag.documents import load_documents
from persian_rag.errors import PersianRagError
from persian_rag.web import (
    FetchedPage,
    decode_html,
    extract_main_content,
    html_to_text,
    load_html,
    load_url,
    load_urls,
)

FIXTURES = Path(__file__).parent / "fixtures"
ARTICLE = (FIXTURES / "article.html").read_text(encoding="utf-8")


def test_main_content_drops_boilerplate() -> None:
    page = extract_main_content(ARTICLE, url="https://example.com/a")
    assert page.title == "کتابخانه ملی و حفظ نسخه های خطی"
    assert page.lang == "fa"
    assert "سی هزار نسخه خطی" in page.text
    assert "کاغذ ژاپنی" in page.text
    assert "نسخه خطی | ۳۰۰۰۰" in page.text
    for noise in ("پربازدید", "دیدگاه کاربر", "حقوق", "اشتراک", "analytics", "فرهنگ"):
        assert noise not in page.text
    assert page.metadata["description"] == "گزارشی درباره نگهداری نسخه های خطی"
    assert page.metadata["author"] == "نویسنده نمونه"
    assert page.metadata["url"] == "https://example.com/a"


def test_paragraphs_are_separated() -> None:
    page = extract_main_content(ARTICLE)
    assert "\n\n" in page.text
    assert page.text.count("کتابخانه ملی ایران") == 1


def test_short_page_falls_back_to_body_text() -> None:
    markup = "<html><body><div><span>سلام دنیا</span></div><script>x()</script></body></html>"
    page = extract_main_content(markup)
    assert page.text == "سلام دنیا"


def test_html_to_text_keeps_everything_visible() -> None:
    text = html_to_text(ARTICLE)
    assert "پربازدیدها" in text
    assert "analytics" not in text
    assert ".menu" not in text


def test_hidden_elements_and_entities() -> None:
    markup = (
        "<body><p hidden>پنهان</p><p style='display:none'>نامرئی</p>"
        "<p>کتاب &amp; قلم<br>خط دوم</p></body>"
    )
    text = html_to_text(markup)
    assert "پنهان" not in text and "نامرئی" not in text
    assert "کتاب & قلم\nخط دوم" in text


def test_load_html_file_and_directory(tmp_path: Path) -> None:
    (tmp_path / "a.html").write_text(ARTICLE, encoding="utf-8")
    (tmp_path / "b.txt").write_text("متن ساده", encoding="utf-8")
    docs = load_html(tmp_path / "a.html")
    assert docs[0].id == "a.html"
    assert docs[0].metadata["title"] == "کتابخانه ملی و حفظ نسخه های خطی"
    assert docs[0].metadata["format"] == "html"
    everything = load_documents(tmp_path, html_extract="all")
    by_id = {doc.id: doc for doc in everything}
    assert "پربازدیدها" in by_id["a.html"].text
    main = {doc.id: doc for doc in load_documents(tmp_path)}
    assert "پربازدیدها" not in main["a.html"].text


def test_decode_html_charsets() -> None:
    assert decode_html("سلام".encode()) == "سلام"
    legacy = '<meta charset="windows-1256"><p>سلام</p>'.encode("cp1256")
    assert "سلام" in decode_html(legacy)
    assert decode_html("سلام".encode("cp1256")) == "سلام"


def _fetcher(pages: dict[str, FetchedPage]):  # type: ignore[no-untyped-def]
    def fetch(url: str) -> FetchedPage:
        return pages[url]

    return fetch


def test_load_url_html_markdown_and_text() -> None:
    pages = {
        "https://ex.com/a": FetchedPage(
            url="https://ex.com/a",
            content=ARTICLE.encode("utf-8"),
            content_type="text/html",
            charset="utf-8",
        ),
        "https://ex.com/readme.md": FetchedPage(
            url="https://ex.com/readme.md",
            content="---\ntitle: راهنما\n---\n# سرآغاز\n\n**متن** پررنگ".encode(),
            content_type="text/markdown",
            charset=None,
        ),
        "https://ex.com/t": FetchedPage(
            url="https://ex.com/t", content=b"plain", content_type="text/plain", charset=None
        ),
    }
    fetch = _fetcher(pages)
    html = load_url("https://ex.com/a", fetcher=fetch)[0]
    assert html.id == "https://ex.com/a"
    assert "کاغذ ژاپنی" in html.text
    md = load_url("https://ex.com/readme.md", fetcher=fetch)[0]
    assert md.metadata["title"] == "راهنما"
    assert "**" not in md.text
    assert load_url("https://ex.com/t", fetcher=fetch)[0].text == "plain"
    assert len(load_urls(list(pages), fetcher=fetch)) == 3


def test_load_url_rejects_binary() -> None:
    page = FetchedPage(url="u", content=b"%PDF", content_type="application/pdf", charset=None)
    with pytest.raises(PersianRagError, match="Unsupported content type"):
        load_url("https://ex.com/f.pdf", fetcher=lambda url: page)


def test_fetch_url_rejects_other_schemes() -> None:
    from persian_rag.web import fetch_url

    with pytest.raises(PersianRagError):
        fetch_url("file:///etc/passwd")
