from __future__ import annotations

import json
from pathlib import Path

import pytest

from persian_rag import cli
from persian_rag.cli import main
from persian_rag.persist import load_index

ARTICLE = (Path(__file__).parent / "fixtures" / "article.html").read_text(encoding="utf-8")


def _ingest(paths: list[str], index: Path, *extra: str) -> int:
    return main(["ingest", *paths, "--index", str(index), "--dimension", "64", *extra])


def test_ingest_update_only_touches_changed_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("حافظ در شیراز زاده شد.", encoding="utf-8")
    (docs / "b.txt").write_text("فردوسی شاهنامه را سرود.", encoding="utf-8")
    (docs / "page.html").write_text(ARTICLE, encoding="utf-8")
    index = tmp_path / "index"
    assert _ingest([str(docs)], index) == 0
    saved = load_index(index)
    assert set(saved.documents) == {"a.txt", "b.txt", "page.html"}
    assert not any("پربازدید" in chunk.text for chunk in saved.chunks)
    capsys.readouterr()

    (docs / "a.txt").write_text("حافظ در شیراز زندگی کرد.", encoding="utf-8")
    (docs / "b.txt").unlink()
    (docs / "c.txt").write_text("تهران پایتخت است.", encoding="utf-8")
    assert _ingest([str(docs)], index, "--update") == 0
    out = capsys.readouterr().out
    assert "added 1, updated 1, deleted 1, unchanged 1" in out
    saved = load_index(index)
    assert sorted(saved.documents) == ["a.txt", "c.txt", "page.html"]

    (docs / "c.txt").unlink()
    assert _ingest([str(docs)], index, "--update", "--keep-missing") == 0
    assert "c.txt" in load_index(index).documents


def test_update_without_existing_index_builds_it(tmp_path: Path) -> None:
    source = tmp_path / "a.md"
    source.write_text("# عنوان\n\nمتن\n\n## بخش\n\nجزئیات", encoding="utf-8")
    index = tmp_path / "idx"
    assert _ingest([str(source)], index, "--update", "--split-sections") == 0
    assert sorted(load_index(index).documents) == ["a.md#بخش", "a.md#عنوان"]


def test_ingest_url(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from persian_rag import web

    def fake_fetch(url: str) -> web.FetchedPage:
        return web.FetchedPage(url=url, content=ARTICLE.encode(), content_type="text/html")

    monkeypatch.setattr(web, "fetch_url", fake_fetch)
    index = tmp_path / "idx"
    assert _ingest(["https://example.com/article"], index) == 0
    assert list(load_index(index).documents) == ["https://example.com/article"]


def _fake_llm(reply: str):  # type: ignore[no-untyped-def]
    prompts: list[str] = []

    def llm(prompt: str) -> str:
        prompts.append(prompt)
        return reply

    return llm, prompts


def test_query_with_citations_and_expansion(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "room.txt").write_text("اطاق نشیمن رو به حیاط است.", encoding="utf-8")
    (docs / "sport.txt").write_text("فوتبال ورزشی گروهی است.", encoding="utf-8")
    index = tmp_path / "idx"
    assert _ingest([str(docs)], index) == 0
    llm, prompts = _fake_llm("اطاق نشیمن رو به حیاط است [1].")
    monkeypatch.setattr(cli, "_llm_from_args", lambda args: llm)
    capsys.readouterr()
    status = main(
        [
            "query",
            "اتاق",
            "--index",
            str(index),
            "--llm",
            "ollama",
            "--citations",
            "--expand",
            "rules",
            "--mode",
            "sparse",
            "--k",
            "1",
            "--json",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["contexts"][0]["document_id"] == "room.txt"
    assert payload["citations"][0]["document_id"] == "room.txt"
    assert "کروشه" in prompts[0]

    main(["query", "اتاق", "--index", str(index), "--llm", "ollama", "--citations", "--k", "1"])
    assert "Sources:" in capsys.readouterr().out


def test_expand_llm_needs_llm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    source = tmp_path / "a.txt"
    source.write_text("متن", encoding="utf-8")
    index = tmp_path / "idx"
    _ingest([str(source)], index)
    assert main(["query", "متن", "--index", str(index), "--expand", "llm"]) == 1
    assert "--expand llm needs" in capsys.readouterr().err


def test_llm_from_args_builds_adapters(monkeypatch: pytest.MonkeyPatch) -> None:
    from persian_rag.llm import OllamaLLM, OpenAICompatibleLLM

    parser = cli.build_parser()
    args = parser.parse_args(
        ["query", "x", "--llm", "ollama", "--ollama-url", "http://h:1", "--llm-max-tokens", "9"]
    )
    llm = cli._llm_from_args(args)
    assert isinstance(llm, OllamaLLM)
    assert llm.base_url == "http://h:1" and llm.max_tokens == 9
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    args = parser.parse_args(["query", "x", "--llm", "openai", "--llm-model", "m"])
    openai = cli._llm_from_args(args)
    assert isinstance(openai, OpenAICompatibleLLM) and openai.model == "m"
    assert cli._llm_from_args(parser.parse_args(["query", "x"])) is None


def test_chat_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "a.txt"
    source.write_text("فردوسی شاهنامه را سرود و داستان رستم را گفت.", encoding="utf-8")
    index = tmp_path / "idx"
    _ingest([str(source)], index)
    lines = iter(["شاهنامه را چه کسی سرود", "و رستم؟", "/reset", "", "/exit"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(lines))
    capsys.readouterr()
    assert main(["chat", "--index", str(index), "--k", "1"]) == 0
    out = capsys.readouterr().out
    assert "[1] a.txt" in out
    assert "(searching: شاهنامه را چه کسی سرود و رستم؟)" in out
    assert "(conversation cleared)" in out


def test_chat_command_eof(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "a.txt"
    source.write_text("متن", encoding="utf-8")
    index = tmp_path / "idx"
    _ingest([str(source)], index)

    def eof(prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", eof)
    assert main(["chat", "--index", str(index)]) == 0
