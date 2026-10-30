import json
import subprocess
import sys
from pathlib import Path

import pytest

from persian_rag.cli import main


def test_version_flag() -> None:
    with pytest.raises(SystemExit) as caught:
        main(["--version"])
    assert caught.value.code == 0


def test_ingest_and_query_roundtrip(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hafez.txt").write_text(
        "حافظ در شهر شیراز زاده شد.\n",
        encoding="utf-8",
    )
    (docs / "sport.md").write_text(
        "# ورزش\n\nفوتبال یک بازی گروهی است.\n",
        encoding="utf-8",
    )
    marked = docs / "vowels.txt"
    marked.write_text("کِتاب روی میز است.\n", encoding="utf-8")
    index = tmp_path / "index"
    status = main(
        [
            "ingest",
            str(docs),
            "--index",
            str(index),
            "--embedder",
            "hashing",
            "--dimension",
            "64",
            "--remove-diacritics",
        ]
    )
    assert status == 0
    captured = capsys.readouterr()
    assert "Indexed 3 document" in captured.out
    stored = (index / "index.json").read_text(encoding="utf-8")
    assert "کتاب" in stored
    assert "کِتاب" not in stored

    status = main(
        [
            "query",
            "حافظ",
            "--index",
            str(index),
            "--json",
            "--k",
            "2",
        ]
    )
    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["answer"] == ""
    assert any("حافظ" in item["text"] for item in payload["contexts"])
    assert payload["contexts"][0]["text"].find("حافظ") != -1


def test_query_reports_a_missing_index(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    status = main(["query", "سلام", "--index", str(tmp_path / "missing")])
    assert status == 1
    assert "error:" in capsys.readouterr().err


def test_quickstart_example() -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [sys.executable, str(root / "examples" / "quickstart.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "شیراز" in completed.stdout


def _ingest_corpus(tmp_path: Path, *extra: str) -> Path:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "hafez.txt").write_text("حافظ در شهر شیراز زاده شد.\n", encoding="utf-8")
    (docs / "ferdowsi.txt").write_text("فردوسی شاهنامه را در توس سرود.\n", encoding="utf-8")
    (docs / "sport.md").write_text("# ورزش\n\nفوتبال یک بازی گروهی است.\n", encoding="utf-8")
    index = tmp_path / "index"
    status = main(["ingest", str(docs), "--index", str(index), "--dimension", "64", *extra])
    assert status == 0
    return index


def test_ingest_with_embedding_cache(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cache = tmp_path / "cache.sqlite"
    _ingest_corpus(tmp_path, "--embedding-cache", str(cache))
    assert cache.exists()
    from persian_rag.cache import SQLiteEmbeddingCache

    with SQLiteEmbeddingCache(cache) as store:
        assert len(store) == 3


def test_eval_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index = _ingest_corpus(tmp_path)
    dataset = tmp_path / "eval.jsonl"
    dataset.write_text(
        '{"question": "حافظ کجا زاده شد؟", "relevant_ids": ["hafez.txt"]}\n'
        '{"question": "شاهنامه را چه کسی سرود؟", "relevant_ids": ["ferdowsi.txt"]}\n',
        encoding="utf-8",
    )
    capsys.readouterr()
    assert main(["eval", str(dataset), "--index", str(index), "--k", "1", "3"]) == 0
    out = capsys.readouterr().out
    assert "Questions: 2" in out
    assert "hit_rate" in out
    assert main(["eval", str(dataset), "--index", str(index), "--json", "--k", "2"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["retrieval"]["metrics"]["@2"]["hit_rate"] == 1.0
    assert main(["eval", str(dataset), "--index", str(index), "--answers"]) == 1
    assert "--answers needs --llm" in capsys.readouterr().err


def test_rerank_llm_requires_llm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    index = _ingest_corpus(tmp_path)
    assert main(["query", "حافظ", "--index", str(index), "--rerank", "llm"]) == 1
    assert "--rerank llm" in capsys.readouterr().err


def test_serve_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pytest.importorskip("fastapi")
    uvicorn = pytest.importorskip("uvicorn")
    index = _ingest_corpus(tmp_path)
    calls: dict[str, object] = {}

    def fake_run(app: object, host: str, port: int) -> None:
        calls.update(app=app, host=host, port=port)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    assert main(["serve", "--index", str(index), "--port", "9001"]) == 0
    assert calls["host"] == "127.0.0.1"
    assert calls["port"] == 9001
