from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "run_benchmark.py"


def test_benchmark_script_runs_offline(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = importlib.util.spec_from_file_location("run_benchmark", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "run_benchmark", module)
    spec.loader.exec_module(module)

    output = tmp_path / "results.json"
    assert module.main(["--repeat", "1", "--dimension", "128", "--json", str(output)]) == 0
    table = capsys.readouterr().out
    assert "| hybrid + rewrite |" in table
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert len(payload["rows"]) == 6
    for row in payload["rows"]:
        assert 0.0 <= row["hit1"] <= row["hit5"] <= 1.0


def test_benchmark_data_is_consistent() -> None:
    data = SCRIPT.parent / "data"
    corpus = [json.loads(line) for line in (data / "corpus.jsonl").read_text("utf-8").splitlines()]
    questions = [json.loads(line) for line in (data / "qa.jsonl").read_text("utf-8").splitlines()]
    ids = {doc["id"] for doc in corpus}
    assert len(ids) == len(corpus) >= 24
    assert len(questions) >= 40
    for item in questions:
        assert set(item["relevant_ids"]) <= ids
    for path in data.iterdir():
        text = path.read_text(encoding="utf-8")
        assert "\u200c" not in text
