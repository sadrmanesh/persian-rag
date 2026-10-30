from __future__ import annotations

import io
import json
import urllib.error
from collections.abc import Mapping
from typing import Any

import pytest

from persian_rag.errors import ConfigError
from persian_rag.llm import (
    LLMRequestError,
    OllamaLLM,
    OpenAICompatibleLLM,
    create_llm,
    urllib_transport,
)


class Recorder:
    def __init__(self, *responses: Any) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], dict[str, str]]] = []

    def __call__(
        self, url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
    ) -> dict[str, Any]:
        self.calls.append((url, dict(payload), dict(headers)))
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def test_openai_payload_and_reply() -> None:
    transport = Recorder({"choices": [{"message": {"content": "سلام"}}]})
    llm = OpenAICompatibleLLM(
        model="m",
        api_key="k",
        base_url="http://server/v1/",
        max_tokens=50,
        system_prompt="فارسی پاسخ بده",
        transport=transport,
    )
    assert llm("پرسش") == "سلام"
    url, payload, headers = transport.calls[0]
    assert url == "http://server/v1/chat/completions"
    assert payload["max_tokens"] == 50
    assert payload["messages"][0] == {"role": "system", "content": "فارسی پاسخ بده"}
    assert headers["Authorization"] == "Bearer k"
    assert "api_key='k'" not in repr(llm)


def test_openai_requires_key_unless_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenAICompatibleLLM()
    transport = Recorder({"choices": [{"message": {"content": "ok"}}]})
    llm = OpenAICompatibleLLM(require_api_key=False, base_url="http://local", transport=transport)
    assert llm("x") == "ok"
    assert "Authorization" not in transport.calls[0][2]


def test_ollama_payload_and_options(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "gpu-box:11434")
    transport = Recorder({"message": {"role": "assistant", "content": "پاسخ"}})
    llm = OllamaLLM(
        model="qwen2.5:7b",
        max_tokens=64,
        options={"num_ctx": 8192},
        keep_alive="5m",
        transport=transport,
    )
    assert llm.chat([{"role": "user", "content": "سلام"}]) == "پاسخ"
    url, payload, _ = transport.calls[0]
    assert url == "http://gpu-box:11434/api/chat"
    assert payload["stream"] is False
    assert payload["options"] == {"temperature": 0.0, "num_predict": 64, "num_ctx": 8192}
    assert payload["keep_alive"] == "5m"


def test_ollama_error_body() -> None:
    llm = OllamaLLM(base_url="http://x", transport=Recorder({"error": "model not found"}))
    with pytest.raises(LLMRequestError, match="model not found"):
        llm("x")


def test_retries_transient_errors_only() -> None:
    sleeps: list[float] = []
    transport = Recorder(
        LLMRequestError("busy", status=429),
        LLMRequestError("down", status=503),
        {"message": {"content": "ok"}},
    )
    llm = OllamaLLM(base_url="http://x", transport=transport, sleep=sleeps.append)
    assert llm("x") == "ok"
    assert sleeps == [1.0, 2.0]
    bad = OllamaLLM(
        base_url="http://x",
        transport=Recorder(LLMRequestError("no", status=400)),
        sleep=sleeps.append,
    )
    with pytest.raises(LLMRequestError):
        bad("x")
    exhausted = OllamaLLM(
        base_url="http://x",
        max_retries=1,
        transport=Recorder(LLMRequestError("a", status=500), LLMRequestError("b", status=500)),
        sleep=lambda _: None,
    )
    with pytest.raises(LLMRequestError, match="b"):
        exhausted("x")


def test_malformed_openai_reply() -> None:
    llm = OpenAICompatibleLLM(api_key="k", transport=Recorder({"choices": []}))
    with pytest.raises(LLMRequestError):
        llm("x")


def test_create_llm() -> None:
    assert isinstance(create_llm("ollama", model="llama3", base_url="http://x"), OllamaLLM)
    llm = create_llm("openai", model="gpt", api_key="k", max_tokens=5)
    assert isinstance(llm, OpenAICompatibleLLM) and llm.max_tokens == 5
    with pytest.raises(ConfigError):
        create_llm("other")


def test_urllib_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response(io.BytesIO):
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

    seen: dict[str, Any] = {}

    def fake_urlopen(request: Any, timeout: float) -> Response:
        seen["body"] = json.loads(request.data)
        seen["headers"] = dict(request.header_items())
        return Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    assert urllib_transport("http://x", {"a": "ب"}, {"X-Test": "1"}, 5) == {"ok": True}
    assert seen["body"] == {"a": "ب"}

    def failing(request: Any, timeout: float) -> Response:
        raise urllib.error.HTTPError("http://x", 503, "busy", {}, io.BytesIO(b"later"))  # type: ignore[arg-type]

    monkeypatch.setattr("urllib.request.urlopen", failing)
    with pytest.raises(LLMRequestError) as info:
        urllib_transport("http://x", {}, {}, 5)
    assert info.value.status == 503 and info.value.retryable
