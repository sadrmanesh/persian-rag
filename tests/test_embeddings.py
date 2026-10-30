import io
import json
import urllib.error
import urllib.request
from email.message import Message

import pytest

from persian_rag.embeddings import (
    EmbedderSpec,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    create_embedder,
)
from persian_rag.errors import ConfigError, PersianRagError
from persian_rag.remote import chat_complete, make_openai_llm


class _Response:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False


def test_hashing_embedder_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(dimension=32, ngram_range=(2, 3), seed=7)
    first = embedder.embed_query("حافظ در شیراز")
    second = embedder.embed_query("حافظ در شیراز")
    other = embedder.embed_documents(["فوتبال گروهی"])[0]
    assert first == second
    assert len(first) == 32
    assert abs(sum(value * value for value in first) - 1.0) < 1e-9
    assert first != other
    empty = embedder.embed_query("")
    assert empty == [0.0] * 32
    assert embedder.embed_documents([]) == []


def test_hashing_spec_roundtrip() -> None:
    spec = HashingEmbedder(dimension=16, ngram_range=(1, 2), seed=3).to_spec()
    restored = create_embedder(EmbedderSpec.from_dict(spec.to_dict()))
    assert isinstance(restored, HashingEmbedder)
    assert restored.embed_query("کتاب") == HashingEmbedder(
        16, ngram_range=(1, 2), seed=3
    ).embed_query("کتاب")


def test_openai_embeddings_are_ordered_by_index(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
        del timeout
        assert request.full_url == "https://example.test/v1/embeddings"
        payload = json.loads(request.data.decode("utf-8"))  # type: ignore[union-attr]
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["سلام", "دنیا"]
        body = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        return _Response(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(
        api_key="test-key",
        base_url="https://example.test/v1",
        dimensions=2,
    )
    assert embedder.embed_documents(["سلام", "دنیا"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert embedder.dimension == 2


def test_openai_http_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
        del timeout
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "Unauthorized",
            Message(),
            io.BytesIO(b'{"error":"no"}'),
        )

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    embedder = OpenAICompatibleEmbedder(api_key="test-key", dimensions=2)
    with pytest.raises(PersianRagError, match="401"):
        embedder.embed_query("سلام")


def test_openai_chat_and_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(request: urllib.request.Request, timeout: float = 0) -> _Response:
        del timeout
        body = {"choices": [{"message": {"content": "شیراز"}}]}
        return _Response(json.dumps(body).encode("utf-8"))

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    assert (
        chat_complete(
            "پرسش", model="gpt-4o-mini", api_key="test-key", base_url="https://example.test/v1"
        )
        == "شیراز"
    )
    with pytest.raises(PersianRagError):
        make_openai_llm(model="gpt-4o-mini", api_key="")


def test_openai_embedder_requires_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        OpenAICompatibleEmbedder()


def test_invalid_embedder_spec() -> None:
    with pytest.raises(ConfigError):
        EmbedderSpec.from_dict({"kind": "local", "dimension": 4})
    with pytest.raises(ConfigError):
        EmbedderSpec(kind="hashing", dimension=4, ngram_min=3, ngram_max=1)
