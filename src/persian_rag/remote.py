"""HTTP helpers for OpenAI-compatible embedding and chat endpoints."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from typing import Any, cast

from persian_rag.errors import PersianRagError

_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def post_json(
    url: str,
    payload: Mapping[str, Any],
    *,
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    """POST ``payload`` as JSON and return the decoded object."""

    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise PersianRagError(f"HTTP {exc.code} calling {url}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise PersianRagError(f"Request to {url} failed: {exc.reason}") from exc

    if not isinstance(raw, (bytes, bytearray)):
        raise PersianRagError(f"Unexpected response body from {url}")
    try:
        parsed = json.loads(bytes(raw).decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise PersianRagError(f"Response from {url} was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise PersianRagError(f"Response from {url} was not a JSON object")
    return cast(dict[str, Any], parsed)


def embed_texts(
    texts: list[str],
    *,
    model: str,
    api_key: str,
    base_url: str = _DEFAULT_BASE_URL,
    timeout: float = 60.0,
    dimensions: int | None = None,
    batch_size: int = 64,
) -> list[list[float]]:
    """Call an OpenAI-compatible ``/embeddings`` endpoint."""

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not texts:
        return []

    vectors: list[list[float]] = []
    root = base_url.rstrip("/")
    for offset in range(0, len(texts), batch_size):
        batch = texts[offset : offset + batch_size]
        payload: dict[str, Any] = {"model": model, "input": batch}
        if dimensions is not None:
            payload["dimensions"] = dimensions
        parsed = post_json(
            f"{root}/embeddings",
            payload,
            api_key=api_key,
            timeout=timeout,
        )
        vectors.extend(_parse_embeddings(parsed, expected=len(batch)))
    return vectors


def chat_complete(
    prompt: str,
    *,
    model: str,
    api_key: str,
    base_url: str = _DEFAULT_BASE_URL,
    timeout: float = 60.0,
    temperature: float = 0.0,
) -> str:
    """Call an OpenAI-compatible ``/chat/completions`` endpoint."""

    root = base_url.rstrip("/")
    parsed = post_json(
        f"{root}/chat/completions",
        {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        },
        api_key=api_key,
        timeout=timeout,
    )
    choices = parsed.get("choices")
    if not isinstance(choices, list) or not choices:
        raise PersianRagError("Chat response did not include choices")
    first = choices[0]
    if not isinstance(first, dict):
        raise PersianRagError("Chat response choice was not an object")
    message = first.get("message")
    if not isinstance(message, dict):
        raise PersianRagError("Chat response did not include a message")
    content = message.get("content")
    if not isinstance(content, str):
        raise PersianRagError("Chat response content was not a string")
    return content


def make_openai_llm(
    *,
    model: str,
    api_key: str,
    base_url: str = _DEFAULT_BASE_URL,
    timeout: float = 60.0,
    temperature: float = 0.0,
) -> Callable[[str], str]:
    """Return an LLM callable compatible with :class:`persian_rag.RAGPipeline`."""

    if not api_key:
        raise PersianRagError("OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key.")

    def complete(prompt: str) -> str:
        return chat_complete(
            prompt,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            temperature=temperature,
        )

    return complete


def _parse_embeddings(payload: Mapping[str, Any], *, expected: int) -> list[list[float]]:
    data = payload.get("data")
    if not isinstance(data, list):
        raise PersianRagError("Embedding response did not include data")
    rows: list[tuple[int, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise PersianRagError("Embedding item was not an object")
        index = item.get("index")
        embedding = item.get("embedding")
        if not isinstance(index, int):
            raise PersianRagError("Embedding item is missing an index")
        rows.append((index, _as_vector(embedding)))
    rows.sort(key=lambda item: item[0])
    if len(rows) != expected:
        raise PersianRagError(f"Expected {expected} embeddings, received {len(rows)}")
    if [index for index, _ in rows] != list(range(expected)):
        raise PersianRagError("Embedding response indexes were incomplete")
    return [vector for _, vector in rows]


def _as_vector(value: Any) -> list[float]:
    if not isinstance(value, list) or not value:
        raise PersianRagError("Embedding vector was empty")
    vector: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PersianRagError("Embedding values must be numeric")
        vector.append(float(item))
    return vector
