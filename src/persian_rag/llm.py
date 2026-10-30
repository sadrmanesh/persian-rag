"""Language model adapters: OpenAI-compatible APIs and local Ollama.

Both adapters are plain callables (``llm(prompt) -> str``) so they plug into
:class:`~persian_rag.pipeline.RAGPipeline`, the rerankers, the query
rewriter, and the chat session. They also expose :meth:`chat` for a list of
messages. Only the standard library is used.

* :class:`OpenAICompatibleLLM` talks to ``/chat/completions`` on OpenAI or
  any compatible server (vLLM, llama.cpp server, LM Studio, OpenRouter, and
  Ollama's ``/v1`` endpoint).
* :class:`OllamaLLM` talks to Ollama's native ``/api/chat`` endpoint, which
  also accepts model options such as ``num_ctx``.

Transient failures (HTTP 408, 409, 429, 5xx, and connection errors) are
retried with exponential backoff.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from persian_rag.errors import ConfigError, PersianRagError

Message = Mapping[str, str]
Transport = Callable[[str, Mapping[str, Any], Mapping[str, str], float], dict[str, Any]]
"""``(url, payload, headers, timeout) -> decoded JSON object``."""

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
_RETRY_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


class LLMRequestError(PersianRagError):
    """A model server returned an error or an unexpected response."""

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status

    @property
    def retryable(self) -> bool:
        return self.status is None or self.status in _RETRY_STATUS


def urllib_transport(
    url: str, payload: Mapping[str, Any], headers: Mapping[str, str], timeout: float
) -> dict[str, Any]:
    """POST JSON with :mod:`urllib` and decode the JSON object reply."""

    body = json.dumps(dict(payload), ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json", **dict(headers)},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise LLMRequestError(f"HTTP {exc.code} from {url}: {detail}", status=exc.code) from exc
    except urllib.error.URLError as exc:
        raise LLMRequestError(f"Request to {url} failed: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMRequestError(f"Request to {url} timed out") from exc
    try:
        parsed = json.loads(bytes(raw).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LLMRequestError(f"Response from {url} was not valid JSON", status=0) from exc
    if not isinstance(parsed, dict):
        raise LLMRequestError(f"Response from {url} was not a JSON object", status=0)
    return cast(dict[str, Any], parsed)


@dataclass(slots=True)
class _BaseLLM:
    model: str
    system_prompt: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None
    timeout: float = 120.0
    max_retries: int = 2
    retry_backoff: float = 1.0
    transport: Transport = field(default=urllib_transport, repr=False)
    sleep: Callable[[float], None] = field(default=time.sleep, repr=False)

    def __call__(self, prompt: str) -> str:
        return self.chat([{"role": "user", "content": prompt}])

    def chat(self, messages: Sequence[Message]) -> str:
        """Send ``messages`` (``role``/``content`` mappings) and return the reply."""

        prepared = [dict(message) for message in messages]
        if not prepared:
            raise ValueError("messages must not be empty")
        if self.system_prompt and prepared[0].get("role") != "system":
            prepared.insert(0, {"role": "system", "content": self.system_prompt})
        return self._extract(self._post(self._payload(prepared)))

    def _post(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        attempt = 0
        while True:
            try:
                return self.transport(self._url(), payload, self._headers(), self.timeout)
            except LLMRequestError as exc:
                if not exc.retryable or attempt >= self.max_retries:
                    raise
                self.sleep(self.retry_backoff * (2**attempt))
                attempt += 1

    def _url(self) -> str:
        raise NotImplementedError

    def _headers(self) -> dict[str, str]:
        return {}

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        raise NotImplementedError

    def _extract(self, response: Mapping[str, Any]) -> str:
        raise NotImplementedError


@dataclass(slots=True)
class OpenAICompatibleLLM(_BaseLLM):
    """Chat model behind an OpenAI-compatible ``/chat/completions`` API.

    ``api_key`` defaults to ``$OPENAI_API_KEY`` and ``base_url`` to
    ``$OPENAI_BASE_URL`` or the OpenAI endpoint. Servers that need no key
    (a local vLLM, for example) work with ``require_api_key=False``.
    """

    model: str = "gpt-4o-mini"
    api_key: str | None = field(default=None, repr=False)
    base_url: str | None = None
    require_api_key: bool = True
    extra_body: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
        if self.require_api_key and not self.api_key:
            raise ConfigError("OpenAI API key is missing. Set OPENAI_API_KEY or pass api_key.")

    def _url(self) -> str:
        return f"{(self.base_url or DEFAULT_OPENAI_BASE_URL).rstrip('/')}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": messages,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        payload.update(self.extra_body)
        return payload

    def _extract(self, response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise LLMRequestError("Chat response did not include choices", status=0)
        message = choices[0].get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMRequestError("Chat response content was not a string", status=0)
        return content


@dataclass(slots=True)
class OllamaLLM(_BaseLLM):
    """Local model served by Ollama (``ollama serve``).

    ``base_url`` defaults to ``$OLLAMA_HOST`` or ``http://localhost:11434``.
    ``options`` is passed through (for example ``{"num_ctx": 8192}``);
    ``temperature`` and ``max_tokens`` map to ``temperature`` and
    ``num_predict``. ``keep_alive`` controls how long the model stays loaded.
    """

    model: str = "qwen2.5:7b"
    base_url: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    keep_alive: str | None = None

    def __post_init__(self) -> None:
        if self.base_url is None:
            host = os.environ.get("OLLAMA_HOST", "") or DEFAULT_OLLAMA_URL
            self.base_url = host if "://" in host else f"http://{host}"

    def _url(self) -> str:
        return f"{(self.base_url or DEFAULT_OLLAMA_URL).rstrip('/')}/api/chat"

    def _payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        options: dict[str, Any] = {"temperature": self.temperature}
        if self.max_tokens is not None:
            options["num_predict"] = self.max_tokens
        options.update(self.options)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        return payload

    def _extract(self, response: Mapping[str, Any]) -> str:
        if isinstance(response.get("error"), str):
            raise LLMRequestError(f"Ollama error: {response['error']}", status=0)
        message = response.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise LLMRequestError("Ollama response did not include message content", status=0)
        return content


LLMProvider = Literal["openai", "ollama"]


def create_llm(
    provider: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> OpenAICompatibleLLM | OllamaLLM:
    """Build an adapter by name: ``"openai"`` or ``"ollama"``."""

    if provider == "openai":
        options: dict[str, Any] = dict(kwargs)
        if model:
            options["model"] = model
        return OpenAICompatibleLLM(api_key=api_key, base_url=base_url, **options)
    if provider == "ollama":
        options = dict(kwargs)
        if model:
            options["model"] = model
        return OllamaLLM(base_url=base_url, **options)
    raise ConfigError(f"Unknown LLM provider: {provider!r}. Expected openai or ollama")
