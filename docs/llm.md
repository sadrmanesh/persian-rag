# LLM adapters

New in 0.3.0. Pipelines accept any callable `str -> str`; these adapters cover the common servers without extra packages.

```python
from persian_rag import OllamaLLM, OpenAICompatibleLLM, create_llm

openai = OpenAICompatibleLLM(model="gpt-4o-mini")  # OPENAI_API_KEY, OPENAI_BASE_URL
local = OpenAICompatibleLLM(
    model="qwen2.5", base_url="http://localhost:8000/v1", require_api_key=False
)
ollama = OllamaLLM(model="qwen2.5:7b")  # OLLAMA_HOST or http://localhost:11434

llm = create_llm("ollama", model="aya-expanse:8b", max_tokens=512)
llm("سلام")  # plain prompt
llm.chat([{"role": "user", "content": "سلام"}])  # message list
```

`OpenAICompatibleLLM` works with any server that implements `/chat/completions`: hosted APIs, vLLM, llama.cpp server, LM Studio, and others. `extra_body` adds provider-specific fields. `OllamaLLM` uses `/api/chat` without streaming and maps `max_tokens` to `num_predict`; `keep_alive` controls how long the model stays loaded.

Shared settings: `system_prompt`, `temperature` (default 0), `max_tokens`, `timeout`, `max_retries` (default 2), and `retry_backoff`. Timeouts, HTTP 408, 409, 429, and 5xx responses are retried with exponential backoff; other errors raise `LLMRequestError` with the status code. API keys are never included in `repr()` or saved indexes.

For tests, pass `transport=` with a function `(url, payload, headers, timeout) -> dict` to replace the HTTP call.

## Command line

```bash
persian-rag query "..." --llm openai --llm-model gpt-4o-mini --llm-max-tokens 400
persian-rag query "..." --llm ollama --llm-model qwen2.5:7b --ollama-url http://gpu-box:11434
```
