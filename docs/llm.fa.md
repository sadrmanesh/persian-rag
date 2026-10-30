# اتصال به مدل های زبانی

خط لوله هر تابع `str -> str` را می پذیرد. از نسخه 0.3.0 این اتصال ها بدون بسته اضافه سرورهای رایج را پوشش می دهند.

```python
from persian_rag import OllamaLLM, OpenAICompatibleLLM, create_llm

openai = OpenAICompatibleLLM(model="gpt-4o-mini")
local = OpenAICompatibleLLM(
    model="qwen2.5", base_url="http://localhost:8000/v1", require_api_key=False
)
ollama = OllamaLLM(model="qwen2.5:7b")

llm = create_llm("ollama", model="aya-expanse:8b", max_tokens=512)
llm("سلام")
llm.chat([{"role": "user", "content": "سلام"}])
```

`OpenAICompatibleLLM` با هر سروری که `/chat/completions` را پیاده کرده باشد کار می کند، از جمله سرویس های میزبانی شده، vLLM، سرور llama.cpp و LM Studio. کلید از `OPENAI_API_KEY` و نشانی از `OPENAI_BASE_URL` خوانده می شود و `extra_body` فیلدهای ویژه هر سرویس را اضافه می کند. `OllamaLLM` از `/api/chat` بدون جریان پیوسته استفاده می کند، نشانی را از `OLLAMA_HOST` یا `http://localhost:11434` می خواند و `max_tokens` را به `num_predict` تبدیل می کند. `keep_alive` تعیین می کند مدل چه مدت در حافظه بماند.

تنظیمات مشترک: `system_prompt`، `temperature` (پیش فرض صفر)، `max_tokens`، `timeout`، `max_retries` (پیش فرض ۲) و `retry_backoff`. پایان مهلت و پاسخ های HTTP با کد 408، 409، 429 و 5xx با فاصله فزاینده دوباره امتحان می شوند و خطاهای دیگر `LLMRequestError` همراه کد وضعیت می دهند. کلیدهای API هرگز در `repr()` یا ایندکس ذخیره شده نمی آیند.

برای آزمون، پارامتر `transport` را با تابعی به شکل `(url, payload, headers, timeout) -> dict` بدهید تا جای فراخوانی HTTP را بگیرد.

## خط فرمان

```bash
persian-rag query "..." --llm openai --llm-model gpt-4o-mini --llm-max-tokens 400
persian-rag query "..." --llm ollama --llm-model qwen2.5:7b --ollama-url http://gpu-box:11434
```
