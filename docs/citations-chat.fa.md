# ارجاع به منبع و گفت و گو

این امکانات از نسخه 0.3.0 اضافه شده اند.

## پاسخ مستند به منبع

```python
from persian_rag import RAGPipeline, format_sources

pipeline = RAGPipeline(retriever, llm=llm, citations=True)
answer = pipeline.query("حافظ اهل کجا بود؟")

print(answer.answer)  # "حافظ اهل شیراز بود [1]."
for citation in answer.citations:
    print(citation.marker, citation.document_id, citation.quote, citation.start, citation.end)
print(answer.uncited)
```

با `citations=True` دستور پیش فرض هر متن بازیابی شده را شماره گذاری می کند (`[1]`، `[2]` و ...) و از مدل می خواهد پس از هر ادعا شماره منبع را بیاورد. سپس پاسخ تجزیه می شود:

* نشانه هایی مانند `[1]`، `[1, 3]`، `[۲]` و `[٢،٣]` شناخته می شوند.
* برای هر ادعای دارای ارجاع، شیء `Citation` متن مرجع، بهترین جمله پشتیبان در آن متن (`quote`) و بازه نویسه ای آن را نگه می دارد. `start` و `end` جایگاه در متن یکدست شده سند، `chunk_start` و `chunk_end` جایگاه در تکه و `answer_start` و `answer_end` جایگاه ادعا در پاسخ هستند.
* `answer.uncited` جمله های محتوایی بدون ارجاع را فهرست می کند تا پاسخ های کم پشتوانه را بتوان مشخص کرد. تابع `extract_citations` شماره هایی را که به هیچ متنی اشاره نمی کنند در `invalid_markers` گزارش می کند.

توابع `extract_citations(answer_text, contexts)` و `format_sources(report)` به تنهایی و با خروجی هر مدلی کار می کنند.

## حافظه گفت و گو

```python
from persian_rag import ChatSession, ConversationMemory

chat = ChatSession(pipeline, memory=ConversationMemory(max_turns=6))
chat.ask("حافظ اهل کجا بود؟")
reply = chat.ask("آرامگاهش کجاست؟")
print(reply.standalone_question)
```

پرسش های دنباله دار پیش از بازیابی به پرسش مستقل بازنویسی می شوند:

| `condense` | رفتار |
| --- | --- |
| `auto` (پیش فرض) | از مدل بازنویسی (یا مدل همگام خط لوله) استفاده می کند و اگر مدلی نباشد به `heuristic` برمی گردد |
| `llm` | همیشه از مدل می پرسد |
| `heuristic` | اگر پیام دنباله دار به نظر برسد (بسیار کوتاه، آغاز با «و» یا «اما»، واژه ارجاعی یا ضمیر ملکی مانند «پایتختش») پرسش قبلی را به آن اضافه می کند |
| `none` | هرگز بازنویسی نمی کند |

نوبت های اخیر به دستور پاسخ نیز افزوده می شوند. `ConversationMemory` آخرین `max_turns` نوبت را نگه می دارد، پاسخ های بلند را کوتاه می کند و با `to_dict()` و `from_dict()` ذخیره می شود. `aask` نسخه ناهمگام است و `reset()` گفت و گو را پاک می کند.

## رابط HTTP

درخواست `POST /chat` با بدنه `{"message": "...", "session_id": null}` گفت و گوی تازه ای می سازد و `session_id` آن را برمی گرداند. برای ادامه همان شناسه را بفرستید. `DELETE /chat/{session_id}` گفت و گو را پایان می دهد. اگر خط لوله `citations=True` داشته باشد، پاسخ `/query` و `/chat` شامل `citations` و `uncited` است. گفت و گوها در حافظه سرور نگه داشته می شوند (حداکثر `max_sessions` و قدیمی ترین ها زودتر حذف می شوند).

## خط فرمان

```bash
persian-rag query "حافظ اهل کجا بود؟" --llm ollama --citations
persian-rag chat --llm openai --citations
```

در فرمان `chat`، دستور `/reset` حافظه را پاک می کند و `/exit` خارج می شود.
