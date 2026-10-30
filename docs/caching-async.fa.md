# حافظه نهان و اجرای ناهمگام

## حافظه نهان بردارها

```python
from persian_rag import CachedEmbedder, OpenAICompatibleEmbedder, SQLiteEmbeddingCache

embedder = CachedEmbedder(
    OpenAICompatibleEmbedder(model="text-embedding-3-small"),
    SQLiteEmbeddingCache(".cache/embeddings.sqlite"),
)
```

کلاس `CachedEmbedder` دور هر بردارساز پیچیده می شود. فقط متن هایی که در حافظه نهان نیستند به مدل فرستاده می شوند و متن های تکراری یک بار بردار می گیرند. کلید هر بردار از تنظیمات مدل، نوع بردار و خود متن ساخته می شود، پس با تغییر مدل هیچ بردار کهنه ای برنمی گردد.

| حافظه | محل نگهداری | توضیح |
| --- | --- | --- |
| `InMemoryEmbeddingCache` | حافظه برنامه | پیش فرض، با سیاست LRU |
| `SQLiteEmbeddingCache` | یک فایل SQLite | بدون وابستگی و ایمن برای چند رشته |

در خط فرمان: `persian-rag ingest ./library --embedding-cache .cache/embeddings.sqlite`

## حافظه نهان نتیجه ها

```python
from persian_rag import LRUCache

retriever.result_cache = LRUCache(max_size=512, ttl=300)
```

پرسش های تکراری با همان متن یکدست شده، همان `k` و همان حالت از حافظه پاسخ داده می شوند. با افزودن تکه تازه، این حافظه پاک می شود.

## اجرای ناهمگام

```python
import asyncio


async def llm(prompt: str) -> str: ...


pipeline = RAGPipeline(retriever, llm)


async def main() -> None:
    answers = await asyncio.gather(
        pipeline.aquery("حافظ اهل کجا بود؟"),
        pipeline.aquery("شاهنامه را چه کسی سرود؟"),
    )
```

- متدهای `aretrieve` بازیابی را در یک رشته جداگانه اجرا می کنند.
- متد `aquery` مدل ناهمگام را مستقیم منتظر می ماند و مدل همگام را در رشته جداگانه اجرا می کند تا حلقه رویداد مسدود نشود.
- اگر مدل ناهمگام باشد، متد `query` خطای `TypeError` می دهد.
