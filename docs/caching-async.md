# Caching and async

## Embedding cache

```python
from persian_rag import CachedEmbedder, OpenAICompatibleEmbedder, SQLiteEmbeddingCache

embedder = CachedEmbedder(
    OpenAICompatibleEmbedder(model="text-embedding-3-small"),
    SQLiteEmbeddingCache(".cache/embeddings.sqlite"),
)
```

`CachedEmbedder` wraps any embedder. Only texts missing from the cache are sent to the model, in one batch, and duplicates inside a batch are embedded once. Keys are SHA-256 hashes of the embedder settings, the vector kind (document or query), and the text, so changing the model or dimension never returns stale vectors.

| Cache | Storage | Notes |
| --- | --- | --- |
| `InMemoryEmbeddingCache(max_size=100_000, ttl=None)` | process memory | LRU, the default |
| `SQLiteEmbeddingCache(path)` | one SQLite file | standard library only, thread safe |

From the CLI: `persian-rag ingest ./library --embedding-cache .cache/embeddings.sqlite`.

## Result cache

```python
from persian_rag import LRUCache

retriever.result_cache = LRUCache(max_size=512, ttl=300)
```

Repeated queries with the same normalized text, `k`, and mode are answered from memory. The cache is cleared whenever chunks are added. You can also pass `result_cache=` to `HybridRetriever`.

## Async

```python
import asyncio


async def llm(prompt: str) -> str: ...  # an async HTTP call to your model


pipeline = RAGPipeline(retriever, llm)


async def main() -> None:
    answers = await asyncio.gather(
        pipeline.aquery("حافظ اهل کجا بود؟"),
        pipeline.aquery("شاهنامه را چه کسی سرود؟"),
    )
```

- `HybridRetriever.aretrieve` and `RAGPipeline.aretrieve` run retrieval in a worker thread.
- `RAGPipeline.aquery` awaits async model functions (including objects with an `async def __call__`) and runs synchronous ones in a worker thread, so the event loop is never blocked.
- `RAGPipeline.query` raises `TypeError` when the model is async.
