# Retrieval and reranking

## Hybrid retrieval

```python
from persian_rag import HybridRetriever

retriever = HybridRetriever(embedder, rrf_k=60, dense_weight=1.0, sparse_weight=1.0)
retriever.add_chunks(chunks)
hits = retriever.retrieve("حافظ اهل کجا بود؟", k=5, mode="hybrid")
```

| Mode | Ranking | Score |
| --- | --- | --- |
| `hybrid` | Reciprocal rank fusion of BM25 and dense lists | Weighted RRF |
| `dense` | Vector store search | Store score (cosine by default) |
| `sparse` | Okapi BM25 over Persian tokens | BM25 |

Each `RetrievalResult` has `chunk_id`, `document_id`, `text`, `score`, character offsets, `metadata`, and the 1-based `dense_rank` and `sparse_rank` inside the candidate pool.

## Reranking

A reranker reorders a larger candidate pool with a more expensive model. Plug one into the pipeline:

```python
from persian_rag import CrossEncoderReranker, RAGPipeline

pipeline = RAGPipeline(
    retriever,
    llm,
    reranker=CrossEncoderReranker(),  # BAAI/bge-reranker-v2-m3
    rerank_candidates=20,
)
answer = pipeline.query("پرسش", k=4)
```

The retriever fetches `max(k, rerank_candidates)` chunks, and the reranker returns the best `k`. Reranked results carry the new score in `score` and the first-stage score in `retrieval_score`.

### Cross-encoder

```bash
pip install "persian-rag[rerank]"
```

`CrossEncoderReranker(model_name=..., device=None, batch_size=16, max_length=512)` scores each (query, passage) pair. The default model is multilingual and handles Persian. Pass `model=` to reuse a loaded encoder.

### LLM reranker

```python
from persian_rag import LLMReranker

reranker = LLMReranker(llm, max_candidates=20, max_passage_chars=1200)
```

The model sees the query and numbered passages and returns an order such as `[3, 1, 2]`. Replies wrapped in prose or code fences are accepted, unknown or repeated numbers are ignored, and passages the model leaves out keep their original order at the end. Scores fall linearly with the final rank.

### Your own reranker

Any object with this method works:

```python
def rerank(
    self, query: str, results: Sequence[RetrievalResult], top_k: int | None = None
) -> list[RetrievalResult]: ...
```

Use `arerank(reranker, query, results)` to run a synchronous reranker from async code.
