# بازیابی و رتبه بندی دوباره

## بازیابی ترکیبی

```python
from persian_rag import HybridRetriever

retriever = HybridRetriever(embedder, rrf_k=60, dense_weight=1.0, sparse_weight=1.0)
retriever.add_chunks(chunks)
hits = retriever.retrieve("حافظ اهل کجا بود؟", k=5, mode="hybrid")
```

| حالت | روش رتبه بندی | امتیاز |
| --- | --- | --- |
| `hybrid` | ترکیب فهرست BM25 و فهرست برداری با RRF | امتیاز وزن دار RRF |
| `dense` | جست و جو در پایگاه برداری | امتیاز پایگاه، به طور پیش فرض کسینوسی |
| `sparse` | الگوریتم BM25 روی واژه های فارسی | امتیاز BM25 |

هر نتیجه شامل شناسه تکه، شناسه سند، متن، امتیاز، جایگاه متن، فراداده و رتبه آن در هر دو فهرست است.

## رتبه بندی دوباره

رتبه بندی دوباره مجموعه بزرگ تری از نامزدها را با مدلی دقیق تر مرتب می کند:

```python
from persian_rag import CrossEncoderReranker, RAGPipeline

pipeline = RAGPipeline(
    retriever,
    llm,
    reranker=CrossEncoderReranker(),
    rerank_candidates=20,
)
answer = pipeline.query("پرسش", k=4)
```

ابتدا به تعداد `rerank_candidates` تکه بازیابی می شود و سپس بهترین `k` تکه نگه داشته می شود. امتیاز جدید در `score` و امتیاز مرحله نخست در `retrieval_score` قرار می گیرد.

### مدل cross-encoder

```bash
pip install "persian-rag[rerank]"
```

این کلاس هر جفت پرسش و متن را امتیاز می دهد. مدل پیش فرض `BAAI/bge-reranker-v2-m3` چندزبانه است و فارسی را پشتیبانی می کند. با پارامتر `model=` می توانید مدلی را که قبلا بارگذاری کرده اید دوباره به کار ببرید.

### رتبه بندی با مدل زبانی

```python
from persian_rag import LLMReranker

reranker = LLMReranker(llm, max_candidates=20, max_passage_chars=1200)
```

مدل پرسش و متن های شماره دار را می بیند و ترتیبی مانند `[3, 1, 2]` برمی گرداند. پاسخ های همراه با توضیح اضافه هم پذیرفته می شوند، شماره های نامعتبر یا تکراری نادیده گرفته می شوند و متن هایی که مدل جا انداخته با همان ترتیب اولیه به انتها می روند.

### رتبه بند دلخواه

هر شیئی که متد `rerank(query, results, top_k)` داشته باشد قابل استفاده است. برای اجرای آن در کد ناهمگام از `arerank` استفاده کنید.
