# شروع کار

## نصب

```bash
pip install persian-rag
```

برای هر قابلیت، افزونه مربوط به آن را اضافه کنید:

| افزونه | قابلیت |
| --- | --- |
| `sentence-transformers` | کلاس `SentenceTransformerEmbedder` |
| `rerank` | کلاس `CrossEncoderReranker` |
| `faiss` | کلاس `FaissVectorStore` |
| `qdrant` | کلاس `QdrantVectorStore` |
| `pgvector` | کلاس `PgVectorStore` |
| `pdf` | خواندن PDF با pypdf |
| `docx` | خواندن Word با python-docx |
| `server` | برنامه FastAPI و فرمان `persian-rag serve` |
| `all` | همه موارد بالا |
| `docs` | ابزار MkDocs برای ساختن همین مستندات |

```bash
pip install "persian-rag[pdf,docx,rerank,server]"
```

پایتون ۳٫۱۰ یا جدیدتر لازم است.

## نخستین سامانه

```python
from persian_rag import HashingEmbedder, RAGPipeline, build_retriever

texts = [
    "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
    "فوتبال ورزشی گروهی با دو تیم است.",
]
retriever = build_retriever(texts, HashingEmbedder())


def llm(prompt: str) -> str:
    return "شیراز"  # call your model here


answer = RAGPipeline(retriever, llm).query("حافظ در کدام شهر زاده شد؟", k=2)
print(answer.answer)
```

کلاس `HashingEmbedder` قطعی است و بدون اینترنت کار می کند، پس برای آزمون ها عالی است، اما معنا را درک نمی کند. برای داده واقعی یکی از این دو را به کار ببرید:

```python
from persian_rag import OpenAICompatibleEmbedder, SentenceTransformerEmbedder

local = SentenceTransformerEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
remote = OpenAICompatibleEmbedder(model="text-embedding-3-small")
```

نسخه دوم کلید را از متغیر محیطی `OPENAI_API_KEY` می خواند.

## از روی فایل ها

```python
from persian_rag import load_documents

documents = load_documents("./library")
retriever = build_retriever(
    [doc.text for doc in documents],
    embedder,
    document_ids=[doc.id for doc in documents],
    metadatas=[doc.metadata for doc in documents],
)
```

یا از خط فرمان:

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag
```

## فراخوانی مدل زبانی

تابع `make_openai_llm` برای هر سرویس گفت و گوی سازگار با OpenAI یک تابع قابل فراخوانی می سازد:

```python
from persian_rag import make_openai_llm

llm = make_openai_llm(model="gpt-4o-mini", api_key="...")
```

قالب پیش فرض پرامپت فارسی است (`DEFAULT_PROMPT_FA`). برای انگلیسی `prompt_template=DEFAULT_PROMPT_EN` را بدهید. قالب دلخواه باید `{context}` و `{question}` را داشته باشد.
