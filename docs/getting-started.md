# Getting started

## Install

```bash
pip install persian-rag
```

Add extras for the features you need:

| Extra | Enables |
| --- | --- |
| `sentence-transformers` | `SentenceTransformerEmbedder` |
| `rerank` | `CrossEncoderReranker` |
| `faiss` | `FaissVectorStore` |
| `qdrant` | `QdrantVectorStore` |
| `pgvector` | `PgVectorStore` |
| `pdf` | PDF loading (pypdf) |
| `docx` | Word loading (python-docx) |
| `server` | FastAPI app and `persian-rag serve` |
| `all` | All of the above |
| `docs` | MkDocs to build this site |

```bash
pip install "persian-rag[pdf,docx,rerank,server]"
```

Python 3.10 or newer is required.

## First pipeline

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
for context in answer.contexts:
    print(context.document_id, round(context.score, 4))
```

`HashingEmbedder` is deterministic and offline, which makes it ideal for tests, but it is not semantic. For real corpora use one of:

```python
from persian_rag import OpenAICompatibleEmbedder, SentenceTransformerEmbedder

local = SentenceTransformerEmbedder("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
remote = OpenAICompatibleEmbedder(model="text-embedding-3-small")  # reads OPENAI_API_KEY
```

## From files

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

Or from the shell:

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag
```

## Calling a model

`make_openai_llm` returns a callable for any OpenAI-compatible chat endpoint:

```python
from persian_rag import make_openai_llm

llm = make_openai_llm(model="gpt-4o-mini", api_key="...")
```

The default prompt is Persian (`DEFAULT_PROMPT_FA`); pass `prompt_template=DEFAULT_PROMPT_EN` for English. Custom templates must contain `{context}` and `{question}`.
