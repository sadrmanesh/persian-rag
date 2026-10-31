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

local = SentenceTransformerEmbedder()  # intfloat/multilingual-e5-small
remote = OpenAICompatibleEmbedder(model="text-embedding-3-small")  # reads OPENAI_API_KEY
```

## Embedding models and prefixes

From 0.4.0 the default `SentenceTransformerEmbedder` model is [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) (118M parameters, 384 dimensions). On Persian retrieval sets it is far ahead of the previous default, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, at the same size; see [Benchmarks](benchmarks.md#embedding-models).

E5 models expect `query: ` before questions and `passage: ` before documents. The embedder adds them for you:

| `prefix` | Behaviour |
| --- | --- |
| `"auto"` (default) | Use the model's own `query`/`document` prompts when it defines non-empty ones. Otherwise add the E5 prefixes for E5 models (`intfloat/multilingual-e5-*`, `e5-*`) and for `persian-rag-embed` models, and nothing for every other model. |
| `"e5"` | Always add `query: ` and `passage: `. |
| `"none"` | Never add a prefix. |

`query_prefix` and `passage_prefix` override the result:

```python
SentenceTransformerEmbedder("intfloat/multilingual-e5-base")  # E5 prefixes
SentenceTransformerEmbedder("BAAI/bge-m3")  # no prefix
SentenceTransformerEmbedder("my-org/my-e5-copy", prefix="e5")  # force E5 prefixes
SentenceTransformerEmbedder("my-org/model", query_prefix="Query: ", passage_prefix="")
```

When the prefixes match the model's own prompts, the embedder calls `encode_query` and `encode_document`, so the model's prompt settings are used as they are. The resolved prefixes are stored in the index, so a saved index is always queried the way it was built.

## Upgrading from 0.3.x

* Code and indexes that name a model keep working. Indexes store the model they were built with, and `persian-rag query`, `chat`, `eval`, and `serve` load that model.
* An index built before 0.4.0 has no stored prefixes, so it is queried without prefixes, exactly as it was built. An E5 index from before 0.4.0 also gets a warning: rebuild it with `persian-rag ingest` to get the prefixes and better retrieval.
* `SentenceTransformerEmbedder()` with no model now loads e5-small. Vectors from the two models are not compatible, even though both have 384 dimensions. Rebuild any index you made with the old default, or keep the old model explicitly:

```bash
persian-rag ingest ./library --index .persian-rag --embedder sentence-transformers \
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

* Loading an index with an embedder for a different model now raises `IndexFormatError` instead of returning wrong results. Different prefixes give a warning.

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
