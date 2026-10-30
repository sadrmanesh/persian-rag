# persian-rag

[![CI](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Persian-first retrieval-augmented generation for Python.

persian-rag is a toolkit for building RAG systems over Farsi text. It normalizes Arabic and Persian characters, half-spaces, digits, and punctuation; loads text, Markdown, PDF, and Word files (repairing reversed PDF text); chunks and embeds passages; retrieves them with BM25, dense vectors, or both; reranks them; and measures the result. Generation is a function you supply, sync or async.

The core install is pure Python with no dependencies. Every heavy piece (models, vector databases, parsers, the HTTP server) is an optional extra.

**Documentation:** English and Persian guides live in [`docs/`](docs/) and build with MkDocs (see [Documentation](#documentation)).

## خلاصه فارسی

persian-rag یک جعبه ابزار پایتون برای ساخت سامانه های بازیابی و تولید پاسخ روی متن فارسی است. نویسه های عربی و فارسی، فاصله ها، رقم ها و نشانه گذاری را یکدست می کند. فایل های متنی، مارک داون، PDF و Word را می خواند و ترتیب وارونه متن فارسی در PDF را اصلاح می کند. متن را تکه تکه می کند و با BM25، بردارهای معنایی یا ترکیب هر دو جست و جو می کند. نتیجه را با رتبه بندی دوباره بهتر می کند و کیفیت بازیابی و پاسخ را می سنجد. تولید پاسخ به تابعی سپرده می شود که خودتان می دهید و می تواند همگام یا ناهمگام باشد. راهنمای کامل فارسی در پوشه docs است.

## Features

- **Normalization** for Persian: Arabic Yeh and Kaf, presentation forms, ZWNJ, digits, optional diacritics, and punctuation.
- **Loaders** for `.txt`, `.md`, `.pdf`, and `.docx`. PDF text in visual (reversed) order is detected and repaired, presentation forms are folded, and bidi control marks are removed.
- **Chunking**: fixed windows with overlap, sentence packing, and recursive separators.
- **Embeddings** through a small protocol, with adapters for `sentence-transformers` and any OpenAI-compatible API, plus a persistent embedding cache.
- **Vector stores**: in-memory, FAISS, Qdrant, and PostgreSQL with pgvector.
- **Hybrid retrieval** with BM25 and [reciprocal rank fusion](https://plg.uwaterloo.ca/gvcormac/cormacksigir09-rrf.pdf), and an optional query result cache.
- **Reranking** with a cross-encoder (default `BAAI/bge-reranker-v2-m3`) or any LLM (listwise).
- **Evaluation**: hit rate, MRR, recall, and nDCG at several cut-offs, plus pluggable answer faithfulness scorers.
- **Async**: `aretrieve` and `aquery`, with native support for `async` model functions.
- **HTTP API** built on FastAPI, and a CLI for `ingest`, `query`, `eval`, and `serve`.
- Fully typed (`mypy --strict`), tested offline, MIT licensed.

## Installation

```bash
pip install persian-rag
```

Pick the extras you need:

| Extra | Installs | Enables |
| --- | --- | --- |
| `sentence-transformers` | sentence-transformers | `SentenceTransformerEmbedder` |
| `rerank` | sentence-transformers | `CrossEncoderReranker` |
| `faiss` | faiss-cpu, numpy | `FaissVectorStore` |
| `qdrant` | qdrant-client | `QdrantVectorStore` |
| `pgvector` | psycopg 3 | `PgVectorStore` |
| `pdf` | pypdf | PDF loading |
| `docx` | python-docx | Word loading |
| `server` | fastapi, uvicorn | `persian_rag.server`, `persian-rag serve` |
| `all` | everything above | |

```bash
pip install "persian-rag[pdf,docx,qdrant,server]"
```

Requires Python 3.10 or newer. The OpenAI-compatible adapters use only the standard library: set `OPENAI_API_KEY`, and `OPENAI_BASE_URL` for compatible servers.

## Quick start

```python
from persian_rag import HashingEmbedder, RAGPipeline, build_retriever

texts = [
    "حافظ در شهر شیراز زاده شد و دیوان او مشهور است.",
    "فوتبال ورزشی گروهی با دو تیم است.",
]
retriever = build_retriever(texts, HashingEmbedder())


def answer(prompt: str) -> str:
    # Replace with a call to your model.
    return "شیراز" if "شیراز" in prompt else "نامشخص"


result = RAGPipeline(retriever, answer).query("حافظ در کدام شهر زاده شد؟")
print(result.answer)
for item in result.contexts:
    print(round(item.score, 4), item.text)
```

`HashingEmbedder` is a deterministic character n-gram hash for tests and offline checks. For semantic search use `SentenceTransformerEmbedder` (for example `"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"`) or `OpenAICompatibleEmbedder`. The same flow is in [`examples/quickstart.py`](examples/quickstart.py).

## Usage

### Normalization

```python
from persian_rag import NormalizeConfig, normalize

normalize("يك كتاب با اعداد ٤٥٦")
# 'یک کتاب با اعداد ۴۵۶'

normalize("کِتاب", NormalizeConfig(remove_diacritics=True, digits="ascii"))
```

`normalize` is idempotent for a fixed config. Documents and queries must use the same `NormalizeConfig`; `build_retriever` and `HybridRetriever` take care of that.

| Option | Default | Effect |
| --- | --- | --- |
| `unify_arabic_chars` | `True` | `ي`/`ى` to `ی`, `ك` to `ک`, `أ`/`إ`/`ٱ` to `ا`, `ؤ` to `و`, `ة` to `ه`. `آ` and `ئ` stay. |
| `compatibility_forms` | `True` | NFKC, so presentation forms fold into base letters. |
| `zwnj` | `"normalize"` | Collapse repeated half-spaces; `"remove"` deletes them; `"space"` turns them into spaces. |
| `correct_half_spaces` | `True` | Join prefixes such as `می` and suffixes such as `ها` and `ترین` with a half-space. |
| `digits` | `"persian"` | `"persian"`, `"ascii"`, or `"keep"`. Arabic-Indic digits are included. |
| `remove_diacritics` | `False` | Strip Arabic vowel marks. |
| `remove_tatweel` | `True` | Remove kashida (`ـ`). |
| `clean_punctuation` | `True` | Collapse repeats and tidy spaces around punctuation. |
| `punctuation` | `"persian"` | Map `,;?%` to `،؛؟٪`, the reverse with `"ascii"`, or `"keep"`. |

### Loading PDF and Word files

```python
from persian_rag import load_documents, load_pdf

documents = load_documents("./library")  # .txt, .md, .pdf, .docx
pages = load_pdf("report.pdf", split_pages=True)  # one document per page
```

Many PDF generators store Persian lines in visual order, so extracted words come out reversed (`باتک` instead of `کتاب`). With `rtl_fix="auto"` (the default) each page is checked using two signals: which contextual glyph forms start and end each word, and whether frequent Persian words appear forwards or backwards. Reversed lines are flipped while numbers and Latin words keep their direction. Use `rtl_fix="reverse"` to force the repair or `"none"` to disable it. `clean_extracted_text` exposes the same cleanup for text from other extractors.

Word documents keep paragraphs and tables in body order; table rows become `cell | cell` lines.

### Retrieval and reranking

```python
from persian_rag import CrossEncoderReranker, HybridRetriever, LLMReranker, RAGPipeline

hits = retriever.retrieve("حافظ اهل کجا بود؟", k=5, mode="hybrid")  # or "dense" / "sparse"

pipeline = RAGPipeline(
    retriever,
    my_llm,
    reranker=CrossEncoderReranker(),  # needs the rerank extra
    rerank_candidates=20,
)
answer = pipeline.query("حافظ اهل کجا بود؟", k=4)
```

The retriever fetches `rerank_candidates` chunks and the reranker keeps the best `k`. Reranked results carry the reranker score in `score` and the first-stage score in `retrieval_score`. `LLMReranker(llm)` asks a model to order numbered passages and tolerates messy replies; any object with a `rerank(query, results, top_k)` method also works.

Hybrid scores are reciprocal rank fusion. Dense mode returns the store score; sparse mode returns Okapi BM25.

### Vector stores

```python
from persian_rag.vectorstores import PgVectorStore, QdrantVectorStore

qdrant = QdrantVectorStore("persian_docs", dimension=384, url="http://localhost:6333")
local = QdrantVectorStore("persian_docs", dimension=384)  # in-process, in memory

pg = PgVectorStore(
    "postgresql://user:pass@localhost/db",
    dimension=384,
    table="persian_chunks",
    create_index=True,  # HNSW
)

retriever = build_retriever(texts, embedder, store=qdrant)
```

All stores share one protocol (`add`, `search`, `__len__`); Qdrant and pgvector also support `delete`. Scores are always larger-is-closer.

### Caching and async

```python
from persian_rag import CachedEmbedder, LRUCache, SQLiteEmbeddingCache

embedder = CachedEmbedder(OpenAICompatibleEmbedder(), SQLiteEmbeddingCache("emb.sqlite"))
retriever = build_retriever(texts, embedder)
retriever.result_cache = LRUCache(max_size=512, ttl=300)

answer = await pipeline.aquery("پرسش")  # async LLMs are awaited, sync ones run in a thread
```

Embedding cache keys include the embedder settings, so changing the model never returns stale vectors. The result cache is cleared whenever chunks are added.

### Evaluation

```python
from persian_rag import evaluate_answers, evaluate_retrieval, load_eval_examples

examples = load_eval_examples("eval.jsonl")
report = evaluate_retrieval(retriever, examples, k_values=(1, 3, 5, 10))
print(report.format_table())

answers = evaluate_answers(pipeline, examples)  # token-overlap faithfulness by default
```

Each JSONL line holds `question`, `relevant_ids` (document ids, or chunk ids with `match="chunk"`), and optionally `reference_answer`. Faithfulness scorers implement `score(question, answer, contexts)`: `TokenOverlapFaithfulness` is offline and deterministic, `LLMFaithfulnessJudge(llm)` asks a model for a 0 to 1 score.

### HTTP API

```python
import uvicorn
from persian_rag.server import create_app

uvicorn.run(create_app(pipeline), host="127.0.0.1", port=8000)
```

Endpoints: `GET /health`, `POST /retrieve` (`{"query": "...", "k": 5, "mode": "hybrid"}`), and `POST /query` (`{"question": "...", "k": 5}`). Interactive docs are served at `/docs`.

### Command line

```bash
persian-rag ingest ./library --index .persian-rag --embedding-cache .cache/emb.sqlite
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --json
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10
persian-rag serve --index .persian-rag --port 8000
```

`ingest` reads `.txt`, `.md`, `.pdf`, and `.docx`, skips hidden files, and writes a UTF-8 JSON index without API keys. Useful flags: `--embedder hashing|sentence-transformers|openai`, `--strategy`, `--chunk-size`, `--rtl-fix`, `--split-pages`, `--mode`, `--rerank cross-encoder|llm`, `--llm openai`, and `--prompt-lang fa|en`. Run `persian-rag <command> --help` for the rest.

## Documentation

```bash
pip install -e ".[docs]"
mkdocs serve
```

The site has matching English and Persian pages for every topic.

## Development

```bash
pip install -e ".[dev,server,qdrant,pgvector,pdf,docx]"
ruff check .
ruff format --check .
mypy src
pytest
```

Tests run offline. Optional-dependency tests skip when the extra is missing. The pgvector integration test runs when `PERSIAN_RAG_PG_DSN` points at PostgreSQL with the `vector` extension, and heavy model tests run with `PERSIAN_RAG_RUN_HEAVY=1`.

Releases are published to PyPI by the `Release` workflow when a `v*` tag is pushed, using trusted publishing (repository variable `PYPI_TRUSTED_PUBLISHING=true`) or a `PYPI_API_TOKEN` secret. Without either, the build still runs and publishing is skipped.

## Roadmap

- Morphological normalization for Persian clitics beyond the affix list.
- OpenSearch and other durable store connectors.
- HTML loading and incremental re-indexing of changed files.
- Citation-style answers with span-level grounding.
- Evaluation recipes for public Persian QA datasets.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and proposals are welcome in Persian or English.

## License

[MIT](LICENSE). Copyright (c) 2026 Naser Sadrmanesh.

## Author

Naser Sadrmanesh

- GitHub: [https://github.com/sadrmanesh](https://github.com/sadrmanesh)
- Site: [https://sadrmanesh.github.io](https://sadrmanesh.github.io)
- Email: [info.sadrmanesh@gmail.com](mailto:info.sadrmanesh@gmail.com)
