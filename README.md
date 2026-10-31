# persian-rag

[![CI](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/sadrmanesh/persian-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-sadrmanesh.github.io-009688)](https://sadrmanesh.github.io/persian-rag/)
[![Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo)

Persian-first retrieval-augmented generation for Python.

persian-rag is a toolkit for building RAG systems over Farsi text. It normalizes Arabic and Persian characters, half-spaces, digits, and punctuation; loads text, Markdown, HTML, PDF, and Word files and web pages (extracting the main content and repairing reversed PDF text); chunks and embeds passages; keeps the index up to date incrementally; retrieves with BM25, dense vectors, or both, with Persian spelling-variant query expansion; reranks; answers with source citations and conversation memory; and measures the result. Generation is any function you supply, or the built-in OpenAI-compatible and Ollama adapters.

The core install is pure Python with no dependencies. Every heavy piece (models, vector databases, parsers, the HTTP server) is an optional extra.

**Documentation:** <https://sadrmanesh.github.io/persian-rag/> in English and Persian. The sources live in [`docs/`](docs/) and build with MkDocs (see [Documentation](#documentation)).

**Live demo:** [persian-rag on Hugging Face Spaces](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo) runs normalization and retrieval in the browser. **Benchmark dataset:** [sadrmanesh/persian-rag-bench](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench) on the Hugging Face Hub.

**Read more:** [Why RAG breaks on Persian text, and what I built to fix it](https://dev.to/sadrmanesh/why-rag-breaks-on-persian-text-and-what-i-built-to-fix-it-normalization-half-spaces-spelling-2jgm) on Dev.to · [The design decisions that fixed it](https://www.linkedin.com/pulse/why-rag-quietly-fails-persian-text-design-decisions-fixed-sadrmanesh-wd4vf/) on LinkedIn

## خلاصه فارسی

persian-rag یک جعبه ابزار پایتون برای ساخت سامانه های بازیابی و تولید پاسخ روی متن فارسی است. نویسه های عربی و فارسی، فاصله ها، رقم ها و نشانه گذاری را یکدست می کند. فایل های متنی، مارک داون، HTML، PDF و Word و صفحه های وب را می خواند، متن اصلی صفحه های وب را جدا می کند و ترتیب وارونه متن فارسی در PDF را اصلاح می کند. متن را تکه تکه می کند و با BM25، بردارهای معنایی یا ترکیب هر دو جست و جو می کند و می تواند پرسش را با املاهای دیگر فارسی (مانند اتاق و اطاق یا تهران و طهران) گسترش دهد. بردارها را در حافظه، FAISS، Qdrant، pgvector، Elasticsearch یا OpenSearch نگه می دارد و فقط سندهای تغییر کرده را دوباره ایندکس می کند. نتیجه را با رتبه بندی دوباره بهتر می کند، پاسخ را همراه ارجاع دقیق به منبع می دهد، گفت و گوی چند نوبتی را پشتیبانی می کند و کیفیت بازیابی و پاسخ را می سنجد. از نسخه 0.4.0 مدل بردارساز پیش فرض `intfloat/multilingual-e5-small` با پیشوندهای E5 است و مدل دقیق تر `intfloat/multilingual-e5-base` هم با `--model e5-base` در دسترس است. تولید پاسخ با تابع دلخواه شما یا اتصال های آماده OpenAI و Ollama انجام می شود. سرور HTTP به صورت ایمیج داکر `ghcr.io/sadrmanesh/persian-rag` هم منتشر می شود. راهنمای کامل فارسی در [سایت مستندات](https://sadrmanesh.github.io/persian-rag/fa/) و پوشه docs است. [نسخه نمایشی زنده](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo) در Hugging Face Spaces و [مجموعه داده محک](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench) در Hugging Face Hub هم در دسترس هستند.

## Features

- **Normalization** for Persian: Arabic Yeh and Kaf, presentation forms, ZWNJ, digits, optional diacritics, and punctuation.
- **Loaders** for `.txt`, `.md`, `.html`, `.pdf`, `.docx`, and web URLs. HTML goes through readability-style main-content extraction, Markdown front matter becomes metadata and files can be split by heading, and PDF text in visual (reversed) order is detected and repaired.
- **Chunking**: fixed windows with overlap, sentence packing, and recursive separators.
- **Embeddings** through a small protocol, with adapters for `sentence-transformers` (default `intfloat/multilingual-e5-small`, or the more accurate `intfloat/multilingual-e5-base` with `--model e5-base`, both with automatic E5 query and passage prefixes) and any OpenAI-compatible API, plus a persistent embedding cache. Indexes remember their model, and loading one with a different model is refused.
- **Vector stores**: in-memory, FAISS, Qdrant, PostgreSQL with pgvector, and Elasticsearch or OpenSearch (which also serve BM25 for hybrid search, without a client package).
- **Incremental indexing**: content hashes, upserts, and deletions, so only changed documents are re-embedded.
- **Hybrid retrieval** with BM25 and [reciprocal rank fusion](https://plg.uwaterloo.ca/gvcormac/cormacksigir09-rrf.pdf), and an optional query result cache.
- **Query rewriting** for Persian spelling variants (اتاق/اطاق, هیئت/هیات, مسئله/مساله), hamza and affix spacing forms, and optional LLM rewrites, fused with RRF.
- **Reranking** with a cross-encoder (default `BAAI/bge-reranker-v2-m3`) or any LLM (listwise).
- **Citations**: answers with `[n]` markers resolved to quoted source spans and character offsets, and a list of uncited claims.
- **Chat**: conversation memory and follow-up question condensing, in Python, the CLI, and the HTTP API.
- **LLM adapters** for OpenAI-compatible servers and Ollama, with retries, using only the standard library.
- **Evaluation**: hit rate, MRR, recall, and nDCG at several cut-offs, plus pluggable answer faithfulness scorers.
- **Async**: `aretrieve` and `aquery`, with native support for `async` model functions.
- **HTTP API** built on FastAPI, and a CLI for `ingest`, `query`, `chat`, `eval`, and `serve`.
- **Benchmarks**: an offline Persian retrieval benchmark with published numbers.
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

### Docker

Every release publishes the HTTP API as a multi-architecture image (`linux/amd64`, `linux/arm64`) on the GitHub Container Registry, with the `server`, `pdf`, and `docx` extras installed:

```bash
docker run --rm -v "$PWD/library:/library:ro" -v persian-rag-data:/data \
  ghcr.io/sadrmanesh/persian-rag:0.4.0 ingest /library --index /data/index
docker run -d -p 8000:8000 -v persian-rag-data:/data ghcr.io/sadrmanesh/persian-rag:0.4.0
```

Tags are `X.Y.Z`, `X.Y`, and `latest`. The container runs as a non-root user and has a health check on `/health`. See [Docker](https://sadrmanesh.github.io/persian-rag/docker/) for language models, custom extras, and building the image yourself.

Requires Python 3.10 or newer. The OpenAI-compatible and Ollama adapters, the web loaders, and the Elasticsearch and OpenSearch stores use only the standard library: set `OPENAI_API_KEY` (and `OPENAI_BASE_URL` for compatible servers) or `OLLAMA_HOST`.

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

`HashingEmbedder` is a deterministic character n-gram hash for tests and offline checks. For semantic search use `SentenceTransformerEmbedder` (default model `intfloat/multilingual-e5-small`, with the E5 `query: `/`passage: ` prefixes added for you; `SentenceTransformerEmbedder("e5-base")` loads the larger, more accurate `intfloat/multilingual-e5-base`) or `OpenAICompatibleEmbedder`. The same flow is in [`examples/quickstart.py`](examples/quickstart.py).

> **Upgrading to 0.4.0:** the default `sentence-transformers` model changed from `paraphrase-multilingual-MiniLM-L12-v2` to `intfloat/multilingual-e5-small`. Existing indexes keep working with the model they were built with. Rebuild indexes made with the old default to use the new one, or pass `--model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to keep it. See [Upgrading from 0.3.x](https://sadrmanesh.github.io/persian-rag/getting-started/#upgrading-from-03x).

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

### Web pages and Markdown

```python
from persian_rag import load_markdown, load_url

page = load_url("https://example.com/fa/article")  # main content only, id is the URL
sections = load_markdown("guide.md", split_sections=True)  # ids like "guide.md#نصب"
```

`load_documents` also reads `.html` files, and `persian-rag ingest` accepts URLs. Navigation, footers, comments, share bars, and hidden elements are removed; `<article>` and `<main>` are preferred when present.

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

### Query rewriting

```python
from persian_rag import QueryRewriter

pipeline = RAGPipeline(retriever, my_llm, query_rewriter=QueryRewriter())
QueryRewriter().variants("اطاق پنج دری")  # ['اطاق پنج دری', 'اتاق پنج دری']
```

Each variant is searched and the rankings are fused with RRF, with the original query weighted double. Pass `llm=` to add model-written rewrites. On the bundled benchmark this lifts BM25 Hit@1 from 0.881 to 0.952.

### Citations and chat

```python
from persian_rag import ChatSession, OllamaLLM, RAGPipeline

pipeline = RAGPipeline(retriever, OllamaLLM(model="qwen2.5:7b"), citations=True)
answer = pipeline.query("حافظ اهل کجا بود؟")
for c in answer.citations:
    print(c.marker, c.document_id, c.quote, c.start, c.end)

chat = ChatSession(pipeline)
chat.ask("حافظ اهل کجا بود؟")
chat.ask("آرامگاهش کجاست؟")  # rewritten to a standalone question before retrieval
```

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

All stores share one protocol (`add`, `search`, `__len__`); all but FAISS also support `delete`. Scores are always larger-is-closer.

```python
from persian_rag import HybridRetriever, IncrementalIndexer, load_documents
from persian_rag.vectorstores import OpenSearchStore

store = OpenSearchStore("http://localhost:9200", "persian_docs")
retriever = HybridRetriever(embedder, store=store, sparse_backend=store)  # BM25 runs in OpenSearch
report = IncrementalIndexer(retriever).sync(load_documents("./library"))
print(report.summary())  # added 3, updated 1, deleted 0, unchanged 40 document(s); ...
```

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

Endpoints: `GET /health`, `POST /retrieve` (`{"query": "...", "k": 5, "mode": "hybrid"}`), `POST /query` (`{"question": "...", "k": 5}`), `POST /chat` (`{"message": "...", "session_id": null}`), and `DELETE /chat/{session_id}`. Answers include `citations` when the pipeline has `citations=True`. Interactive docs are served at `/docs`. The same API ships as the `ghcr.io/sadrmanesh/persian-rag` Docker image (see [Docker](#docker)).

### Command line

```bash
persian-rag ingest ./library https://example.com/post --index .persian-rag --embedding-cache .cache/emb.sqlite
persian-rag ingest ./library --index .persian-rag --update
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --json
persian-rag query "اطاق پنج دری" --expand rules --llm ollama --citations
persian-rag chat --index .persian-rag --llm openai
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10
persian-rag serve --index .persian-rag --port 8000
```

`ingest` reads `.txt`, `.md`, `.html`, `.pdf`, `.docx`, and URLs, skips hidden files, and writes a UTF-8 JSON index without API keys. Useful flags: `--embedder hashing|sentence-transformers|openai`, `--model`, `--prefix auto|e5|none`, `--strategy`, `--chunk-size`, `--rtl-fix`, `--split-pages`, `--split-sections`, `--update`, `--mode`, `--expand rules|llm`, `--rerank cross-encoder|llm`, `--llm openai|ollama`, `--citations`, and `--prompt-lang fa|en`. Run `persian-rag <command> --help` for the rest.

## Benchmarks

`python benchmarks/run_benchmark.py` scores sparse, dense, and hybrid retrieval, with and without query rewriting, on a bundled set of 34 Persian documents and 42 questions. It runs offline. Results for 0.3.0 with the hashing embedder:

| Configuration | Hit@1 | MRR@10 | Hit@1 on spelling variants |
|---|---:|---:|---:|
| sparse | 0.881 | 0.881 | 0.583 |
| sparse + rewrite | 0.952 | 0.964 | 0.833 |
| hybrid | 0.952 | 0.964 | 0.833 |
| hybrid + rewrite | 0.952 | 0.964 | 0.833 |

With `--embedder sentence-transformers`, the 0.4.0 default model `intfloat/multilingual-e5-small` (with E5 prefixes) is far ahead of the previous default `paraphrase-multilingual-MiniLM-L12-v2` at the same size (nDCG@10, measured on CPU):

| Model | MIRACL fa dev* | PQuAD validation | persian-rag-bench |
|---|---:|---:|---:|
| paraphrase-multilingual-MiniLM-L12-v2 (old default) | 0.4424 | 0.6437 | 0.8637 |
| multilingual-e5-small with prefixes (new default) | 0.7461 | 0.9242 | 0.9881 |
| multilingual-e5-base with prefixes (`--model e5-base`) | **0.7657** | **0.9247** | **1.0000** |

`intfloat/multilingual-e5-base` (278M parameters, 768 dimensions) is the more accurate option: +0.02 nDCG@10 on MIRACL fa dev, and much more robust to queries typed with Arabic yeh and kaf and without half spaces when the text is not normalized (MIRACL 0.7089 against 0.6645, PQuAD 0.8906 against 0.8639). It is about 2.4 times the size, its vectors take twice the space, and it encodes about 2.4 times slower on CPU (22 against 54 PQuAD paragraphs per second on 8 cores; a single query takes 13 ms against 6 ms). Pass `--model e5-base` (or the full name) to use it; e5-small stays the default.

\* 632 queries against a pooled corpus of 6,405 judged passages, not the full MIRACL corpus, so not comparable with published MIRACL scores.

Full table, latency, and method: [Benchmarks](https://sadrmanesh.github.io/persian-rag/benchmarks/). The same corpus and questions are on the Hugging Face Hub as [sadrmanesh/persian-rag-bench](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench).

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

Tests run offline. Optional-dependency tests skip when the extra is missing. The pgvector integration test runs when `PERSIAN_RAG_PG_DSN` points at PostgreSQL with the `vector` extension, the search engine tests run with `PERSIAN_RAG_ES_URL` or `PERSIAN_RAG_OPENSEARCH_URL`, and heavy model tests run with `PERSIAN_RAG_RUN_HEAVY=1`.

Releases are published to PyPI by the `Release` workflow when a `v*` tag is pushed, using trusted publishing (repository variable `PYPI_TRUSTED_PUBLISHING=true`) or a `PYPI_API_TOKEN` secret. Without either, the build still runs and publishing is skipped. The same workflow pushes the Docker image to `ghcr.io/sadrmanesh/persian-rag` with the workflow's `GITHUB_TOKEN`.

## Roadmap

- Morphological normalization for Persian clitics beyond the affix list.
- Search engine stores from the command line, and more durable store connectors.
- Streaming answers in the HTTP API.
- Evaluation recipes for public Persian QA datasets and semantic embedding baselines in the benchmark.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Issues and proposals are welcome in Persian or English.

## License

[MIT](LICENSE). Copyright (c) 2026 Naser Sadrmanesh.

## Author

Naser Sadrmanesh

- GitHub: [https://github.com/sadrmanesh](https://github.com/sadrmanesh)
- Site: [https://sadrmanesh.github.io](https://sadrmanesh.github.io)
- Email: [info.sadrmanesh@gmail.com](mailto:info.sadrmanesh@gmail.com)
