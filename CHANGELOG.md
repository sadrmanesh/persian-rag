# Changelog

All notable changes to persian-rag are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-10-30

### Added

- Web loaders: `load_url`, `load_urls`, and `load_html` with readability-style main-content extraction (`extract_main_content`, `html_to_text`). Navigation, footers, comments, share bars, and hidden elements are removed, `<article>` and `<main>` are preferred, charsets are detected from headers and `<meta>`, and page title, language, description, author, and publish date go into metadata. Standard library only.
- Markdown loading with front matter metadata, a title from the first heading, and optional splitting into one document per heading (`load_markdown(..., split_sections=True)`, `split_markdown_sections`). `load_documents` reads `.html` and `.htm` files and accepts `split_sections` and `html_extract`.
- `ElasticsearchStore` and `OpenSearchStore`: vectors, text, and metadata in one index, k-NN for dense search and BM25 with the `persian` analyzer for sparse search, with add, upsert, delete, and API key or basic authentication. They talk to the REST API directly and need no client package. `store_from_url` builds one from an `elasticsearch://` or `opensearch+https://` URL.
- `HybridRetriever(sparse_backend=...)` to run the BM25 half in a search engine, plus `delete_document`, `delete_chunks`, `document_ids`, and `chunk_ids_for_document`. `InMemoryVectorStore` and the in-process BM25 index support deletes.
- Incremental indexing: `content_hash`, `IndexManifest`, and `IncrementalIndexer` with `sync`, `upsert`, and `delete`. Unchanged documents are skipped and vectors of unchanged chunks are reused. Saved indexes store the manifest.
- Persian query rewriting: `QueryRewriter` with spelling groups (اتاق/اطاق, تهران/طهران, هیئت/هیات, مسئله/مساله, and more), hamza and alef madda forms, affix spacing variants, optional Arabic letter variants, and optional LLM rewrites. `RewritingRetriever` fuses the variant rankings with RRF and records `matched_queries`.
- Citation-grounded answers: `RAGPipeline(citations=True)` numbers the contexts, and `extract_citations` resolves `[n]` markers (Persian and Arabic-Indic digits included) to the best supporting sentence with document, chunk, and answer offsets. `RAGAnswer` gained `citations`, `uncited`, and `to_dict()`.
- Conversation memory: `ConversationMemory` and `ChatSession` with LLM or heuristic condensing of follow-up questions, sync and async.
- LLM adapters: `OpenAICompatibleLLM`, `OllamaLLM`, and `create_llm`, with retries for timeouts, 408, 409, 429, and 5xx responses, and a pluggable transport. Standard library only.
- HTTP API: `POST /chat` and `DELETE /chat/{session_id}`, and `citations` and `uncited` in `/query` responses.
- CLI: `persian-rag chat`; `ingest` accepts several paths and URLs and gains `--split-sections`, `--html-extract`, `--update`, and `--keep-missing`; pipeline commands gain `--llm ollama`, `--ollama-url`, `--llm-max-tokens`, `--citations`, and `--expand none|rules|llm`.
- Benchmark: `benchmarks/run_benchmark.py` with a 34-document Persian corpus and 42 labelled questions, comparing sparse, dense, and hybrid retrieval with and without query rewriting. Results are published in the documentation.
- Documentation pages for web loaders, Elasticsearch and OpenSearch, incremental indexing, query rewriting, citations and chat, LLM adapters, and benchmarks, in English and Persian.
- CI job running integration tests against Elasticsearch 8 and OpenSearch 2.

### Changed

- `persian-rag ingest` stores a document manifest in `index.json` and reports document and chunk counts. Indexes from 0.2.0 still load; the first `--update` treats every document as updated, reuses vectors of unchanged chunks, and writes the manifest.
- `--llm openai` now uses `OpenAICompatibleLLM`, so it retries transient errors.

## [0.2.0] - 2026-10-30

### Added

- Rerankers: a `Reranker` protocol, `CrossEncoderReranker` (`rerank` extra, default model `BAAI/bge-reranker-v2-m3`), and a listwise `LLMReranker` that tolerates loosely formatted replies.
- `RAGPipeline(reranker=..., rerank_candidates=...)` and `RAGPipeline.retrieve()`; reranked results keep the first-stage score in `RetrievalResult.retrieval_score`.
- Vector stores in the new `persian_rag.vectorstores` package: `QdrantVectorStore` (`qdrant` extra) and `PgVectorStore` for PostgreSQL with pgvector (`pgvector` extra), both with delete support.
- PDF (`pdf` extra) and Word (`docx` extra) loaders. PDF text is cleaned of presentation forms and bidi controls, and lines stored in visual order are detected and repaired. `load_documents` now reads `.pdf` and `.docx` files and can split PDFs by page.
- Evaluation module: hit rate, MRR, recall, and nDCG at several cut-offs, document or chunk matching, JSONL datasets, and a faithfulness scorer protocol with token-overlap and LLM-judge implementations.
- FastAPI server (`server` extra) with `/health`, `/retrieve`, and `/query`, created through `persian_rag.server.create_app`.
- Async support: `HybridRetriever.aretrieve`, `RAGPipeline.aretrieve`, and `RAGPipeline.aquery`, which accepts async or sync model callables.
- Caching: `CachedEmbedder` with in-memory and SQLite embedding caches, and an LRU/TTL result cache for `HybridRetriever`.
- CLI: `persian-rag eval` and `persian-rag serve`, plus `--rerank`, `--rtl-fix`, `--split-pages`, and `--embedding-cache` options.
- Bilingual documentation site (English and Persian) built with MkDocs Material.
- Release workflow that publishes to PyPI on version tags through trusted publishing or an API token, and skips publishing when neither is configured.
- CI jobs for Python 3.10 to 3.13 with all extras, a minimal-dependency run, PostgreSQL with pgvector, the documentation build, and package checks.

### Changed

- `MissingDependencyError` is raised for missing optional packages. It subclasses both `PersianRagError` and `ImportError`.
- `RAGPipeline.query` raises `TypeError` when given an async model; use `aquery` instead.
- The `all` extra now installs every optional integration.
- README rewritten with an extras table, usage guides, and a Persian summary.

## [0.1.0] - 2026-10-30

### Added

- Python package `persian_rag` for Python 3.10 and newer.
- Persian normalization: Arabic and Persian letters, ZWNJ, digits, optional diacritics, and punctuation.
- Sentence splitting, tokenization, and fixed, sentence, and recursive chunking.
- Embedder protocol with hashing, sentence-transformers, and OpenAI-compatible adapters.
- In-memory cosine store and an optional FAISS adapter.
- BM25, dense retrieval, and reciprocal rank fusion.
- `RAGPipeline` with a caller-supplied LLM callable.
- `persian-rag ingest` and `persian-rag query` commands, plus a JSON index.
- Unit tests, Ruff, mypy, and a GitHub Actions workflow.

[0.2.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.2.0
[0.1.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.1.0
