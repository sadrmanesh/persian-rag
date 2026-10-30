# Changelog

All notable changes to persian-rag are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
