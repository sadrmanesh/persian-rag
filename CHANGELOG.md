# Changelog

All notable changes to persian-rag are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.1.0
