# Changelog

All notable changes to persian-rag are documented here. The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `intfloat/multilingual-e5-base` as a supported, more accurate embedding model. `SentenceTransformerEmbedder("e5-base")`, `--model e5-base`, and the full model name all work, with the E5 `query: `/`passage: ` prefixes added automatically. e5-small stays the default.
- Short model names: `e5-small` and `e5-base` resolve to the full `intfloat/multilingual-e5-*` names wherever a `sentence-transformers` model name is accepted (Python, CLI, saved index specs, `run_benchmark.py`). Indexes store the full name.
- `E5_BASE_MODEL`, `ST_MODEL_ALIASES`, and `resolve_model_name` in the public API.
- e5-small against e5-base in the docs and README. nDCG@10, normalized text: MIRACL fa dev 0.7461 against 0.7657, PQuAD validation 0.9242 against 0.9247, persian-rag-bench 0.9881 against 1.0000. With perturbed queries (Arabic yeh and kaf, no half spaces, no normalization): MIRACL 0.6645 against 0.7089, PQuAD 0.8639 against 0.8906. e5-base is 278M parameters and 768 dimensions against 118M and 384, and encodes about 2.4 times slower on CPU.

## [0.4.0] - 2026-10-31

### Changed

- **Breaking default:** `SentenceTransformerEmbedder()` and `persian-rag ingest --embedder sentence-transformers` now use `intfloat/multilingual-e5-small` instead of `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Both have 118M parameters and 384 dimensions, but their vectors are not compatible. Code that names a model, and indexes built with any model, keep working, because indexes store their model. Rebuild indexes made with the old default to use the new model, or pass `--model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` to keep it.
- The Docker image sets `HF_HOME=/data/.cache/huggingface`, so models are cached in the `/data` volume when the image is built with the `sentence-transformers` extra.
- `benchmarks/run_benchmark.py` defaults to e5-small for `--embedder sentence-transformers` and has a `--prefix` option.
- The package version is now 0.4.0.

### Added

- E5 query and passage prefixes. `SentenceTransformerEmbedder(prefix="auto")` adds `query: ` and `passage: ` for E5 models (`multilingual-e5-*`, `e5-*`) and `persian-rag-embed` models, and nothing for other models. When a model defines non-empty `query` and `document` prompts, those are used through `encode_query` and `encode_document`. `prefix="e5"` and `prefix="none"` force the choice, and `query_prefix` and `passage_prefix` set custom strings.
- `persian-rag ingest --prefix auto|e5|none`, `--query-prefix`, and `--passage-prefix`.
- Indexes store the resolved prefixes, so a saved index is always queried the way it was built. Indexes from earlier versions have no stored prefixes and are queried without them; an E5 index from before 0.4.0 also gives a warning that suggests a rebuild.
- Model mismatch detection: loading an index with an embedder of a different kind or model raises `IndexFormatError` with a rebuild hint, and different prefixes give a warning.
- `GET /health` returns an `embedder` object with `kind`, `model`, `dimension`, `query_prefix`, and `passage_prefix`.
- `DEFAULT_ST_MODEL`, `LEGACY_ST_MODEL`, `is_e5_model`, and `resolve_prefixes` in the public API.
- An optional Docker Hub job in the release workflow. When the `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets are set, it copies the released GHCR image to `docker.io/<username>/persian-rag` with the same tags. Without them it skips.
- Links to the [live demo](https://huggingface.co/spaces/sadrmanesh/persian-rag-demo) on Hugging Face Spaces and the [benchmark dataset](https://huggingface.co/datasets/sadrmanesh/persian-rag-bench) on the Hugging Face Hub in the README and docs, with a Spaces badge.
- `.github/workflows/mirror.yml`, which mirrors `main` and tags to `gitlab.com/sadrmanesh/persian-rag` when the `GITLAB_TOKEN` secret is set, and skips cleanly otherwise.
- Embedding model results in the docs and README. nDCG@10 on CPU with normalized text, MiniLM-L12-v2 against e5-small with prefixes:
  - MIRACL fa dev (632 queries, pooled corpus of 6,405 judged passages, not the full MIRACL corpus, so not comparable with published MIRACL scores): 0.4424 against 0.7461 (MRR@10 0.4271 against 0.7138, Recall@10 0.6356 against 0.9521).
  - PQuAD validation (5,995 questions, 1,103 paragraphs): 0.6437 against 0.9242 (MRR@10 0.5905 against 0.9042, Recall@10 0.8113 against 0.9845).
  - persian-rag-bench, dense: 0.8637 against 0.9881. Without prefixes e5-small scores 0.9824, and 0.8333 instead of 0.9167 Hit@1 on the spelling subset.

### Fixed

- `SentenceTransformerEmbedder` no longer logs a deprecation warning on newer sentence-transformers versions, which renamed `get_sentence_embedding_dimension` to `get_embedding_dimension`.

## [0.3.1] - 2026-10-30

### Added

- A production `Dockerfile` for the HTTP API. The image installs `persian-rag[server,pdf,docx]` on Python 3.12 from a wheel built in a separate stage, runs as a non-root user, serves `/data/index` on port 8000, and has a health check on `GET /health`. The `EXTRAS` build argument adds other extras, such as `sentence-transformers`.
- A `.dockerignore` that keeps tests, docs, caches, and local indexes out of the build context.
- The release workflow publishes the image to the GitHub Container Registry as `ghcr.io/sadrmanesh/persian-rag` for `linux/amd64` and `linux/arm64`, tagged `X.Y.Z`, `X.Y`, and `latest`, with OCI labels that link it to this repository.
- CI builds the image, indexes a sample document inside it, checks `/health` and `/retrieve`, waits for the health check to report healthy, and checks that the container does not run as root.
- A Docker page in English and Persian, and Docker sections in the README, the HTTP API page, and the release guide.

### Changed

- The package version is now 0.3.1.

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

[0.4.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.4.0
[0.3.1]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.3.1
[0.3.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.3.0
[0.2.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.2.0
[0.1.0]: https://github.com/sadrmanesh/persian-rag/releases/tag/v0.1.0
