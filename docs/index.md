# persian-rag

Persian-first retrieval-augmented generation for Python.

persian-rag turns a folder of Persian documents into a searchable index and a question answering pipeline. Each stage is small, typed, and replaceable:

1. **Load** text, Markdown, HTML, PDF, and Word files and web pages, with main-content extraction and reversed PDF text repair.
2. **Normalize** Arabic and Persian letters, half-spaces, digits, and punctuation.
3. **Chunk** text by size, sentences, or recursive separators.
4. **Embed** with sentence-transformers, any OpenAI-compatible API, or your own class, with optional caching.
5. **Store** vectors in memory, FAISS, Qdrant, PostgreSQL with pgvector, Elasticsearch, or OpenSearch, and update them incrementally.
6. **Retrieve** with BM25, dense search, or both fused with reciprocal rank fusion, optionally expanding queries with Persian spelling variants.
7. **Rerank** with a cross-encoder or an LLM.
8. **Generate** with any sync or async function, or the built-in OpenAI-compatible and Ollama adapters, with source citations and conversation memory.
9. **Evaluate** retrieval (hit rate, MRR, recall, nDCG) and answer faithfulness.
10. **Serve** the pipeline over HTTP with FastAPI.

The core package has no dependencies. Heavy pieces are optional extras, so you install only what you use.

## Where to start

- New here? Read [Getting started](getting-started.md).
- Working with scanned or exported PDFs? See [Loading documents](loaders.md).
- Tuning quality? See [Retrieval and reranking](retrieval.md) and [Evaluation](evaluation.md).
- Deploying? See [Vector stores](vector-stores.md), [Elasticsearch and OpenSearch](search-engines.md), [Incremental indexing](incremental.md), and [HTTP API](server.md).
- Building a chat assistant? See [Citations and chat](citations-chat.md) and [LLM adapters](llm.md).
- Comparing settings? See [Benchmarks](benchmarks.md).

Every page is also available in Persian through the language selector.
