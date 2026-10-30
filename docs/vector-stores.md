# Vector stores

All stores implement the same protocol: `add(ids, vectors, texts, metadatas)`, `search(vector, k)`, and `len()`. Scores are always larger-is-closer. Pass a store to `build_retriever(..., store=...)` or `HybridRetriever(embedder, store)`.

| Store | Extra | Persistence | Delete |
| --- | --- | --- | --- |
| `InMemoryVectorStore` | none | process memory | no |
| `FaissVectorStore` | `faiss` | process memory | no |
| `QdrantVectorStore` | `qdrant` | server, cloud, local folder, or memory | yes |
| `PgVectorStore` | `pgvector` | PostgreSQL | yes |

## Qdrant

```python
from persian_rag.vectorstores import QdrantVectorStore

server = QdrantVectorStore("persian_docs", dimension=384, url="http://localhost:6333")
cloud = QdrantVectorStore("persian_docs", dimension=384, url="https://...", api_key="...")
on_disk = QdrantVectorStore("persian_docs", dimension=384, path="./qdrant-data")
in_memory = QdrantVectorStore("persian_docs", dimension=384)
```

String ids are mapped to deterministic UUIDs; the original id, text, and metadata live in the payload. The collection is created on first use (`distance="cosine"`, `"dot"`, or `"euclid"`). When reopening an existing collection the dimension is read from it. Pass `client=` to reuse a configured `QdrantClient`.

## PostgreSQL with pgvector

```python
from persian_rag.vectorstores import PgVectorStore

store = PgVectorStore(
    "postgresql://user:pass@localhost:5432/db",
    dimension=384,
    table="persian_chunks",
    distance="cosine",  # or "l2", "inner_product"
    create_index=True,  # HNSW index
)
```

The store creates the `vector` extension (when the role may), a table with `id`, `embedding`, `text`, and `metadata JSONB`, and optionally an HNSW index. Pass `connection=` to reuse an autocommit psycopg 3 connection. Table names are validated, and all values are sent as query parameters.

To run a local database for development:

```bash
docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 pgvector/pgvector:pg16
```

## FAISS

```python
from persian_rag import FaissVectorStore

store = FaissVectorStore(dimension=384, metric="cosine")
```
