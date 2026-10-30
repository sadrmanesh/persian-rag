# Elasticsearch and OpenSearch

New in 0.3.0. `ElasticsearchStore` and `OpenSearchStore` keep vectors, text, and metadata in one index and serve both halves of hybrid search: k-NN for the dense side and BM25 with the built-in `persian` analyzer for the sparse side. They talk to the REST API directly, so no client package is required.

```python
from persian_rag import HashingEmbedder, HybridRetriever, build_retriever
from persian_rag.vectorstores import ElasticsearchStore, OpenSearchStore, store_from_url

store = ElasticsearchStore("http://localhost:9200", "persian_docs")
# store = OpenSearchStore("https://search.example.com", "persian_docs", username="admin", password="...")
# store = store_from_url("opensearch+https://search.example.com", "persian_docs")

embedder = HashingEmbedder()
retriever = build_retriever(texts, embedder, document_ids=ids, store=store)

# Later, in another process: the index already holds everything.
retriever = HybridRetriever(embedder, store=store, sparse_backend=store)
results = retriever.retrieve("دیوان حافظ", k=5)  # hybrid, dense, or sparse
```

With `sparse_backend=store` the in-process BM25 index is not used, so a new process does not need to load any chunks before searching.

## Options

| Argument | Default | Purpose |
| --- | --- | --- |
| `url` | `ELASTICSEARCH_URL` or `OPENSEARCH_URL`, then `http://localhost:9200` | Cluster address |
| `index` | `persian_rag` | Lowercase index name |
| `dimension` | read from the mapping | Required only to create a new index before the first `add` |
| `api_key` | `ELASTICSEARCH_API_KEY` / `OPENSEARCH_API_KEY` | Sent as `Authorization: ApiKey ...` |
| `username`, `password` | password from `..._PASSWORD` | Basic authentication |
| `analyzer` | `persian` | Analyzer for the `text` field |
| `refresh` | `wait_for` | Refresh policy for writes: `true`, `false`, or `wait_for` |
| `num_candidates` | 100 | Candidates per shard for Elasticsearch k-NN |
| `verify_tls`, `ca_file`, `timeout` | | Connection settings |
| `transport` | urllib | Inject a custom HTTP function, for tests or a different client |

The index is created on first write with a `dense_vector` (Elasticsearch, cosine similarity, HNSW) or `knn_vector` (OpenSearch, Lucene HNSW, cosine) field. Documents are written with the `_bulk` API, `add` refuses ids that already exist, `upsert` replaces them, and `delete(ids)` removes them. `drop_index()` deletes the whole index.

## Incremental updates

The stores support deletes, so `HybridRetriever.delete_document`, `delete_chunks`, and the [incremental indexer](incremental.md) work with them.

## Local servers

```bash
docker run -p 9200:9200 -e discovery.type=single-node -e xpack.security.enabled=false \
  docker.elastic.co/elasticsearch/elasticsearch:8.15.3
docker run -p 9201:9200 -e discovery.type=single-node -e DISABLE_SECURITY_PLUGIN=true \
  -e DISABLE_INSTALL_DEMO_CONFIG=true opensearchproject/opensearch:2.17.1
```

The integration tests run against both in CI:

```bash
PERSIAN_RAG_ES_URL=http://localhost:9200 PERSIAN_RAG_OPENSEARCH_URL=http://localhost:9201 pytest -m search
```

The command line keeps using the JSON index; the search engine stores are a library feature.
