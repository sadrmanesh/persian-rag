# Incremental indexing

New in 0.3.0. Re-running ingestion on a changed folder only touches what changed.

```python
from persian_rag import IncrementalIndexer, load_documents

indexer = IncrementalIndexer(retriever, chunking=chunking)
report = indexer.sync(load_documents("./library"))
print(report.summary())
# added 2, updated 1, deleted 0, unchanged 40 document(s); +9 / -3 chunk(s), 12 vector(s) reused
```

* Every document gets a SHA-256 `content_hash` of its normalized text, metadata, normalization settings, and chunking settings. Changing any of them marks the document as updated.
* Updated documents are re-chunked; chunks whose text did not change keep their vectors, so only new text is embedded.
* `sync(documents)` deletes documents that are no longer present. Use `delete_missing=False`, or `upsert(documents)`, to only add and replace.
* `delete(["old.md"])` removes documents by id.
* `indexer.manifest` (`IndexManifest`) maps document ids to their hash and chunk ids. It is plain JSON through `to_dict()` / `from_dict()`, and `save_index(..., manifest=...)` stores it in `index.json`.

Deletes require a store with a `delete` method: the in-memory, Qdrant, pgvector, Elasticsearch, and OpenSearch stores have one; FAISS does not.

## Command line

```bash
persian-rag ingest ./library --index .persian-rag            # full build, stores a manifest
persian-rag ingest ./library --index .persian-rag --update   # only what changed
persian-rag ingest ./new-notes --index .persian-rag --update --keep-missing
```

`--update` loads the existing index, compares hashes, and prints a summary. `--keep-missing` keeps documents that are not in the given paths. The embedder and chunking settings stored in the index are reused, so the flags only need to name the paths. Indexes written by 0.2.0 have no manifest; the first `--update` treats every document as updated (reusing the vectors of unchanged chunks) and writes one.
