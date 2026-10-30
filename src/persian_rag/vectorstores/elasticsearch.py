"""Elasticsearch and OpenSearch store with server-side BM25.

One index holds the chunk text, its position, metadata, and the embedding.
The store implements the :class:`~persian_rag.store.VectorStore` protocol for
dense k-NN search and also ``search_text`` for BM25, so a single object can
serve both sides of :class:`~persian_rag.retrieval.HybridRetriever`::

    store = ElasticsearchStore("http://localhost:9200", "persian_docs", dimension=384)
    retriever = HybridRetriever(embedder, store=store, sparse_backend=store)

The client speaks the REST API with the standard library, so no extra
package is needed. Both engines ship a built-in ``persian`` analyzer, which
is the default for the text field.

Requirements: Elasticsearch 8.x (``dense_vector`` with ``knn`` search) or
OpenSearch 2.x with the k-NN plugin (``knn_vector`` with the Lucene engine).
"""

from __future__ import annotations

import base64
import json
import os
import ssl
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from persian_rag.errors import PersianRagError
from persian_rag.store import VectorHit, _check_parallel, _metadata_rows, _shared_dimension

Flavor = Literal["elasticsearch", "opensearch"]
Refresh = Literal["true", "false", "wait_for"]

_CHUNK_FIELDS = ("_document_id", "_start", "_end", "_index")


@dataclass(frozen=True, slots=True)
class HttpResponse:
    """Status and decoded JSON body (``None`` when the body is empty)."""

    status: int
    body: Any


Transport = Callable[[str, str, bytes | None, Mapping[str, str]], HttpResponse]
"""``(method, url, body, headers) -> HttpResponse``; injectable for tests."""


def urllib_transport(
    timeout: float = 30.0, *, verify: bool = True, ca_file: str | None = None
) -> Transport:
    """Return a transport built on :mod:`urllib.request`."""

    context: ssl.SSLContext | None = None
    if ca_file is not None:
        context = ssl.create_default_context(cafile=ca_file)
    elif not verify:
        context = ssl.create_default_context()
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE

    def send(method: str, url: str, body: bytes | None, headers: Mapping[str, str]) -> HttpResponse:
        request = urllib.request.Request(url, data=body, method=method, headers=dict(headers))
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=context) as response:
                return HttpResponse(int(response.status), _decode(response.read()))
        except urllib.error.HTTPError as exc:
            return HttpResponse(int(exc.code), _decode(exc.read()))
        except urllib.error.URLError as exc:
            raise PersianRagError(f"Request to {url} failed: {exc.reason}") from exc

    return send


def _decode(raw: bytes) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return raw.decode("utf-8", errors="replace")


class ElasticsearchStore:
    """Dense and lexical search on an Elasticsearch index.

    Parameters:
        url: Cluster URL. Defaults to ``$ELASTICSEARCH_URL`` or
            ``http://localhost:9200``.
        index: Index name. Created on first write when ``create_index`` is true.
        dimension: Vector size. Optional when the index already exists or
            until the first write.
        api_key: Sent as ``Authorization: ApiKey ...``. Defaults to
            ``$ELASTICSEARCH_API_KEY``.
        username, password: HTTP basic authentication. The password defaults
            to ``$ELASTICSEARCH_PASSWORD``.
        analyzer: Analyzer for the text field (default ``persian``).
        refresh: Refresh policy for writes. ``wait_for`` makes new chunks
            searchable before the call returns.
        transport: Replace the HTTP layer, for example in tests.

    Scores follow the engine: for cosine similarity both engines report
    ``(1 + cosine) / 2``, so larger is closer. BM25 scores are unbounded.
    """

    flavor: Flavor = "elasticsearch"
    stores_chunk_fields = True
    _env_prefix = "ELASTICSEARCH"

    def __init__(
        self,
        url: str | None = None,
        index: str = "persian_rag",
        dimension: int | None = None,
        *,
        api_key: str | None = None,
        username: str | None = None,
        password: str | None = None,
        analyzer: str = "persian",
        refresh: Refresh = "wait_for",
        create_index: bool = True,
        num_candidates: int = 100,
        timeout: float = 30.0,
        verify_tls: bool = True,
        ca_file: str | None = None,
        transport: Transport | None = None,
    ) -> None:
        if dimension is not None and dimension < 1:
            raise ValueError("dimension must be >= 1")
        if refresh not in ("true", "false", "wait_for"):
            raise ValueError("refresh must be 'true', 'false', or 'wait_for'")
        if num_candidates < 1:
            raise ValueError("num_candidates must be >= 1")
        if not index or index != index.lower() or "/" in index:
            raise ValueError("index must be a non-empty lowercase name without '/'")
        prefix = self._env_prefix
        self.url = (url or os.environ.get(f"{prefix}_URL") or "http://localhost:9200").rstrip("/")
        self.index = index
        self.analyzer = analyzer
        self.refresh = refresh
        self.create_index = create_index
        self.num_candidates = num_candidates
        self._transport = transport or urllib_transport(timeout, verify=verify_tls, ca_file=ca_file)
        self._headers = {"Content-Type": "application/json", "Accept": "application/json"}
        key = api_key if api_key is not None else os.environ.get(f"{prefix}_API_KEY")
        if key:
            self._headers["Authorization"] = f"ApiKey {key}"
        elif username:
            secret = password if password is not None else os.environ.get(f"{prefix}_PASSWORD", "")
            token = base64.b64encode(f"{username}:{secret}".encode()).decode("ascii")
            self._headers["Authorization"] = f"Basic {token}"
        self._dimension = dimension
        self._exists: bool | None = None

    # ------------------------------------------------------------------ HTTP

    def _request(
        self,
        method: str,
        path: str,
        payload: Any = None,
        *,
        ndjson: Sequence[Mapping[str, Any]] | None = None,
        allow: tuple[int, ...] = (),
    ) -> HttpResponse:
        headers = dict(self._headers)
        body: bytes | None = None
        if ndjson is not None:
            lines = [json.dumps(item, ensure_ascii=False) for item in ndjson]
            body = ("\n".join(lines) + "\n").encode("utf-8")
            headers["Content-Type"] = "application/x-ndjson"
        elif payload is not None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        response = self._transport(method, f"{self.url}/{path.lstrip('/')}", body, headers)
        if response.status >= 400 and response.status not in allow:
            raise PersianRagError(
                f"{self.flavor} returned HTTP {response.status} for {method} /{path}: "
                f"{_error_reason(response.body)}"
            )
        return response

    # --------------------------------------------------------------- mapping

    @property
    def dimension(self) -> int | None:
        return self._dimension

    def index_exists(self) -> bool:
        if self._exists is None:
            response = self._request("HEAD", self.index, allow=(404,))
            self._exists = response.status < 400
            if self._exists and self._dimension is None:
                self._dimension = self._mapped_dimension()
        return self._exists

    def _mapped_dimension(self) -> int | None:
        response = self._request("GET", f"{self.index}/_mapping")
        body = response.body if isinstance(response.body, dict) else {}
        for entry in body.values():
            props = (
                entry.get("mappings", {}).get("properties", {}) if isinstance(entry, dict) else {}
            )
            vector = props.get("embedding", {})
            size = vector.get("dims", vector.get("dimension"))
            if isinstance(size, int):
                return size
        return None

    def vector_mapping(self, dimension: int) -> dict[str, Any]:
        return {"type": "dense_vector", "dims": dimension, "index": True, "similarity": "cosine"}

    def index_settings(self) -> dict[str, Any]:
        return {}

    def index_body(self, dimension: int) -> dict[str, Any]:
        """The settings and mappings used when the store creates its index."""

        body: dict[str, Any] = {
            "mappings": {
                "properties": {
                    "text": {"type": "text", "analyzer": self.analyzer},
                    "document_id": {"type": "keyword"},
                    "start": {"type": "integer"},
                    "end": {"type": "integer"},
                    "chunk_index": {"type": "integer"},
                    "metadata": {"type": "object", "enabled": False},
                    "embedding": self.vector_mapping(dimension),
                }
            }
        }
        settings = self.index_settings()
        if settings:
            body["settings"] = settings
        return body

    def ensure_index(self, dimension: int) -> None:
        """Create the index with the expected mapping when it does not exist."""

        if self.index_exists():
            if self._dimension is not None and self._dimension != dimension:
                raise ValueError(f"Expected dimension {self._dimension}, got {dimension}")
            self._dimension = dimension
            return
        if not self.create_index:
            raise PersianRagError(f"Index {self.index!r} does not exist")
        self._request("PUT", self.index, self.index_body(dimension))
        self._exists = True
        self._dimension = dimension

    def drop_index(self) -> None:
        """Delete the whole index. Missing indexes are ignored."""

        self._request("DELETE", self.index, allow=(404,))
        self._exists = False

    # ------------------------------------------------------------------ write

    def add(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Add new chunks. Ids that already exist in the index are rejected."""

        _check_parallel(ids, vectors, texts, metadatas)
        if not ids:
            return
        self.ensure_index(_shared_dimension(vectors))
        taken = self._existing(ids)
        if taken:
            raise ValueError(f"Duplicate vector id: {', '.join(sorted(taken))}")
        self._bulk_index(ids, vectors, texts, metadatas)

    def upsert(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        """Insert or replace chunks by id."""

        _check_parallel(ids, vectors, texts, metadatas)
        if not ids:
            return
        self.ensure_index(_shared_dimension(vectors))
        self._bulk_index(ids, vectors, texts, metadatas)

    def _bulk_index(
        self,
        ids: Sequence[str],
        vectors: Sequence[Sequence[float]],
        texts: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]] | None,
    ) -> None:
        width = self._dimension
        lines: list[dict[str, Any]] = []
        for doc_id, vector, text, meta in zip(
            ids, vectors, texts, _metadata_rows(metadatas, len(ids)), strict=True
        ):
            if width is not None and len(vector) != width:
                raise ValueError(f"Expected dimension {width}, got {len(vector)}")
            lines.append({"index": {"_index": self.index, "_id": doc_id}})
            lines.append(_source(text, vector, meta))
        self._bulk(lines)

    def delete(self, ids: Sequence[str]) -> None:
        """Remove chunks by id. Unknown ids are ignored."""

        if not ids or not self.index_exists():
            return
        self._bulk([{"delete": {"_index": self.index, "_id": doc_id}} for doc_id in ids])

    def _bulk(self, lines: Sequence[Mapping[str, Any]]) -> None:
        response = self._request("POST", f"_bulk?refresh={self.refresh}", ndjson=lines)
        body = response.body if isinstance(response.body, dict) else {}
        if not body.get("errors"):
            return
        problems = []
        for item in body.get("items", []):
            for action, result in item.items():
                status = int(result.get("status", 200))
                if status >= 400 and not (action == "delete" and status == 404):
                    problems.append(f"{result.get('_id')}: {_error_reason(result)}")
        if problems:
            raise PersianRagError("Bulk request failed for " + "; ".join(problems[:5]))

    def _existing(self, ids: Sequence[str]) -> set[str]:
        response = self._request("POST", f"{self.index}/_mget", {"ids": list(ids)}, allow=(404,))
        body = response.body if isinstance(response.body, dict) else {}
        return {str(doc["_id"]) for doc in body.get("docs", []) if doc.get("found")}

    # ------------------------------------------------------------------- read

    def knn_query(self, vector: list[float], k: int) -> dict[str, Any]:
        return {
            "size": k,
            "knn": {
                "field": "embedding",
                "query_vector": vector,
                "k": k,
                "num_candidates": max(self.num_candidates, k),
            },
        }

    def search(self, vector: Sequence[float], k: int = 5) -> list[VectorHit]:
        """Approximate nearest neighbours by cosine similarity."""

        if k < 1:
            raise ValueError("k must be >= 1")
        if not self.index_exists():
            return []
        values = [float(item) for item in vector]
        if self._dimension is not None and len(values) != self._dimension:
            raise ValueError(f"Expected dimension {self._dimension}, got {len(values)}")
        return self._hits(self.knn_query(values, k))

    def search_text(self, query: str, k: int = 5) -> list[VectorHit]:
        """BM25 search over the analyzed chunk text."""

        if k < 1:
            raise ValueError("k must be >= 1")
        if not query.strip() or not self.index_exists():
            return []
        return self._hits({"size": k, "query": {"match": {"text": {"query": query}}}})

    def _hits(self, body: dict[str, Any]) -> list[VectorHit]:
        body["_source"] = {"excludes": ["embedding"]}
        response = self._request("POST", f"{self.index}/_search", body)
        payload = response.body if isinstance(response.body, dict) else {}
        hits: list[VectorHit] = []
        for item in payload.get("hits", {}).get("hits", []):
            source = item.get("_source") or {}
            meta = source.get("metadata")
            metadata: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
            metadata.update(
                {
                    "_document_id": source.get("document_id", ""),
                    "_start": source.get("start", 0),
                    "_end": source.get("end", len(str(source.get("text", "")))),
                    "_index": source.get("chunk_index", 0),
                }
            )
            hits.append(
                VectorHit(
                    id=str(item.get("_id")),
                    score=float(item.get("_score") or 0.0),
                    text=str(source.get("text", "")),
                    metadata=metadata,
                )
            )
        return hits

    def __len__(self) -> int:
        if not self.index_exists():
            return 0
        response = self._request("GET", f"{self.index}/_count", allow=(404,))
        body = response.body if isinstance(response.body, dict) else {}
        return int(body.get("count", 0))


class OpenSearchStore(ElasticsearchStore):
    """The same store for OpenSearch 2.x with the k-NN plugin.

    Environment defaults use the ``OPENSEARCH_`` prefix (``OPENSEARCH_URL``,
    ``OPENSEARCH_API_KEY``, ``OPENSEARCH_PASSWORD``).
    """

    flavor: Flavor = "opensearch"
    _env_prefix = "OPENSEARCH"

    def vector_mapping(self, dimension: int) -> dict[str, Any]:
        return {
            "type": "knn_vector",
            "dimension": dimension,
            "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "lucene"},
        }

    def index_settings(self) -> dict[str, Any]:
        return {"index": {"knn": True}}

    def knn_query(self, vector: list[float], k: int) -> dict[str, Any]:
        return {"size": k, "query": {"knn": {"embedding": {"vector": vector, "k": k}}}}


def _source(text: str, vector: Sequence[float], meta: dict[str, Any]) -> dict[str, Any]:
    plain = {key: value for key, value in meta.items() if key not in _CHUNK_FIELDS}
    return {
        "text": text,
        "document_id": str(meta.get("_document_id", "")),
        "start": int(meta.get("_start", 0)),
        "end": int(meta.get("_end", len(text))),
        "chunk_index": int(meta.get("_index", 0)),
        "metadata": plain,
        "embedding": [float(value) for value in vector],
    }


def _error_reason(body: Any) -> str:
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            return str(error.get("reason") or error.get("type") or error)
        if error is not None:
            return str(error)
    text = str(body) if body is not None else ""
    return text[:300] or "no details"


def store_from_url(url: str, index: str, **options: Any) -> ElasticsearchStore:
    """Build a store from ``elasticsearch://`` or ``opensearch://`` style URLs.

    ``opensearch+https://host:9200`` selects OpenSearch over HTTPS;
    ``elasticsearch://host:9200`` selects Elasticsearch over HTTP.
    """

    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme
    flavor, _, transport = scheme.partition("+")
    if flavor not in ("elasticsearch", "opensearch", "es", "os"):
        raise ValueError("URL scheme must start with elasticsearch or opensearch")
    http_scheme = transport or "http"
    target = urllib.parse.urlunsplit((http_scheme, parsed.netloc, parsed.path, "", ""))
    cls: type[ElasticsearchStore] = (
        OpenSearchStore if flavor in ("opensearch", "os") else ElasticsearchStore
    )
    return cls(target, index, **options)
