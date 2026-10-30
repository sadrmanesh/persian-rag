"""An in-memory stand-in for the Elasticsearch/OpenSearch REST API."""

from __future__ import annotations

import json
import math
import urllib.parse
from collections.abc import Mapping
from typing import Any

from persian_rag.tokenize import tokenize
from persian_rag.vectorstores.elasticsearch import HttpResponse


class FakeSearchCluster:
    def __init__(self) -> None:
        self.indexes: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def __call__(
        self, method: str, url: str, body: bytes | None, headers: Mapping[str, str]
    ) -> HttpResponse:
        parts = urllib.parse.urlsplit(url)
        path = parts.path.strip("/")
        self.headers.append(dict(headers))
        if headers.get("Content-Type") == "application/x-ndjson":
            payload: Any = [
                json.loads(line) for line in (body or b"").decode().splitlines() if line
            ]
        else:
            payload = json.loads(body) if body else None
        self.calls.append((method, path, payload))
        segments = path.split("/")
        name = segments[0]
        if path == "_bulk":
            return self._bulk(payload)
        if len(segments) == 1:
            if method == "HEAD":
                return HttpResponse(200 if name in self.indexes else 404, None)
            if method == "PUT":
                self.indexes[name] = {"body": payload, "docs": {}}
                return HttpResponse(200, {"acknowledged": True})
            if method == "DELETE":
                existed = self.indexes.pop(name, None)
                return HttpResponse(200 if existed else 404, {})
        index = self.indexes.get(name)
        if index is None:
            return HttpResponse(404, {"error": {"type": "index_not_found_exception"}})
        action = segments[1]
        if action == "_mapping":
            return HttpResponse(200, {name: {"mappings": index["body"]["mappings"]}})
        if action == "_count":
            return HttpResponse(200, {"count": len(index["docs"])})
        if action == "_mget":
            docs = [{"_id": i, "found": i in index["docs"]} for i in payload["ids"]]
            return HttpResponse(200, {"docs": docs})
        if action == "_search":
            return HttpResponse(200, self._search(index, payload))
        return HttpResponse(400, {"error": f"unsupported {method} {path}"})

    def _bulk(self, lines: list[dict[str, Any]]) -> HttpResponse:
        items = []
        position = 0
        while position < len(lines):
            header = lines[position]
            action, meta = next(iter(header.items()))
            index = self.indexes.setdefault(meta["_index"], {"body": {}, "docs": {}})
            if action == "index":
                index["docs"][meta["_id"]] = lines[position + 1]
                items.append({"index": {"_id": meta["_id"], "status": 201}})
                position += 2
            else:
                found = index["docs"].pop(meta["_id"], None)
                items.append({"delete": {"_id": meta["_id"], "status": 200 if found else 404}})
                position += 1
        errors = any(item[next(iter(item))]["status"] >= 400 for item in items)
        return HttpResponse(200, {"errors": errors, "items": items})

    def _search(self, index: dict[str, Any], body: dict[str, Any]) -> dict[str, Any]:
        size = int(body.get("size", 10))
        docs = index["docs"]
        scored: list[tuple[float, str]] = []
        if "knn" in body:
            vector = body["knn"]["query_vector"]
            scored = [(_cosine(vector, doc["embedding"]), doc_id) for doc_id, doc in docs.items()]
        elif "knn" in body.get("query", {}):
            vector = body["query"]["knn"]["embedding"]["vector"]
            scored = [(_cosine(vector, doc["embedding"]), doc_id) for doc_id, doc in docs.items()]
        else:
            terms = set(tokenize(body["query"]["match"]["text"]["query"]))
            for doc_id, doc in docs.items():
                overlap = len(terms & set(tokenize(doc["text"])))
                if overlap:
                    scored.append((float(overlap), doc_id))
        scored.sort(key=lambda item: (-item[0], item[1]))
        hits = []
        for score, doc_id in scored[:size]:
            source = {k: v for k, v in docs[doc_id].items() if k != "embedding"}
            hits.append({"_id": doc_id, "_score": score, "_source": source})
        return {"hits": {"hits": hits}}


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return (1 + (dot / norm if norm else 0.0)) / 2
