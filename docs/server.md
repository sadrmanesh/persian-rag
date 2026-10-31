# HTTP API

```bash
pip install "persian-rag[server]"
persian-rag serve --index .persian-rag --host 127.0.0.1 --port 8000
```

Or with Docker, from the image published on every release (see [Docker](docker.md)):

```bash
docker run -d -p 8000:8000 -v persian-rag-data:/data ghcr.io/sadrmanesh/persian-rag:0.4.0
```

Or from code:

```python
import uvicorn
from persian_rag.server import create_app

app = create_app(pipeline, title="My Persian search", max_k=50)
uvicorn.run(app, host="127.0.0.1", port=8000)
```

`create_app` returns a regular FastAPI application, so you can mount it, add middleware, or add your own routes. Handlers are async and use `RAGPipeline.aquery`, so slow model calls do not block other requests.

## Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | | status, version, chunk count, whether an LLM and reranker are set, and `embedder` (`kind`, `model`, `dimension`, `query_prefix`, `passage_prefix`) |
| `POST` | `/retrieve` | `{"query": "...", "k": 5, "mode": "hybrid"}` | `{"query", "results": [...]}` |
| `POST` | `/query` | `{"question": "...", "k": 5, "mode": "hybrid", "include_prompt": false}` | `{"question", "answer", "contexts", "citations", "uncited", "prompt"}` |
| `POST` | `/chat` | `{"message": "...", "session_id": null, "k": 5, "reset": false}` | the `/query` fields plus `session_id`, `standalone_question`, `turn` |
| `DELETE` | `/chat/{session_id}` | | `{"deleted": true}`, or `false` for an unknown session |

`/chat` keeps a conversation per `session_id`; see [Citations and chat](citations-chat.md). `create_app(..., max_sessions=1000, max_turns=6)` bounds the memory it uses. `citations` is filled when the pipeline has `citations=True`.

Each result has the same fields as `RetrievalResult.to_dict()`. Requests with an empty query, an unknown mode, or `k` above `max_k` get HTTP 422. Interactive documentation is served at `/docs`.

```bash
curl -s localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "حافظ اهل کجا بود؟", "k": 3}'
```

The server has no authentication. Put it behind your own gateway before exposing it.
