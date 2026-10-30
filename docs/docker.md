# Docker

New in 0.3.1. Every release publishes a Docker image of the HTTP API to the GitHub Container Registry:

```text
ghcr.io/sadrmanesh/persian-rag
```

| Tag | Points to |
| --- | --- |
| `0.3.1` | That exact release |
| `0.3` | The newest 0.3.x release |
| `latest` | The newest release |

Images are built for `linux/amd64` and `linux/arm64`. They contain `persian-rag[server,pdf,docx]` on Python 3.12, run as a non-root user (uid 10001), and listen on port 8000. The working directory and volume is `/data`, and the default command serves the index in `/data/index`.

## Index and serve

Build an index from a folder of documents into a named volume, then serve it:

```bash
docker volume create persian-rag-data

docker run --rm \
  -v "$PWD/library:/library:ro" \
  -v persian-rag-data:/data \
  ghcr.io/sadrmanesh/persian-rag:0.3.1 ingest /library --index /data/index

docker run -d --name persian-rag -p 8000:8000 \
  -v persian-rag-data:/data \
  ghcr.io/sadrmanesh/persian-rag:0.3.1
```

```bash
curl -s localhost:8000/health
curl -s localhost:8000/retrieve -H 'content-type: application/json' \
  -d '{"query": "حافظ اهل کجا بود؟", "k": 3}'
```

The entrypoint is the `persian-rag` command, so any subcommand works: `ingest`, `query`, `chat`, `eval`, or `serve` with your own flags. An index you built on the host can be mounted directly with `-v "$PWD/.persian-rag:/data/index:ro"`.

## Answers with a language model

The image includes the OpenAI-compatible and Ollama adapters. Pass the key as an environment variable and the flags to `serve`:

```bash
docker run -d -p 8000:8000 -v persian-rag-data:/data \
  -e OPENAI_API_KEY \
  ghcr.io/sadrmanesh/persian-rag:0.3.1 \
  serve --index /data/index --host 0.0.0.0 --port 8000 --llm openai --citations
```

For Ollama on the host, add `-e OLLAMA_HOST=http://host.docker.internal:11434` (and `--add-host=host.docker.internal:host-gateway` on Linux) and use `--llm ollama`.

## Health check

The image has a Docker `HEALTHCHECK` that calls `GET /health` every 30 seconds. `docker ps` shows the container as `healthy` once the index is loaded. If you change the port in the command, also set `-e PERSIAN_RAG_PORT=<port>` so the check uses it.

## Building the image yourself

```bash
docker build -t persian-rag .
docker build -t persian-rag --build-arg EXTRAS=server,pdf,docx,sentence-transformers .
```

`EXTRAS` selects the [extras](getting-started.md) installed in the image. The published image leaves out `sentence-transformers`, which adds a few gigabytes; build your own image with it to serve an index made with `--embedder sentence-transformers`.

The server has no authentication. Put it behind your own gateway before exposing it.
