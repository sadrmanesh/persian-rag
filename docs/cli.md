# Command line

```bash
persian-rag --help
persian-rag <command> --help
```

## ingest

```bash
persian-rag ingest ./library --index .persian-rag
persian-rag ingest ./library https://example.com/post --index .persian-rag
persian-rag ingest ./library --index .persian-rag --update
```

Accepts any mix of files, folders, and `http(s)` URLs. Reads `.txt`, `.md`, `.html`, `.pdf`, and `.docx` files and writes a UTF-8 JSON index (`index.json`) with normalized chunks and vectors. API keys are never stored.

| Flag | Default | Purpose |
| --- | --- | --- |
| `--embedder` | `hashing` | `hashing`, `sentence-transformers`, or `openai` |
| `--model`, `--dimension`, `--base-url` | | Embedder settings |
| `--strategy` | `recursive` | `fixed`, `sentence`, or `recursive` |
| `--chunk-size`, `--overlap`, `--sentence-overlap` | 800, 120, 1 | Chunking |
| `--digits`, `--zwnj`, `--remove-diacritics`, `--no-half-space` | | Normalization |
| `--rtl-fix` | `auto` | PDF reading order repair: `auto`, `reverse`, `none` |
| `--split-pages` | off | One document per PDF page |
| `--embedding-cache PATH` | off | Reuse embeddings from a SQLite cache |
| `--split-sections` | off | One document per Markdown heading |
| `--html-extract` | `main` | `main` keeps the article body, `all` keeps every visible block |
| `--update` | off | Update an existing index, re-embedding only new or changed documents ([details](incremental.md)) |
| `--keep-missing` | off | With `--update`, keep documents that are not in the input |

## query

```bash
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --k 5 --json
OPENAI_API_KEY=... persian-rag query "پرسش" --llm openai --rerank llm
persian-rag query "اطاق پنج دری" --expand rules
persian-rag query "پرسش" --llm ollama --citations
```

With `--citations` the answer is followed by a numbered source list with the quoted spans.

## chat

```bash
persian-rag chat --index .persian-rag --llm ollama --citations --max-turns 6
```

An interactive session with conversation memory. `--condense auto|llm|heuristic|none` controls how follow-ups become standalone questions ([details](citations-chat.md)). Type `/reset` to clear the memory and `/exit` to quit.

## eval

```bash
persian-rag eval eval.jsonl --index .persian-rag --k 1 3 5 10 --match document
```

Add `--answers` (with `--llm openai`) to score answer faithfulness, and `--judge llm` to use the model as the judge.

## serve

```bash
persian-rag serve --index .persian-rag --port 8000
```

## Shared options

`query`, `chat`, `eval`, and `serve` accept `--index`, `--llm none|openai|ollama`, `--llm-model`, `--llm-max-tokens`, `--base-url`, `--ollama-url`, `--prompt-lang fa|en`, `--citations`, `--expand none|rules|llm`, `--rerank none|cross-encoder|llm`, `--rerank-model`, and `--rerank-candidates`. See [LLM adapters](llm.md) and [Query rewriting](query-rewriting.md).
