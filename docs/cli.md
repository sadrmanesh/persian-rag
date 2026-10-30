# Command line

```bash
persian-rag --help
persian-rag <command> --help
```

## ingest

```bash
persian-rag ingest ./library --index .persian-rag
```

Reads `.txt`, `.md`, `.pdf`, and `.docx` files and writes a UTF-8 JSON index (`index.json`) with normalized chunks and vectors. API keys are never stored.

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

## query

```bash
persian-rag query "حافظ اهل کجا بود؟" --index .persian-rag --k 5 --json
OPENAI_API_KEY=... persian-rag query "پرسش" --llm openai --rerank llm
```

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

`query`, `eval`, and `serve` accept `--index`, `--llm none|openai`, `--llm-model`, `--base-url`, `--prompt-lang fa|en`, `--rerank none|cross-encoder|llm`, `--rerank-model`, and `--rerank-candidates`.
