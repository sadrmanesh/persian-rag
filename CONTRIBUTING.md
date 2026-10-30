# Contributing to persian-rag

Thanks for considering a contribution. The project is maintained by Naser Sadrmanesh ([info.sadrmanesh@gmail.com](mailto:info.sadrmanesh@gmail.com)).

Issues and pull requests may be written in Persian or English.

## Setup

```bash
git clone https://github.com/sadrmanesh/persian-rag
cd persian-rag
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Python 3.10 or newer is required.

## Checks

Run these before opening a pull request:

```bash
ruff check .
ruff format --check .
mypy src
pytest
```

`ruff format .` rewrites formatting. Keep the public modules typed. New behavior needs a unit test. Persian fixtures belong in `tests/fixtures/` or inline in the test when they are only a few words.

Tests that download model weights must be marked `heavy` and skipped unless `PERSIAN_RAG_RUN_HEAVY=1`. Do not make the default suite depend on network access or on the `sentence-transformers` and `faiss` extras.

## Pull requests

- Describe the user-facing change and how you tested it.
- Do not add benchmark numbers unless the change includes the data, the script, and the command used to produce them.
- Keep API keys, indexes, and virtual environments out of the commit.
- Version bumps belong in `src/persian_rag/version.py`, `pyproject.toml`, and `CHANGELOG.md` together.

## Scope of 0.1.x

Bug fixes and small, tested improvements to normalization, chunking, retrieval, and the CLI are in scope. New vector-store backends, parsers, and hosted services are better discussed in an issue first so the public API stays small.
