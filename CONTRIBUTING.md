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
# optional integrations used by the full test suite
pip install -e ".[dev,server,qdrant,pgvector,pdf,docx]"
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

Tests that download model weights must be marked `heavy` and skipped unless `PERSIAN_RAG_RUN_HEAVY=1`. Do not make the default suite depend on network access. Tests for optional extras use `pytest.importorskip` or fakes so the suite also passes with only the core dependencies installed. Tests that need a live PostgreSQL server are marked `postgres` and read the connection string from `PERSIAN_RAG_PG_DSN`.

Documentation lives in `docs/` with Persian pages next to English ones (`page.md` and `page.fa.md`). Build it with `pip install -e ".[docs]"` and `mkdocs build --strict`.

## Pull requests

- Describe the user-facing change and how you tested it.
- Do not add benchmark numbers unless the change includes the data, the script, and the command used to produce them.
- Keep API keys, indexes, and virtual environments out of the commit.
- Version bumps belong in `src/persian_rag/version.py`, `pyproject.toml`, and `CHANGELOG.md` together.

## Scope of 0.2.x

Bug fixes and small, tested improvements to normalization, chunking, retrieval, loaders, evaluation, the server, and the CLI are in scope. New vector-store backends, parsers, and hosted services are better discussed in an issue first so the public API stays small.
