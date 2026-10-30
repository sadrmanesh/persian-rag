"""Command line interface for ingesting files and querying an index."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import cast

from persian_rag.chunk import ChunkingConfig, ChunkStrategy
from persian_rag.documents import load_documents
from persian_rag.embeddings import (
    EmbedderSpec,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from persian_rag.errors import ConfigError, PersianRagError
from persian_rag.normalize import DigitStyle, NormalizeConfig, ZwnjMode
from persian_rag.persist import load_index, retriever_from_saved, save_index
from persian_rag.pipeline import DEFAULT_PROMPT_EN, DEFAULT_PROMPT_FA, RAGPipeline, build_retriever
from persian_rag.remote import make_openai_llm
from persian_rag.retrieval import RetrievalMode
from persian_rag.version import __version__

_DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process status code."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except (PersianRagError, FileNotFoundError, OSError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if not isinstance(result, int):
        print("error: command did not return a status code", file=sys.stderr)
        return 1
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="persian-rag",
        description="Persian-first retrieval-augmented generation",
    )
    parser.add_argument("--version", action="version", version=f"persian-rag {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="Index a text or Markdown file or directory")
    ingest.add_argument("path", type=Path, help="File or directory of .txt and .md files")
    ingest.add_argument(
        "--index",
        type=Path,
        default=Path(".persian-rag"),
        help="Directory (or .json file) to write (default: .persian-rag)",
    )
    ingest.add_argument(
        "--strategy",
        choices=("fixed", "sentence", "recursive"),
        default="recursive",
    )
    ingest.add_argument("--chunk-size", type=int, default=800)
    ingest.add_argument("--overlap", type=int, default=120)
    ingest.add_argument("--sentence-overlap", type=int, default=1)
    ingest.add_argument("--digits", choices=("persian", "ascii", "keep"), default="persian")
    ingest.add_argument(
        "--remove-diacritics",
        action="store_true",
        help="Strip Arabic vowel marks before indexing",
    )
    ingest.add_argument("--zwnj", choices=("normalize", "remove", "space"), default="normalize")
    ingest.add_argument(
        "--no-half-space",
        action="store_true",
        help="Do not join common Persian affixes with ZWNJ",
    )
    ingest.add_argument("--no-clean-punctuation", action="store_true")
    ingest.add_argument(
        "--embedder",
        choices=("hashing", "sentence-transformers", "openai"),
        default="hashing",
        help="hashing is offline and not semantic (default)",
    )
    ingest.add_argument("--model", default=None, help="Model name for the selected embedder")
    ingest.add_argument(
        "--dimension",
        type=int,
        default=None,
        help="Vector size (hashing default 256; optional for OpenAI embeddings)",
    )
    ingest.add_argument(
        "--base-url",
        default=None,
        help="OpenAI-compatible base URL (default: env OPENAI_BASE_URL or the official API)",
    )
    ingest.set_defaults(func=_cmd_ingest)

    query = commands.add_parser("query", help="Search an index and optionally generate an answer")
    query.add_argument("question")
    query.add_argument("--index", type=Path, default=Path(".persian-rag"))
    query.add_argument("--k", type=int, default=5)
    query.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid")
    query.add_argument("--llm", choices=("none", "openai"), default="none")
    query.add_argument("--llm-model", default=_DEFAULT_OPENAI_CHAT_MODEL)
    query.add_argument("--base-url", default=None)
    query.add_argument("--prompt-lang", choices=("fa", "en"), default="fa")
    query.add_argument("--json", action="store_true", help="Print the result as JSON")
    query.add_argument("--show-prompt", action="store_true")
    query.set_defaults(func=_cmd_query)
    return parser


def _cmd_ingest(args: argparse.Namespace) -> int:
    documents = load_documents(args.path)
    normalize_config = _normalize_from_args(args)
    chunking = _chunking_from_args(args)
    embedder = _embedder_from_args(args)
    retriever = build_retriever(
        [document.text for document in documents],
        embedder,
        document_ids=[document.id for document in documents],
        metadatas=[document.metadata for document in documents],
        normalize_config=normalize_config,
        chunking=chunking,
    )
    if len(retriever) == 0:
        raise PersianRagError("No chunks were produced from the input")
    target = save_index(args.index, retriever, embedder.to_spec(), chunking)
    print(f"Indexed {len(documents)} file(s), {len(retriever)} chunk(s) -> {target}")
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    saved = load_index(args.index)
    spec = _spec_with_base_url(saved.embedder, args.base_url)
    embedder = create_embedder(spec)
    retriever = retriever_from_saved(saved, embedder)
    llm = None
    if args.llm == "openai":
        llm = make_openai_llm(
            model=_optional_str(args.llm_model) or _DEFAULT_OPENAI_CHAT_MODEL,
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            base_url=_base_url(args.base_url),
        )
    template = DEFAULT_PROMPT_FA if args.prompt_lang == "fa" else DEFAULT_PROMPT_EN
    pipeline = RAGPipeline(retriever, llm, prompt_template=template)
    answer = pipeline.query(str(args.question), k=int(args.k), mode=_mode_from_args(args))
    if bool(args.json):
        payload: dict[str, object] = {
            "question": answer.question,
            "answer": answer.answer,
            "contexts": [item.to_dict() for item in answer.contexts],
        }
        if bool(args.show_prompt):
            payload["prompt"] = answer.prompt
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if answer.answer:
        print(answer.answer)
    else:
        print("(no LLM configured; showing retrieved context)")
    print()
    if not answer.contexts:
        print("No chunks matched.")
    for index, item in enumerate(answer.contexts, start=1):
        print(f"[{index}] score={item.score:.6f} id={item.chunk_id}")
        print(item.text)
        print()
    if bool(args.show_prompt):
        print("--- prompt ---")
        print(answer.prompt)
    return 0


def _normalize_from_args(args: argparse.Namespace) -> NormalizeConfig:
    return NormalizeConfig(
        digits=_digits_from_args(args),
        zwnj=_zwnj_from_args(args),
        remove_diacritics=bool(args.remove_diacritics),
        correct_half_spaces=not bool(args.no_half_space),
        clean_punctuation=not bool(args.no_clean_punctuation),
    )


def _chunking_from_args(args: argparse.Namespace) -> ChunkingConfig:
    strategy = cast(
        ChunkStrategy,
        _choice(args.strategy, ("fixed", "sentence", "recursive"), "strategy"),
    )
    return ChunkingConfig(
        strategy=strategy,
        chunk_size=int(args.chunk_size),
        overlap=int(args.overlap),
        sentence_overlap=int(args.sentence_overlap),
    )


def _digits_from_args(args: argparse.Namespace) -> DigitStyle:
    digits = _choice(args.digits, ("persian", "ascii", "keep"), "digits")
    if digits == "ascii":
        return "ascii"
    if digits == "keep":
        return "keep"
    return "persian"


def _zwnj_from_args(args: argparse.Namespace) -> ZwnjMode:
    zwnj = _choice(args.zwnj, ("normalize", "remove", "space"), "zwnj")
    if zwnj == "remove":
        return "remove"
    if zwnj == "space":
        return "space"
    return "normalize"


def _mode_from_args(args: argparse.Namespace) -> RetrievalMode:
    mode = _choice(args.mode, ("hybrid", "dense", "sparse"), "mode")
    if mode == "dense":
        return "dense"
    if mode == "sparse":
        return "sparse"
    return "hybrid"


def _embedder_from_args(
    args: argparse.Namespace,
) -> HashingEmbedder | SentenceTransformerEmbedder | OpenAICompatibleEmbedder:
    kind = _choice(args.embedder, ("hashing", "sentence-transformers", "openai"), "embedder")
    model = _optional_str(args.model)
    dimension = args.dimension
    if dimension is not None and (not isinstance(dimension, int) or dimension < 1):
        raise ConfigError("--dimension must be a positive integer")
    if kind == "hashing":
        return HashingEmbedder(dimension=dimension or 256)
    if kind == "sentence-transformers":
        if model is None:
            return SentenceTransformerEmbedder()
        return SentenceTransformerEmbedder(model_name=model)
    return OpenAICompatibleEmbedder(
        model=model or _DEFAULT_OPENAI_EMBEDDING_MODEL,
        base_url=_base_url(args.base_url),
        dimensions=dimension if isinstance(dimension, int) else None,
    )


def _spec_with_base_url(spec: EmbedderSpec, base_url: object) -> EmbedderSpec:
    override = _optional_str(base_url)
    if spec.kind != "openai" or override is None:
        return spec
    return EmbedderSpec(
        kind=spec.kind,
        dimension=spec.dimension,
        model=spec.model,
        base_url=override,
        ngram_min=spec.ngram_min,
        ngram_max=spec.ngram_max,
        seed=spec.seed,
    )


def _base_url(value: object) -> str:
    return _optional_str(value) or os.environ.get("OPENAI_BASE_URL", _DEFAULT_BASE_URL)


def _optional_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _choice(value: object, options: tuple[str, ...], label: str) -> str:
    if isinstance(value, str) and value in options:
        return value
    joined = ", ".join(options)
    raise ConfigError(f"Invalid {label}: {value!r}. Expected one of: {joined}")


if __name__ == "__main__":
    raise SystemExit(main())
