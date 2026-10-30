"""Command line interface: ingest, query, chat, eval, and serve."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

from persian_rag.chunk import ChunkingConfig, ChunkStrategy
from persian_rag.documents import load_documents
from persian_rag.embeddings import (
    Embedder,
    EmbedderSpec,
    HashingEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from persian_rag.errors import ConfigError, MissingDependencyError, PersianRagError
from persian_rag.incremental import IncrementalIndexer, IndexManifest
from persian_rag.normalize import DigitStyle, NormalizeConfig, ZwnjMode
from persian_rag.persist import load_index, resolve_index_path, retriever_from_saved, save_index
from persian_rag.pipeline import DEFAULT_PROMPT_EN, DEFAULT_PROMPT_FA, RAGAnswer, RAGPipeline
from persian_rag.rerank import CrossEncoderReranker, LLMReranker, Reranker
from persian_rag.retrieval import HybridRetriever, RetrievalMode
from persian_rag.version import __version__

if TYPE_CHECKING:
    from persian_rag.chat import CondenseMode
    from persian_rag.documents import SourceDocument
    from persian_rag.llm import OllamaLLM, OpenAICompatibleLLM
    from persian_rag.loaders import RtlFix

_DEFAULT_OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
_DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o-mini"
_DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
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

    ingest = commands.add_parser("ingest", help="Index a file or a directory of documents")
    ingest.add_argument(
        "path",
        nargs="+",
        help="Files, directories (.txt, .md, .html, .pdf, .docx), or http(s) URLs",
    )
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
    ingest.add_argument(
        "--rtl-fix",
        choices=("auto", "reverse", "none"),
        default="auto",
        help="Repair PDF lines stored in visual order (default: auto-detect)",
    )
    ingest.add_argument(
        "--split-pages", action="store_true", help="Index each PDF page as its own document"
    )
    ingest.add_argument(
        "--split-sections",
        action="store_true",
        help="Index each Markdown section as its own document",
    )
    ingest.add_argument(
        "--html-extract",
        choices=("main", "all"),
        default="main",
        help="Keep only the main content of HTML pages (default) or all text",
    )
    ingest.add_argument(
        "--update",
        action="store_true",
        help="Update an existing index: re-embed only new or changed documents",
    )
    ingest.add_argument(
        "--keep-missing",
        action="store_true",
        help="With --update, keep documents that are no longer in the input",
    )
    ingest.add_argument(
        "--embedding-cache",
        type=Path,
        default=None,
        help="SQLite file that caches embeddings between runs",
    )
    ingest.set_defaults(func=_cmd_ingest)

    query = commands.add_parser("query", help="Search an index and optionally generate an answer")
    query.add_argument("question")
    _add_pipeline_arguments(query)
    query.add_argument("--k", type=int, default=5)
    query.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid")
    query.add_argument("--json", action="store_true", help="Print the result as JSON")
    query.add_argument("--show-prompt", action="store_true")
    query.set_defaults(func=_cmd_query)

    chat = commands.add_parser("chat", help="Ask follow-up questions in an interactive session")
    _add_pipeline_arguments(chat)
    chat.add_argument("--k", type=int, default=5)
    chat.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid")
    chat.add_argument("--max-turns", type=int, default=6, help="Turns kept in memory")
    chat.add_argument(
        "--condense",
        choices=("auto", "llm", "heuristic", "none"),
        default="auto",
        help="How follow-ups become standalone search queries",
    )
    chat.set_defaults(func=_cmd_chat)

    evaluate = commands.add_parser("eval", help="Measure retrieval and answer quality")
    evaluate.add_argument("dataset", type=Path, help="JSONL file of labelled questions")
    _add_pipeline_arguments(evaluate)
    evaluate.add_argument(
        "--k", type=int, nargs="+", default=[1, 3, 5, 10], help="Cut-offs (default: 1 3 5 10)"
    )
    evaluate.add_argument("--mode", choices=("hybrid", "dense", "sparse"), default="hybrid")
    evaluate.add_argument(
        "--match",
        choices=("document", "chunk"),
        default="document",
        help="Compare relevant_ids with document ids or chunk ids",
    )
    evaluate.add_argument(
        "--answers",
        action="store_true",
        help="Also generate answers and score faithfulness (needs --llm)",
    )
    evaluate.add_argument(
        "--judge",
        choices=("overlap", "llm"),
        default="overlap",
        help="Faithfulness scorer for --answers (default: token overlap)",
    )
    evaluate.add_argument("--json", action="store_true", help="Print the report as JSON")
    evaluate.set_defaults(func=_cmd_eval)

    serve = commands.add_parser("serve", help="Serve an index over HTTP (needs [server])")
    _add_pipeline_arguments(serve)
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=_cmd_serve)
    return parser


def _add_pipeline_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--index", type=Path, default=Path(".persian-rag"))
    parser.add_argument("--llm", choices=("none", "openai", "ollama"), default="none")
    parser.add_argument(
        "--llm-model",
        default=None,
        help=f"Chat model (default: {_DEFAULT_OPENAI_CHAT_MODEL} or {_DEFAULT_OLLAMA_MODEL})",
    )
    parser.add_argument("--llm-max-tokens", type=int, default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--ollama-url",
        default=None,
        help="Ollama server (default: env OLLAMA_HOST or http://localhost:11434)",
    )
    parser.add_argument("--prompt-lang", choices=("fa", "en"), default="fa")
    parser.add_argument(
        "--citations",
        action="store_true",
        help="Ask the model to cite sources and resolve [n] markers to spans",
    )
    parser.add_argument(
        "--expand",
        choices=("none", "rules", "llm"),
        default="none",
        help="Search spelling variants of the query; llm also asks the model for rewrites",
    )
    parser.add_argument(
        "--rerank",
        choices=("none", "cross-encoder", "llm"),
        default="none",
        help="Rerank retrieved chunks (cross-encoder needs [rerank]; llm needs --llm)",
    )
    parser.add_argument("--rerank-model", default=None, help="Cross-encoder model name")
    parser.add_argument("--rerank-candidates", type=int, default=20)


def _load_inputs(args: argparse.Namespace) -> list[SourceDocument]:
    from persian_rag.web import load_url

    rtl_fix = cast("RtlFix", _choice(args.rtl_fix, ("auto", "reverse", "none"), "rtl-fix"))
    html_extract: Literal["main", "all"] = (
        "all" if _choice(args.html_extract, ("main", "all"), "html-extract") == "all" else "main"
    )
    documents: list[SourceDocument] = []
    for raw in args.path:
        source = str(raw)
        if source.startswith(("http://", "https://")):
            documents.extend(load_url(source, extract=html_extract))
            continue
        documents.extend(
            load_documents(
                Path(source),
                rtl_fix=rtl_fix,
                split_pages=bool(args.split_pages),
                split_sections=bool(args.split_sections),
                html_extract=html_extract,
            )
        )
    return documents


def _cmd_ingest(args: argparse.Namespace) -> int:
    documents = _load_inputs(args)
    target = resolve_index_path(args.index)
    with ExitStack() as stack:

        def cached(model: Embedder) -> Embedder:
            if args.embedding_cache is None:
                return model
            from persian_rag.cache import CachedEmbedder, SQLiteEmbeddingCache

            cache = stack.enter_context(SQLiteEmbeddingCache(args.embedding_cache))
            return CachedEmbedder(model, cache)

        if bool(args.update) and target.is_file():
            saved = load_index(target)
            spec = _spec_with_base_url(saved.embedder, args.base_url)
            retriever = retriever_from_saved(saved, cached(create_embedder(spec)))
            manifest = saved.manifest() if saved.documents else None
            indexer = IncrementalIndexer(retriever, chunking=saved.chunking, manifest=manifest)
            report = indexer.sync(documents, delete_missing=not bool(args.keep_missing))
            embedder_spec = saved.embedder
            chunking = saved.chunking
        else:
            chunking = _chunking_from_args(args)
            embedder = _embedder_from_args(args)
            retriever = HybridRetriever(
                cached(embedder), normalize_config=_normalize_from_args(args)
            )
            indexer = IncrementalIndexer(retriever, chunking=chunking, manifest=IndexManifest())
            report = indexer.sync(documents)
            embedder_spec = embedder.to_spec()
    if len(retriever) == 0:
        raise PersianRagError("No chunks were produced from the input")
    written = save_index(target, retriever, embedder_spec, chunking, manifest=indexer.manifest)
    if bool(args.update):
        print(f"Updated index: {report.summary()}")
    print(f"Indexed {len(indexer.manifest)} document(s), {len(retriever)} chunk(s) -> {written}")
    return 0


def _build_pipeline(args: argparse.Namespace) -> RAGPipeline:
    saved = load_index(args.index)
    spec = _spec_with_base_url(saved.embedder, args.base_url)
    embedder = create_embedder(spec)
    retriever = retriever_from_saved(saved, embedder)
    llm = _llm_from_args(args)
    reranker: Reranker | None = None
    rerank = _choice(args.rerank, ("none", "cross-encoder", "llm"), "rerank")
    if rerank == "cross-encoder":
        model = _optional_str(args.rerank_model)
        reranker = CrossEncoderReranker(model) if model else CrossEncoderReranker()
    elif rerank == "llm":
        if llm is None:
            raise ConfigError("--rerank llm needs --llm openai or --llm ollama")
        reranker = LLMReranker(llm)
    citations = bool(args.citations)
    if citations:
        from persian_rag.citations import CITATION_PROMPT_EN, CITATION_PROMPT_FA

        template = CITATION_PROMPT_FA if args.prompt_lang == "fa" else CITATION_PROMPT_EN
    else:
        template = DEFAULT_PROMPT_FA if args.prompt_lang == "fa" else DEFAULT_PROMPT_EN
    rewriter = None
    expand = _choice(args.expand, ("none", "rules", "llm"), "expand")
    if expand != "none":
        from persian_rag.rewrite import QueryRewriter

        if expand == "llm" and llm is None:
            raise ConfigError("--expand llm needs --llm openai or --llm ollama")
        rewriter = QueryRewriter(llm=llm if expand == "llm" else None)
    return RAGPipeline(
        retriever,
        llm,
        prompt_template=template,
        reranker=reranker,
        rerank_candidates=int(args.rerank_candidates),
        citations=citations,
        query_rewriter=rewriter,
    )


def _llm_from_args(args: argparse.Namespace) -> OpenAICompatibleLLM | OllamaLLM | None:
    from persian_rag.llm import create_llm

    provider = _choice(args.llm, ("none", "openai", "ollama"), "llm")
    max_tokens = args.llm_max_tokens
    if max_tokens is not None and (not isinstance(max_tokens, int) or max_tokens < 1):
        raise ConfigError("--llm-max-tokens must be a positive integer")
    model = _optional_str(args.llm_model)
    if provider == "openai":
        return create_llm(
            "openai",
            model=model or _DEFAULT_OPENAI_CHAT_MODEL,
            base_url=_base_url(args.base_url),
            api_key=os.environ.get("OPENAI_API_KEY", ""),
            max_tokens=max_tokens,
        )
    if provider == "ollama":
        return create_llm(
            "ollama",
            model=model or _DEFAULT_OLLAMA_MODEL,
            base_url=_optional_str(args.ollama_url),
            max_tokens=max_tokens,
        )
    return None


def _cmd_query(args: argparse.Namespace) -> int:
    pipeline = _build_pipeline(args)
    answer = pipeline.query(str(args.question), k=int(args.k), mode=_mode_from_args(args))
    if bool(args.json):
        payload = answer.to_dict(include_prompt=bool(args.show_prompt))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    if answer.answer:
        print(answer.answer)
        _print_citations(answer)
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


def _print_citations(answer: RAGAnswer) -> None:
    if not answer.citations and not answer.uncited:
        return
    from persian_rag.citations import CitationReport, format_sources

    report = CitationReport(answer.citations, answer.uncited, ())
    print()
    print("Sources:")
    print(format_sources(report) or "(none)")
    for sentence in answer.uncited:
        print(f"! no citation: {sentence}")


def _cmd_chat(args: argparse.Namespace) -> int:
    from persian_rag.chat import ChatSession, ConversationMemory

    pipeline = _build_pipeline(args)
    condense = _choice(args.condense, ("auto", "llm", "heuristic", "none"), "condense")
    session = ChatSession(
        pipeline,
        memory=ConversationMemory(int(args.max_turns)),
        condense=cast("CondenseMode", condense),
    )
    print("persian-rag chat. Type /reset to start over, /exit to quit.")
    if pipeline.llm is None:
        print("(no LLM configured; showing retrieved context)")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            print()
            return 0
        if not line:
            continue
        if line in {"/exit", "/quit"}:
            return 0
        if line == "/reset":
            session.reset()
            print("(conversation cleared)")
            continue
        reply = session.ask(line, k=int(args.k), mode=_mode_from_args(args))
        if reply.standalone_question != line:
            print(f"(searching: {reply.standalone_question})")
        answer = reply.answer
        if answer.answer:
            print(answer.answer)
            _print_citations(answer)
        else:
            for index, item in enumerate(answer.contexts, start=1):
                print(f"[{index}] {item.document_id}: {' '.join(item.text.split())[:200]}")
        print()


def _cmd_eval(args: argparse.Namespace) -> int:
    from persian_rag.evaluation import (
        FaithfulnessScorer,
        LLMFaithfulnessJudge,
        MatchOn,
        TokenOverlapFaithfulness,
        evaluate_answers,
        evaluate_retrieval,
        load_eval_examples,
    )

    examples = load_eval_examples(args.dataset)
    pipeline = _build_pipeline(args)
    match: MatchOn = (
        "chunk" if _choice(args.match, ("document", "chunk"), "match") == "chunk" else "document"
    )
    report = evaluate_retrieval(
        pipeline,
        examples,
        k_values=[int(value) for value in args.k],
        mode=_mode_from_args(args),
        match=match,
    )
    answers = None
    if bool(args.answers):
        if pipeline.llm is None:
            raise ConfigError("--answers needs --llm openai or --llm ollama")
        scorer: FaithfulnessScorer = TokenOverlapFaithfulness()
        if args.judge == "llm":
            llm = pipeline.llm
            scorer = LLMFaithfulnessJudge(lambda prompt: str(llm(prompt)))
        answers = evaluate_answers(
            pipeline, examples, scorer=scorer, k=max(args.k), mode=_mode_from_args(args)
        )
    if bool(args.json):
        payload: dict[str, object] = {"retrieval": report.to_dict(include_queries=True)}
        if answers is not None:
            payload["answers"] = answers.to_dict(include_answers=True)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0
    print(f"Questions: {report.num_queries} (match on {report.match})")
    print(report.format_table())
    if answers is not None:
        print()
        print(f"Mean faithfulness: {_format_optional(answers.mean_faithfulness)}")
        print(f"Mean reference F1: {_format_optional(answers.mean_reference_f1)}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn
    except ImportError as exc:
        raise MissingDependencyError(
            "uvicorn is not installed. Install it with: pip install 'persian-rag[server]'"
        ) from exc
    from persian_rag.server import create_app

    app = create_app(_build_pipeline(args))
    uvicorn.run(app, host=str(args.host), port=int(args.port))
    return 0


def _format_optional(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


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
