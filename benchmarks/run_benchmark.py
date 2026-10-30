"""Retrieval benchmark on a small Persian QA set.

Runs offline by default with the hashing embedder, so the numbers are
reproducible on any machine:

    python benchmarks/run_benchmark.py
    python benchmarks/run_benchmark.py --json results.json
    python benchmarks/run_benchmark.py --embedder sentence-transformers \\
        --model intfloat/multilingual-e5-small

Each configuration is a retrieval mode (sparse, dense, hybrid) with or
without the rule-based query rewriter. Latency is the wall-clock time of
one ``retrieve`` call, so it depends on the machine.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persian_rag import (
    ChunkingConfig,
    EvalExample,
    HashingEmbedder,
    QueryRewriter,
    RewritingRetriever,
    __version__,
    build_retriever,
    evaluate_retrieval,
)
from persian_rag.embeddings import Embedder
from persian_rag.retrieval import RetrievalMode

DATA = Path(__file__).resolve().parent / "data"
MODES: tuple[RetrievalMode, ...] = ("sparse", "dense", "hybrid")


@dataclass
class Row:
    name: str
    mode: str
    expansion: bool
    hit1: float
    hit5: float
    mrr10: float
    ndcg10: float
    spelling_hit1: float
    p50_ms: float
    p95_ms: float

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def make_embedder(args: argparse.Namespace) -> Embedder:
    if args.embedder == "hashing":
        return HashingEmbedder(dimension=args.dimension)
    from persian_rag.embeddings import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(args.model)


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))
    return ordered[index]


def run(args: argparse.Namespace) -> list[Row]:
    corpus = load_jsonl(DATA / "corpus.jsonl")
    raw_questions = load_jsonl(DATA / "qa.jsonl")
    examples = [EvalExample.from_dict(item) for item in raw_questions]
    spelling = [
        example
        for example, item in zip(examples, raw_questions, strict=True)
        if "spelling" in item.get("tags", [])
    ]
    retriever = build_retriever(
        [f"{doc['title']}\n{doc['text']}" for doc in corpus],
        make_embedder(args),
        document_ids=[doc["id"] for doc in corpus],
        chunking=ChunkingConfig(chunk_size=400, overlap=0),
    )
    expanded = RewritingRetriever(retriever, QueryRewriter())
    rows: list[Row] = []
    for expansion in (False, True):
        target: Any = expanded if expansion else retriever
        for mode in MODES:
            report = evaluate_retrieval(target, examples, k_values=(1, 5, 10), mode=mode)
            spelling_report = evaluate_retrieval(target, spelling, k_values=(1,), mode=mode)
            timings: list[float] = []
            for _ in range(args.repeat):
                for example in examples:
                    start = time.perf_counter()
                    target.retrieve(example.question, k=10, mode=mode)
                    timings.append((time.perf_counter() - start) * 1000)
            rows.append(
                Row(
                    name=f"{mode}{' + rewrite' if expansion else ''}",
                    mode=mode,
                    expansion=expansion,
                    hit1=report.metrics[1]["hit_rate"],
                    hit5=report.metrics[5]["hit_rate"],
                    mrr10=report.metrics[10]["mrr"],
                    ndcg10=report.metrics[10]["ndcg"],
                    spelling_hit1=spelling_report.metrics[1]["hit_rate"],
                    p50_ms=statistics.median(timings),
                    p95_ms=percentile(timings, 0.95),
                )
            )
    return rows


def markdown_table(rows: Sequence[Row]) -> str:
    lines = [
        "| Configuration | Hit@1 | Hit@5 | MRR@10 | nDCG@10 | Hit@1 (spelling) | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row.name} | {row.hit1:.3f} | {row.hit5:.3f} | {row.mrr10:.3f} | "
            f"{row.ndcg10:.3f} | {row.spelling_hit1:.3f} | {row.p50_ms:.2f} | {row.p95_ms:.2f} |"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--embedder", choices=["hashing", "sentence-transformers"], default="hashing"
    )
    parser.add_argument("--model", default="intfloat/multilingual-e5-small")
    parser.add_argument("--dimension", type=int, default=512)
    parser.add_argument("--repeat", type=int, default=5, help="Timing rounds per configuration")
    parser.add_argument("--json", type=Path, help="Also write the results to this JSON file")
    args = parser.parse_args(argv)

    rows = run(args)
    print(markdown_table(rows))
    if args.json:
        payload = {
            "persian_rag": __version__,
            "python": platform.python_version(),
            "embedder": args.embedder if args.embedder == "hashing" else args.model,
            "rows": [row.to_dict() for row in rows],
        }
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
