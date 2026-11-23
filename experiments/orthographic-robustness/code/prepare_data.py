"""Build the evaluation collections.

MIRACL-fa-100k : 632 MIRACL fa dev queries; corpus = every passage judged in the
                 dev qrels (positive or negative) + uniformly sampled passages from
                 the full 2.2M fa corpus (seed 42) up to N_TOTAL passages.
PQuAD-val      : all 1,103 PQuAD validation paragraphs; all answerable questions;
                 relevant = the paragraph the question was written on.
Gold           : persian-rag-bench (34 passages, 42 queries).

Author: Mohammad Naser Sadrmanesh
"""
import gzip, json, random, sys
import pandas as pd
from common import DATA, SEED, write_jsonl, read_jsonl

N_TOTAL = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000


def miracl():
    topics = {}
    for line in open(DATA / "topics.miracl-v1.0-fa-dev.tsv", encoding="utf-8"):
        qid, q = line.rstrip("\n").split("\t")
        topics[qid] = q
    qrels, judged = {}, set()
    for line in open(DATA / "qrels.miracl-v1.0-fa-dev.tsv"):
        qid, _, did, r = line.split()
        judged.add(did)
        if int(r) > 0:
            qrels.setdefault(qid, []).append(did)
    rng = random.Random(SEED)
    keep, pool_ids = {}, []
    total = 0
    for i in range(5):
        with gzip.open(DATA / f"fa-docs-{i}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                total += 1
                if d["docid"] in judged:
                    keep[d["docid"]] = d
                else:
                    pool_ids.append(d["docid"])
    need = N_TOTAL - len(keep)
    sample = set(rng.sample(pool_ids, need))
    for i in range(5):
        with gzip.open(DATA / f"fa-docs-{i}.jsonl.gz", "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if d["docid"] in sample:
                    keep[d["docid"]] = d
    docs = [{"id": d["docid"], "text": (d["title"] + " " + d["text"]).strip()} for d in keep.values()]
    docs.sort(key=lambda x: x["id"])
    queries = [{"id": q, "text": topics[q], "rel": qrels[q]} for q in topics if q in qrels]
    write_jsonl(DATA / "miracl_corpus.jsonl", docs)
    write_jsonl(DATA / "miracl_queries.jsonl", queries)
    print(f"MIRACL full corpus={total} judged={len(judged)} kept={len(docs)} queries={len(queries)}")


def pquad():
    df = pd.read_parquet(DATA / "pquad_validation.parquet")
    docs, queries = [], []
    for ai, art in enumerate(df.itertuples()):
        for pi, p in enumerate(art.paragraphs):
            did = f"p{ai}_{pi}"
            docs.append({"id": did, "text": p["context"]})
            for qa in p["qas"]:
                if not qa["is_impossible"]:
                    queries.append({"id": str(qa["id"]), "text": qa["question"], "rel": [did], "cluster": did})
    write_jsonl(DATA / "pquad_corpus.jsonl", docs)
    write_jsonl(DATA / "pquad_queries.jsonl", queries)
    print(f"PQuAD paragraphs={len(docs)} questions={len(queries)}")


def gold():
    docs = [{"id": d["id"], "text": (d.get("title", "") + " " + d["text"]).strip()} for d in read_jsonl(DATA / "gold_corpus.jsonl")]
    queries = [{"id": q["id"], "text": q["question"], "rel": q["relevant_ids"], "tags": q["tags"]} for q in read_jsonl(DATA / "gold_qa.jsonl")]
    write_jsonl(DATA / "gold_corpus_prep.jsonl", docs)
    write_jsonl(DATA / "gold_queries.jsonl", queries)
    print(f"Gold passages={len(docs)} queries={len(queries)} spelling={sum('spelling' in q['tags'] for q in queries)}")


if __name__ == "__main__":
    gold(); pquad(); miracl()
