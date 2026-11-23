"""Run all retrieval systems over all query conditions and store top-100 runs.

Usage:
  python run_retrieval.py <collection> sparse   # BM25 variants, rewrite, hashing
  python run_retrieval.py <collection> dense <model_key>
  python run_retrieval.py <collection> fusion   # RRF hybrids from cached runs
collections: gold | pquad | miracl (100k) | miracl30k (nested subset)

Author: Mohammad Naser Sadrmanesh
"""
import hashlib, pickle, sys, time
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
from common import (DATA, CACHE, read_jsonl, BM25, TOKENIZERS, tok_npr, norm_npr, topk, rrf)

DEPTH = 100


def load(coll):
    base = "miracl" if coll.startswith("miracl") else coll
    cfile = "gold_corpus_prep" if base == "gold" else f"{base}_corpus"
    docs = read_jsonl(DATA / f"{cfile}.jsonl")
    idx = np.arange(len(docs))
    if coll == "miracl30k":
        idx = np.load(DATA / "miracl_sub30k_idx.npy")
        docs = [docs[i] for i in idx]
    queries = read_jsonl(DATA / f"{base}_queries.jsonl")
    pert = read_jsonl(DATA / f"{base}_queries_perturbed.jsonl")
    return docs, idx, queries, pert


def run_path(coll, system):
    p = CACHE / "runs" / coll
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{system}.pkl"


def save_run(coll, system, run):
    pickle.dump(run, open(run_path(coll, system), "wb"))


def load_run(coll, system):
    return pickle.load(open(run_path(coll, system), "rb"))


def as_row(idx_arr, scores):
    keep = idx_arr[scores > 0]
    out = np.full(DEPTH, -1, dtype=np.int32)
    out[:len(keep)] = keep[:DEPTH]
    return out


# ------------------------------------------------------------------ sparse
def sparse(coll):
    docs, _, queries, pert = load(coll)
    texts = [d["text"] for d in docs]
    from persian_rag.rewrite import QueryRewriter
    rewriter = QueryRewriter()
    for name, tok in TOKENIZERS.items():
        if run_path(coll, name).exists() and (name != "bm25_npr" or run_path(coll, "bm25_npr_rewrite").exists()):
            print(coll, name, "cached", flush=True)
            continue
        t0 = time.time()
        index = BM25([tok(t) for t in texts], k1=1.5, b=0.75)
        print(coll, name, f"indexed {len(texts)} in {time.time()-t0:.0f}s", flush=True)
        run = {}
        for r in pert:
            i, s = topk(index.scores(tok(r["text"])), DEPTH)
            run[(r["qid"], r["cond"], r["rate"])] = as_row(i, s)
        save_run(coll, name, run)
        if name == "bm25_npr":
            # rule-based query rewriting (persian-rag QueryRewriter), fused with RRF:
            # original weight 2.0, variants share weight 1.0 (RewritingRetriever defaults)
            run = {}
            for r in pert:
                vs = rewriter.variants(r["text"]) or [r["text"]]
                w = [2.0] + ([1.0 / (len(vs) - 1)] * (len(vs) - 1) if len(vs) > 1 else [])
                agg = defaultdict(float)
                for v, wt in zip(vs, w):
                    i, s = topk(index.scores(tok_npr(v)), DEPTH)
                    for rank, d in enumerate(i[s > 0]):
                        agg[int(d)] += wt / (60 + rank + 1)
                ranked = sorted(agg, key=lambda d: (-agg[d], d))[:DEPTH]
                row = np.full(DEPTH, -1, dtype=np.int32); row[:len(ranked)] = ranked
                run[(r["qid"], r["cond"], r["rate"])] = row
            save_run(coll, "bm25_npr_rewrite", run)
        print(coll, name, f"done in {time.time()-t0:.0f}s", flush=True)
    hashing(coll, texts, pert)


# ------------------------------------------------------------------ hashing embedder
def hashing(coll, texts, pert, dim=512):
    if run_path(coll, "hashing").exists():
        return
    from persian_rag.embeddings import HashingEmbedder
    he = HashingEmbedder(dimension=dim, ngram_range=(2, 4))
    memo = {}

    def vec_sparse(text):
        # identical to HashingEmbedder._vector, but memoises the bucket hash per n-gram
        padded = f" {text} "
        cols, vals = [], []
        for width in range(2, 5):
            for o in range(len(padded) - width + 1):
                g = padded[o:o + width]
                b = memo.get(g)
                if b is None:
                    b = memo[g] = he._bucket(g)
                cols.append(b[0]); vals.append(b[1])
        v = np.zeros(dim, np.float32)
        np.add.at(v, cols, vals)
        n = np.linalg.norm(v)
        return v / n if n else v

    t0 = time.time()
    D = np.stack([vec_sparse(norm_npr(t)) for t in texts])
    # sanity check against the library implementation
    ref = np.array(he.embed_query(norm_npr(texts[0])), np.float32)
    assert np.allclose(ref, D[0], atol=1e-5), "hashing mismatch"
    print(coll, "hashing indexed", f"{time.time()-t0:.0f}s", flush=True)
    run = {}
    for r in pert:
        s = D @ vec_sparse(norm_npr(r["text"]))
        i, sc = topk(s, DEPTH)
        row = i.astype(np.int32)
        run[(r["qid"], r["cond"], r["rate"])] = row
    save_run(coll, "hashing", run)


# ------------------------------------------------------------------ dense
def load_emb(coll, key):
    from pathlib import Path
    if coll == "miracl30k":
        d30 = CACHE / "emb" / f"miracl_sub30k_{key}"
        if (d30 / "idx.npy").exists():
            d = d30
        else:  # slice the 100k matrix
            E = load_emb("miracl", key)
            return E[np.load(DATA / "miracl_sub30k_idx.npy")]
    else:
        d = CACHE / "emb" / f"{coll}_{key}"
    chunks = sorted(d.glob("chunk_*.npy"))
    return np.concatenate([np.load(c) for c in chunks]).astype(np.float32)


def dense(coll, key, qnorm=False):
    from encode_corpus import load_model
    docs, _, queries, pert = load(coll)
    E = load_emb(coll, key)
    assert E.shape[0] == len(docs), (E.shape, len(docs))
    if coll == "pquad":
        # CPU budget: dense models on PQuAD use the main + attested-variant
        # conditions only (no dose-response grid)
        pert = [r for r in pert if r["rate"] == 0.15 or r["cond"].startswith("real")]
    m, qp, _ = load_model(key)
    qcache_f = CACHE / "qemb" / f"{key}.pkl"
    qcache_f.parent.mkdir(parents=True, exist_ok=True)
    qcache = pickle.load(open(qcache_f, "rb")) if qcache_f.exists() else {}
    full = pert
    for qn in ([False, True] if qnorm else [False]):
        # N_PR query normalisation is evaluated on the main + attested conditions only
        pert = [r for r in full if r["rate"] == 0.15 or r["cond"].startswith("real")] if qn else full
        texts = [qp + (norm_npr(r["text"]) if qn else r["text"]) for r in pert]
        todo = sorted({t for t in texts if t not in qcache})
        if todo:
            t0 = time.time()
            embs = m.encode(todo, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            qcache.update(zip(todo, embs.astype(np.float32)))
            pickle.dump(qcache, open(qcache_f, "wb"))
            print(coll, key, f"encoded {len(todo)} queries in {time.time()-t0:.0f}s", flush=True)
        Q = np.stack([qcache[t] for t in texts])
        run = {}
        for b0 in range(0, len(Q), 256):
            S = Q[b0:b0 + 256] @ E.T
            for j, s in enumerate(S):
                i, _ = topk(s, DEPTH)
                r = pert[b0 + j]
                run[(r["qid"], r["cond"], r["rate"])] = i.astype(np.int32)
        save_run(coll, key + ("_qnorm" if qn else ""), run)
        print(coll, key + ("_qnorm" if qn else ""), "run saved", flush=True)


# ------------------------------------------------------------------ fusion
def fusion(coll):
    pairs = {
        "hybrid_bm25npr_hashing": ("bm25_npr", "hashing"),
        "hybrid_bm25npr_e5small": ("bm25_npr", "e5_small"),
        "hybrid_bm25npr_e5base": ("bm25_npr", "e5_base"),
        "hybrid_bm25npr_bgem3": ("bm25_npr", "bge_m3"),
        "hybrid_bm25npr_minilm": ("bm25_npr", "minilm"),
    }
    for name, (a, b) in pairs.items():
        if not (run_path(coll, a).exists() and run_path(coll, b).exists()):
            print("skip", name); continue
        A, B = load_run(coll, a), load_run(coll, b)
        run = {}
        for key in A:
            if key not in B:
                continue
            fused = rrf([A[key][A[key] >= 0], B[key][B[key] >= 0]], k=60, depth=DEPTH)[:DEPTH]
            row = np.full(DEPTH, -1, dtype=np.int32); row[:len(fused)] = fused
            run[key] = row
        save_run(coll, name, run)
        print(coll, name, "fused", flush=True)


if __name__ == "__main__":
    coll, stage = sys.argv[1], sys.argv[2]
    if stage == "sparse":
        sparse(coll)
    elif stage == "dense":
        dense(coll, sys.argv[3], qnorm=True)
    elif stage == "fusion":
        fusion(coll)
