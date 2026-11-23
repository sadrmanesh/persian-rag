"""Shared utilities: normalizers, tokenizers, BM25, metrics, fusion.

Author: Mohammad Naser Sadrmanesh
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
CACHE = ROOT / "cache_store"   # embeddings / runs (rebuildable)
FIGS = ROOT / "figures"
SEED = 42


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- normalizers
_WS_RE = re.compile(r"\S+")
_PUNCT_STRIP = "؟?!.,،؛;:«»\"'()[]{}<>-–—…/\\|"


def tok_whitespace(text: str) -> list[str]:
    """Raw whitespace split; only strips surrounding punctuation. No normalization."""
    out = []
    for t in text.split():
        t = t.strip(_PUNCT_STRIP)
        if t:
            out.append(t.lower())
    return out


_hazm_norm = None
_hazm_tok = None


def tok_hazm(text: str) -> list[str]:
    global _hazm_norm, _hazm_tok
    if _hazm_norm is None:
        import hazm
        _hazm_norm = hazm.Normalizer()  # default config: unicodes_replacement & persian_style on
        _hazm_tok = hazm.WordTokenizer()
    t = _hazm_norm.normalize(text)
    return [w.lower() for w in _hazm_tok.tokenize(t) if w.strip(_PUNCT_STRIP)]


_pv_norm = None
_pv_tok = None


def tok_parsivar(text: str) -> list[str]:
    global _pv_norm, _pv_tok
    if _pv_norm is None:
        import parsivar
        _pv_norm = parsivar.Normalizer()
        _pv_tok = parsivar.Tokenizer()
    t = _pv_norm.normalize(text)
    return [w.lower() for w in _pv_tok.tokenize_words(t) if w.strip(_PUNCT_STRIP)]


def tok_npr(text: str) -> list[str]:
    from persian_rag.tokenize import tokenize
    return tokenize(text)  # persian-rag default NormalizeConfig


def norm_npr(text: str) -> str:
    from persian_rag.normalize import normalize
    return normalize(text)


TOKENIZERS = {
    "bm25_ws": tok_whitespace,
    "bm25_hazm": tok_hazm,
    "bm25_parsivar": tok_parsivar,
    "bm25_npr": tok_npr,
}


# ---------------------------------------------------------------- BM25
class BM25:
    """Okapi BM25 (Robertson/Lucene idf variant) over a scipy sparse matrix."""

    def __init__(self, docs_tokens, k1=1.5, b=0.75):
        self.vocab = {}
        rows, cols, vals = [], [], []
        dl = np.zeros(len(docs_tokens), dtype=np.float32)
        for i, toks in enumerate(docs_tokens):
            c = Counter(toks)
            dl[i] = len(toks)
            for w, n in c.items():
                j = self.vocab.setdefault(w, len(self.vocab))
                rows.append(i); cols.append(j); vals.append(n)
        N = len(docs_tokens)
        tf = sp.csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(N, len(self.vocab)))
        df = np.bincount(tf.indices, minlength=len(self.vocab)).astype(np.float32)
        self.idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        avgdl = dl.mean()
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = tf.data * (k1 + 1) / denom
        self.W = sp.csc_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape)
        self.N = N

    def scores(self, q_tokens):
        s = np.zeros(self.N, dtype=np.float32)
        for w, n in Counter(q_tokens).items():
            j = self.vocab.get(w)
            if j is None:
                continue
            a, bnd = self.W.indptr[j], self.W.indptr[j + 1]
            s[self.W.indices[a:bnd]] += n * self.idf[j] * self.W.data[a:bnd]
        return s


def topk(scores, k):
    k = min(k, len(scores))
    idx = np.argpartition(-scores, k - 1)[:k]
    idx = idx[np.argsort(-scores[idx], kind="stable")]
    # drop zero-score docs for sparse runs
    return idx, scores[idx]


def rrf(rank_lists, k=60, depth=100):
    """rank_lists: list of arrays of doc indices (best first)."""
    agg = {}
    for lst in rank_lists:
        for r, d in enumerate(lst[:depth]):
            agg[int(d)] = agg.get(int(d), 0.0) + 1.0 / (k + r + 1)
    items = sorted(agg.items(), key=lambda x: -x[1])
    return np.array([d for d, _ in items], dtype=np.int64)


# ---------------------------------------------------------------- metrics
def metrics_for(ranked, rel: set, k_ndcg=10, k_rec=100):
    ranked = list(ranked)
    gains = [1.0 if d in rel else 0.0 for d in ranked[:k_ndcg]]
    dcg = sum(g / math.log2(i + 2) for i, g in enumerate(gains))
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k_ndcg)))
    ndcg = dcg / ideal if ideal else 0.0
    rr = 0.0
    for i, d in enumerate(ranked[:10]):
        if d in rel:
            rr = 1.0 / (i + 1); break
    hit1 = 1.0 if ranked and ranked[0] in rel else 0.0
    rec = len(set(ranked[:k_rec]) & rel) / len(rel) if rel else 0.0
    return {"ndcg10": ndcg, "mrr10": rr, "hit1": hit1, "recall100": rec}
