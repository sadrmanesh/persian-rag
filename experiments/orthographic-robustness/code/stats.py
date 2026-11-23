"""Paired significance testing over per-query results.

For every comparison we report: mean paired difference, 95% bootstrap CI
(10,000 resamples; cluster bootstrap over PQuAD paragraphs, query-level
bootstrap elsewhere), two-sided paired randomization (sign-flip) test with
10,000 permutations (plus its asymptotic normal form), Wilcoxon signed-rank p (secondary), Cohen's d_z
(mean / SD of the paired differences) and Holm-adjusted p over the FULL family
of comparisons run on that collection.

Usage: python stats.py <collection> [metric]
Author: Mohammad Naser Sadrmanesh
"""
import sys
import numpy as np
import pandas as pd
from scipy import stats as st
from common import RESULTS, SEED

B = 10_000
MAIN = ["clean", "char", "zwnj", "typo", "comp", "real", "real_ortho", "real_arabic", "real_zwnj"]
GAIN_PAIRS = [
    ("bm25_npr", "bm25_ws"), ("bm25_npr", "bm25_hazm"), ("bm25_npr", "bm25_parsivar"),
    ("bm25_hazm", "bm25_ws"), ("bm25_parsivar", "bm25_ws"),
    ("bm25_npr_rewrite", "bm25_npr"), ("hashing", "bm25_npr"),
    ("hybrid_bm25npr_hashing", "bm25_npr"),
    ("e5_small", "bm25_npr"), ("e5_base", "bm25_npr"), ("bge_m3", "bm25_npr"), ("minilm", "bm25_npr"),
    ("e5_small_qnorm", "e5_small"), ("e5_base_qnorm", "e5_base"), ("bge_m3_qnorm", "bge_m3"),
    ("hybrid_bm25npr_e5small", "e5_small"), ("hybrid_bm25npr_e5small", "bm25_npr"),
    ("hybrid_bm25npr_e5base", "e5_base"), ("hybrid_bm25npr_e5base", "bm25_npr"),
    ("hybrid_bm25npr_bgem3", "bge_m3"), ("hybrid_bm25npr_bgem3", "bm25_npr"),
    ("bge_m3", "e5_base"), ("e5_base", "e5_small"),
]


def sign_flip_p(d, rng):
    d = np.asarray(d, float)
    obs = abs(d.mean())
    if np.allclose(d, 0):
        return 1.0
    cnt = 0
    for b0 in range(0, B, 1000):
        s = rng.choice([-1.0, 1.0], size=(1000, len(d)))
        cnt += (np.abs((s * d).mean(1)) >= obs - 1e-12).sum()
    return (cnt + 1) / (B + 1)


def boot_ci(d, clusters, rng):
    d = np.asarray(d, float)
    if clusters is None:
        idx = rng.integers(0, len(d), size=(B, len(d)))
        means = d[idx].mean(1)
    else:
        # cluster bootstrap: resample clusters, then take all their queries
        cl = pd.Series(d).groupby(np.asarray(clusters))
        sums, ns = cl.sum().values, cl.size().values
        k = len(sums)
        idx = rng.integers(0, k, size=(B, k))
        means = sums[idx].sum(1) / ns[idx].sum(1)
    return np.percentile(means, [2.5, 97.5])


def test(d, clusters, rng):
    d = np.asarray(d, float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    try:
        w = st.wilcoxon(d, zero_method="wilcox").pvalue if np.any(d != 0) else 1.0
    except ValueError:
        w = 1.0
    lo, hi = boot_ci(d, clusters, rng)
    # asymptotic form of the same sign-flip randomization distribution
    # (variance of the permuted mean = sum(d^2)/n^2); resolves p below 1/B
    se0 = np.sqrt((d ** 2).sum()) / len(d)
    p_asym = 2 * st.norm.sf(abs(d.mean()) / se0) if se0 > 0 else 1.0
    return {"p_perm_asym": p_asym,"n": len(d), "mean_diff": d.mean(), "ci_lo": lo, "ci_hi": hi,
            "p_perm": sign_flip_p(d, rng), "p_wilcoxon": w, "d_z": d.mean() / sd if sd > 0 else np.nan,
            "wins": int((d > 0).sum()), "losses": int((d < 0).sum()), "ties": int((d == 0).sum())}


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for r, i in enumerate(order):
        run = max(run, (m - r) * p[i])
        adj[i] = min(1.0, run)
    return adj


def main(coll, metric="ndcg10"):
    rng = np.random.default_rng(SEED)
    df = pd.read_csv(RESULTS / f"{coll}_per_query.csv.gz", dtype={"qid": str})
    df = df[(df.rate == 0.15) | df.cond.str.startswith("real")]
    use_cluster = coll == "pquad"
    piv = df.pivot_table(index=["qid"], columns=["system", "cond"], values=metric)
    changed = df[df.system == "bm25_ws"].pivot_table(index="qid", columns="cond", values="changed")
    clusters = df.drop_duplicates("qid").set_index("qid")["cluster"].reindex(piv.index).values
    systems = sorted(df.system.unique())
    rows = []
    # A) within-system degradation: noisy - clean
    for s in systems:
        for c in MAIN[1:]:
            if (s, c) not in piv:
                continue
            mask = np.ones(len(piv), bool) if c in ("char", "typo", "comp") else changed[c].reindex(piv.index).fillna(0).values.astype(bool)
            if mask.sum() < 2:
                continue
            d = (piv[(s, c)] - piv[(s, "clean")]).values[mask]
            r = test(d, clusters[mask] if use_cluster else None, rng)
            rows.append({"family": "degradation", "cond": c, "system_a": s, "system_b": "(clean)",
                         "subset": "all" if mask.all() else "changed", **r})
    # B) system gains at fixed condition
    for a, b in GAIN_PAIRS:
        for c in MAIN:
            if (a, c) not in piv or (b, c) not in piv:
                continue
            d = (piv[(a, c)] - piv[(b, c)]).values
            r = test(d, clusters if use_cluster else None, rng)
            rows.append({"family": "gain", "cond": c, "system_a": a, "system_b": b, "subset": "all", **r})
    out = pd.DataFrame(rows)
    # p used for Holm: asymptotic randomization p for n >= 100 (Monte Carlo
    # p is floored at 1/(B+1)), Monte Carlo p for small samples
    out["p_used"] = np.where(out.n >= 100, out.p_perm_asym, out.p_perm)
    out["p_holm"] = holm(out.p_used.values)
    out.insert(0, "metric", metric)
    out.insert(0, "collection", coll)
    out.to_csv(RESULTS / f"{coll}_stats_{metric}.csv", index=False, float_format="%.5g")
    print(coll, metric, "comparisons:", len(out), "significant after Holm (0.05):", int((out.p_holm < 0.05).sum()))
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "ndcg10")
