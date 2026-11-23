"""Honest significance analysis for the 42-query gold benchmark.

Compares BM25+N_PR with and without rule-based rewriting (clean queries) on
all 42 queries and on the 12 'spelling' queries. Hit@1 is binary, so the
appropriate paired test is the exact McNemar (binomial) test on discordant
pairs; we also give a sign-flip randomization test on MRR@10, Wilcoxon on
MRR@10, a bootstrap CI and Cohen's d_z / Cohen's g for the binary outcome.

Author: Mohammad Naser Sadrmanesh
"""
import json
import numpy as np
import pandas as pd
from scipy import stats as st
from common import RESULTS, SEED
from stats import test

df = pd.read_csv(RESULTS / "gold_per_query.csv.gz", dtype={"qid": str})
df = df[df.cond == "clean"]
rng = np.random.default_rng(SEED)
out = {}
pairs = [("bm25_npr_rewrite", "bm25_npr")]
for extra in ("e5_small", "e5_base", "bge_m3", "hybrid_bm25npr_e5small"):
    if extra in set(df.system):
        pairs.append((extra, "bm25_npr"))
for a, b in pairs:
    for subset in ("all", "spelling"):
        A = df[df.system == a].set_index("qid").sort_index()
        Bq = df[df.system == b].set_index("qid").sort_index()
        if subset == "spelling":
            m = A.tags.fillna("").str.contains("spelling")
            A, Bq = A[m], Bq[m]
        h_a, h_b = A.hit1.values, Bq.hit1.values
        n01 = int(((h_a == 1) & (h_b == 0)).sum())   # a right, b wrong
        n10 = int(((h_a == 0) & (h_b == 1)).sum())
        nd = n01 + n10
        p_mcnemar = st.binomtest(n01, nd, 0.5).pvalue if nd else 1.0
        dz_hit = (h_a - h_b).mean() / (h_a - h_b).std(ddof=1) if (h_a - h_b).std(ddof=1) > 0 else float("nan")
        r = test(A.mrr10.values - Bq.mrr10.values, None, rng)
        out[f"{a} vs {b} [{subset}]"] = {
            "n": int(len(A)), "hit1_a": float(h_a.mean()), "hit1_b": float(h_b.mean()),
            "discordant_a_better": n01, "discordant_b_better": n10,
            "exact_mcnemar_p_two_sided": float(p_mcnemar),
            "cohens_g": float(n01 / nd - 0.5) if nd else None,
            "d_z_hit1": float(dz_hit),
            "mrr10_a": float(A.mrr10.mean()), "mrr10_b": float(Bq.mrr10.mean()),
            "mrr10_diff_ci95": [float(r["ci_lo"]), float(r["ci_hi"])],
            "mrr10_sign_flip_p": float(r["p_perm"]), "mrr10_wilcoxon_p": float(r["p_wilcoxon"]),
            "d_z_mrr10": float(r["d_z"]),
        }
json.dump(out, open(RESULTS / "gold_significance.json", "w"), indent=1)
print(json.dumps(out, indent=1))
