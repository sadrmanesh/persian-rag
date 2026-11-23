"""Result figures: degradation (grouped bars + CIs), dose-response, retention heatmap.

Author: Mohammad Naser Sadrmanesh
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from figstyle import OKABE, save
from common import RESULTS, FIGS

NAMES = {
    "bm25_ws": "BM25 (whitespace)", "bm25_hazm": "BM25 + Hazm", "bm25_parsivar": "BM25 + Parsivar",
    "bm25_npr": "BM25 + N_PR", "bm25_npr_rewrite": "BM25 + N_PR + rewrite", "hashing": "Hashing (char 2–4g)",
    "hybrid_bm25npr_hashing": "RRF(BM25+N_PR, Hashing)", "e5_small": "mE5-small", "e5_base": "mE5-base",
    "bge_m3": "BGE-M3 (dense)", "minilm": "MiniLM-L12",
    "e5_small_qnorm": "mE5-small + N_PR(q)", "e5_base_qnorm": "mE5-base + N_PR(q)", "bge_m3_qnorm": "BGE-M3 + N_PR(q)",
    "minilm_qnorm": "MiniLM-L12 + N_PR(q)",
    "hybrid_bm25npr_e5small": "RRF(BM25+N_PR, mE5-small)", "hybrid_bm25npr_e5base": "RRF(BM25+N_PR, mE5-base)",
    "hybrid_bm25npr_bgem3": "RRF(BM25+N_PR, BGE-M3)", "hybrid_bm25npr_minilm": "RRF(BM25+N_PR, MiniLM)",
}
ORDER = ["bm25_ws", "bm25_hazm", "bm25_parsivar", "bm25_npr", "bm25_npr_rewrite", "hashing",
         "hybrid_bm25npr_hashing", "minilm", "minilm_qnorm", "e5_small", "e5_small_qnorm", "e5_base", "e5_base_qnorm",
         "bge_m3", "bge_m3_qnorm", "hybrid_bm25npr_minilm", "hybrid_bm25npr_e5small", "hybrid_bm25npr_e5base", "hybrid_bm25npr_bgem3"]
FIG_SYSTEMS = ["bm25_ws", "bm25_hazm", "bm25_parsivar", "bm25_npr", "bm25_npr_rewrite", "hashing",
               "minilm", "e5_small", "e5_base", "bge_m3", "e5_small_qnorm",
               "hybrid_bm25npr_e5small", "hybrid_bm25npr_e5base", "hybrid_bm25npr_bgem3"]
CONDS = ["char", "zwnj", "typo", "comp", "real"]
CCOL = {"char": OKABE[4], "zwnj": OKABE[0], "typo": OKABE[2], "comp": OKABE[5], "real": OKABE[6]}
TITLES = {"miracl": "MIRACL-fa dev (632 queries) vs 100k passages", "miracl30k": "MIRACL-fa dev (632 queries) vs 30k passages",
          "pquad": "PQuAD validation (5,995 questions) vs 1,103 paragraphs", "gold": "Gold benchmark (42 queries, 34 passages)"}


def degradation(coll):
    st = pd.read_csv(RESULTS / f"{coll}_stats_ndcg10.csv")
    st = st[st.family == "degradation"]
    systems = [s for s in FIG_SYSTEMS if s in set(st.system_a)]
    fig, ax = plt.subplots(figsize=(max(7.0, 0.55 * len(systems) + 2.5), 3.6))
    w = 0.16
    for k, c in enumerate(CONDS):
        sub = st[st.cond == c].set_index("system_a")
        xs, ys, lo, hi = [], [], [], []
        for i, s in enumerate(systems):
            if s in sub.index:
                r = sub.loc[s]
                xs.append(i + (k - 2) * w); ys.append(r.mean_diff)
                lo.append(max(0, r.mean_diff - r.ci_lo)); hi.append(max(0, r.ci_hi - r.mean_diff))
        lab = {"real": "T_real (attested)†", "zwnj": "T_zwnj†"}.get(c, "T_" + c)
        ax.bar(xs, ys, w, yerr=[lo, hi], capsize=1.2, color=CCOL[c], label=lab, error_kw={"lw": 0.5}, edgecolor="none")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(systems)))
    ax.set_xticklabels([NAMES[s] for s in systems], rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("ΔnDCG@10 (noisy − clean)")
    ax.set_title(TITLES.get(coll, coll), fontsize=9)
    ax.legend(fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.07), frameon=False)
    ax.grid(axis="x", visible=False)
    save(fig, FIGS / f"fig4_degradation_{coll}", f"Within-system degradation, {coll}")


def dose(coll):
    df = pd.read_csv(RESULTS / f"{coll}_summary.csv")
    systems = [s for s in FIG_SYSTEMS if s in set(df.system)]
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharey=True)
    styles = ["-", "--", "-.", ":"]
    for ax, op in zip(axes, ["typo", "comp"]):
        k = 0
        for s in systems:
            sub = df[(df.system == s) & (df.cond == op)].sort_values("rate")
            if len(sub) < 5:
                continue
            clean = df[(df.system == s) & (df.cond == "clean")].ndcg10.iloc[0]
            x = [0.0] + sub.rate.tolist(); y = [clean] + sub.ndcg10.tolist()
            ax.plot(x, y, styles[k // 8 % 4], marker="os^Dv<>p"[k % 8], ms=3, lw=1.1,
                    color=[OKABE[i] for i in (0, 1, 2, 4, 5, 6, 7, 8)][k % 8], label=NAMES[s])
            k += 1
        ax.set_title(f"T_{op}" + (" (T_typo ∘ T_zwnj ∘ T_char)" if op == "comp" else ""), fontsize=9)
        ax.set_xlabel("EditRate (per-token edit probability; 0 = clean)")
        ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20, 0.30])
    axes[0].set_ylabel("nDCG@10")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.suptitle(TITLES.get(coll, coll), fontsize=9, y=1.0)
    save(fig, FIGS / f"fig5_dose_response_{coll}", f"Dose-response, {coll}")


def retention(coll):
    g = pd.read_csv(RESULTS / f"{coll}_summary.csv")
    conds = ["char", "zwnj", "typo", "comp", "real", "real_arabic", "real_ortho", "real_zwnj"]
    systems = [s for s in ORDER if s in set(g.system)]
    M = np.full((len(systems), len(conds)), np.nan)
    for i, s in enumerate(systems):
        for j, c in enumerate(conds):
            r = g[(g.system == s) & (g.cond == c) & ((g.rate == 0.15) | g.cond.str.startswith("real"))]
            if r.empty:
                continue
            r = r.iloc[0]
            if c in ("char", "typo", "comp"):
                M[i, j] = r.retention_ndcg10
            elif r.n_changed and r.clean_ndcg10_changed > 0:
                M[i, j] = r.ndcg10_changed / r.clean_ndcg10_changed
    fig, ax = plt.subplots(figsize=(6.4, 0.32 * len(systems) + 1.4))
    im = ax.imshow(M, cmap="cividis", vmin=np.nanmin(M), vmax=1.0, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if M[i, j] < (np.nanmin(M) + 1) / 2 else "black")
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(["T_char", "T_zwnj†", "T_typo", "T_comp", "T_real†", "real: Arabic†", "real: ortho†", "real: ZWNJ†"], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(systems))); ax.set_yticklabels([NAMES[s] for s in systems], fontsize=7)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Retention ratio R = nDCG@10 noisy / clean", fontsize=7)
    ax.set_title(TITLES.get(coll, coll), fontsize=9)
    save(fig, FIGS / f"fig6_retention_heatmap_{coll}", f"Retention ratio heatmap, {coll}")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    for coll in sys.argv[1:]:
        degradation(coll); retention(coll)
        if coll.startswith("miracl") or coll == "pquad":
            dose(coll)
        print("figures", coll)
