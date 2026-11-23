"""Render markdown tables from results/*.csv for the manuscript source file.

Author: Mohammad Naser Sadrmanesh
"""
import json
import pandas as pd
from common import RESULTS
from figures import NAMES, ORDER

MAINC = ["clean", "char", "zwnj", "typo", "comp"]
REALC = ["real", "real_arabic", "real_ortho", "real_zwnj"]


def f4(x):
    return "–" if pd.isna(x) else f"{x:.4f}"


def table_main(coll, metric="ndcg10"):
    g = pd.read_csv(RESULTS / f"{coll}_summary.csv")
    g = g[(g.rate == 0.15) | g.cond.str.startswith("real")]
    piv = g.pivot_table(index="system", columns="cond", values=metric)
    systems = [s for s in ORDER if s in piv.index]
    hdr = "| System | " + " | ".join(MAINC) + " | ΔnDCG char | ΔnDCG zwnj | ΔnDCG typo | ΔnDCG comp |"
    lines = [hdr, "|" + "---|" * (1 + len(MAINC) + 4)]
    for s in systems:
        r = piv.loc[s]
        d = [r[c] - r["clean"] for c in MAINC[1:]]
        lines.append(f"| {NAMES[s]} | " + " | ".join(f4(r.get(c)) for c in MAINC) + " | " + " | ".join(f"{x:+.4f}" for x in d) + " |")
    return "\n".join(lines)


def table_changed(coll):
    g = pd.read_csv(RESULTS / f"{coll}_summary.csv")
    g = g[g.cond.isin(["zwnj"] + REALC) & ((g.rate == 0.15) | g.cond.str.startswith("real"))]
    systems = [s for s in ORDER if s in set(g.system)]
    conds = ["zwnj"] + REALC
    n = {c: int(g[g.cond == c].n_changed.max()) if g[g.cond == c].n_changed.notna().any() else 0 for c in conds}
    lines = ["| System | " + " | ".join(f"{c} (n={n[c]})" for c in conds) + " |", "|" + "---|" * (1 + len(conds))]
    for s in systems:
        row = []
        for c in conds:
            r = g[(g.system == s) & (g.cond == c)]
            row.append("–" if r.empty or pd.isna(r.delta_ndcg10_changed.iloc[0]) else f"{r.clean_ndcg10_changed.iloc[0]:.3f}→{r.ndcg10_changed.iloc[0]:.3f} ({r.delta_ndcg10_changed.iloc[0]:+.3f})")
        lines.append(f"| {NAMES[s]} | " + " | ".join(row) + " |")
    return "\n".join(lines)


def table_dose(coll):
    g = pd.read_csv(RESULTS / f"{coll}_summary.csv")
    systems = [s for s in ORDER if s in set(g.system)]
    rates = [0.05, 0.10, 0.15, 0.20, 0.30]
    out = []
    for op in ["typo", "comp"]:
        lines = [f"**T_{op}** (nDCG@10; EditRate 0 = clean)", "", "| System | 0 | " + " | ".join(f"{r:.2f}" for r in rates) + " | slope (ΔnDCG per +0.10 EditRate) |", "|" + "---|" * (len(rates) + 3)]
        for s in systems:
            sub = g[(g.system == s) & (g.cond == op)].set_index("rate")
            if len(sub) < len(rates):
                continue
            clean = g[(g.system == s) & (g.cond == "clean")].ndcg10.iloc[0]
            ys = [sub.loc[r, "ndcg10"] for r in rates]
            import numpy as np
            x = np.array([0] + rates); y = np.array([clean] + ys)
            slope = np.polyfit(x, y, 1)[0] * 0.1
            lines.append(f"| {NAMES[s]} | {clean:.4f} | " + " | ".join(f"{v:.4f}" for v in ys) + f" | {slope:+.4f} |")
        out.append("\n".join(lines))
    return "\n\n".join(out)


def table_stats(coll, family, metric="ndcg10", conds=None, pairs=None):
    st = pd.read_csv(RESULTS / f"{coll}_stats_{metric}.csv")
    st = st[st.family == family]
    if conds:
        st = st[st.cond.isin(conds)]
    if pairs:
        st = st[[ (a, b) in pairs for a, b in zip(st.system_a, st.system_b)]]
    lines = ["| A | B | cond | subset | n | mean Δ (A−B) | 95% CI | d_z | W/L/T | p (rand.) | p (Holm) |", "|" + "---|" * 11]
    for r in st.itertuples():
        lines.append(f"| {NAMES.get(r.system_a, r.system_a)} | {NAMES.get(r.system_b, r.system_b)} | {r.cond} | {r.subset} | {r.n} | {r.mean_diff:+.4f} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {r.d_z:+.2f} | {r.wins}/{r.losses}/{r.ties} | {r.p_used:.2g} | {r.p_holm:.2g} |")
    return "\n".join(lines), len(pd.read_csv(RESULTS / f"{coll}_stats_{metric}.csv"))


if __name__ == "__main__":
    import sys
    for c in sys.argv[1:]:
        print(table_main(c)); print(); print(table_changed(c)); print()
