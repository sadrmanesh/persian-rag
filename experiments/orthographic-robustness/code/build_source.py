"""Assemble the revision source document from parts + result tables.

Author: Mohammad Naser Sadrmanesh
"""
import json
from pathlib import Path
import pandas as pd
from common import RESULTS, ROOT
from make_tables import table_main, table_changed, table_dose, table_stats

P = ROOT / "source_parts"
DENSE = {"e5_small", "e5_base", "bge_m3", "minilm"}


def have(coll):
    return (RESULTS / f"{coll}_summary.csv").exists()


def pending_note(coll):
    if not have(coll):
        return "\n> **PENDING**: results for this collection are not yet available.\n"
    s = set(pd.read_csv(RESULTS / f"{coll}_summary.csv").system)
    miss = sorted(DENSE - s)
    expected = {"pquad": {"e5_small", "e5_base", "minilm"}, "miracl": {"e5_small", "minilm"},
                "miracl30k": DENSE, "gold": DENSE}[coll]
    miss = sorted(expected - s)
    return f"\n> **PENDING** rows for: {', '.join(miss)} (run in progress; table will be regenerated).\n" if miss else ""


def stats_block(coll, title):
    if not (RESULTS / f"{coll}_stats_ndcg10.csv").exists():
        return f"\n#### {title}\n> **PENDING**\n"
    pairs = [("bm25_npr", "bm25_ws"), ("bm25_npr", "bm25_hazm"), ("bm25_npr", "bm25_parsivar"),
             ("bm25_npr_rewrite", "bm25_npr"), ("hashing", "bm25_npr"),
             ("e5_small", "bm25_npr"), ("e5_small_qnorm", "e5_small"), ("e5_base_qnorm", "e5_base"),
             ("bge_m3_qnorm", "bge_m3"), ("bge_m3", "e5_base"), ("e5_base", "e5_small"),
             ("hybrid_bm25npr_e5small", "e5_small"), ("hybrid_bm25npr_e5base", "e5_base"), ("hybrid_bm25npr_bgem3", "bge_m3")]
    t, n = table_stats(coll, "gain", conds=["clean", "char", "typo", "comp", "real"], pairs=pairs)
    d, _ = table_stats(coll, "degradation", conds=["char", "zwnj", "typo", "comp", "real_ortho"])
    return (f"\n#### {title}\nFamily size for Holm correction on this collection: **{n} comparisons**.\n\n"
            f"**System gains G (A − B), nDCG@10:**\n\n{t}\n\n**Within-system degradation ΔnDCG@10 (noisy − clean):**\n\n{d}\n")


def main():
    out = [(P / "01_setup.md").read_text()]
    out.append("\n## 3. Results\n\nAll values are means over queries; Δ columns are within-system (noisy − clean). "
               "Rows marked N_PR(q) use the dense model with an N_PR-normalised query.\n")
    for k, (coll, title) in enumerate([("miracl", "MIRACL-fa dev vs 100k passages"), ("miracl30k", "MIRACL-fa dev vs 30k passages (incl. mE5-base, BGE-M3)"),
                                       ("pquad", "PQuAD validation (5,995 questions, 1,103 paragraphs)"), ("gold", "Gold benchmark (42 queries)")], 1):
        out.append(f"\n### Table 1.{k} — nDCG@10, {title}\n" + pending_note(coll))
        if have(coll):
            out.append(table_main(coll) + "\n")
            out.append(f"\n**Table 2.{k} — changed-query subsets** (nDCG@10 clean → noisy (Δ) on the queries the operator actually changed; n = number of changed queries):\n\n" + table_changed(coll) + "\n")
    out.append("\n### Table 3 — Recall@100, MIRACL-fa-30k\n")
    g = pd.read_csv(RESULTS / "miracl30k_summary.csv"); g = g[g.rate == 0.15]
    pv = g.pivot_table(index="system", columns="cond", values="recall100")[["clean", "char", "zwnj", "typo", "comp"]]
    from figures import NAMES, ORDER
    lines = ["| System | clean | char | zwnj | typo | comp |", "|---|---|---|---|---|---|"]
    for s in [s for s in ORDER if s in pv.index]:
        lines.append(f"| {NAMES[s]} | " + " | ".join(f"{v:.3f}" for v in pv.loc[s]) + " |")
    out.append("\n".join(lines) + "\n")
    for coll in ("miracl", "miracl30k", "pquad"):
        if have(coll):
            out.append(f"\n### Table 5 ({coll}) — dose–response over EditRate\n\n" + table_dose(coll) + "\n")
    out.append((P / "03_findings.md").read_text())
    out.append("\n## 5. Statistical results\n\nTwo-sided paired randomisation tests; 95% bootstrap CIs; d_z = mean/SD of paired differences; "
               "W/L/T = queries where A wins/loses/ties; p (Holm) adjusted over the full family. Full tables, including MRR and every condition: `results/*_stats_*.csv`.\n")
    for coll, title in [("miracl", "MIRACL-fa-100k"), ("miracl30k", "MIRACL-fa-30k"), ("pquad", "PQuAD (cluster bootstrap over paragraphs)")]:
        out.append(stats_block(coll, title))
    gs = json.load(open(RESULTS / "gold_significance.json"))
    lines = ["\n#### Gold benchmark (exact tests)\n", "| Comparison | subset | n | Hit@1 A | Hit@1 B | discordant A>B / B>A | exact McNemar p | Cohen's g | d_z (Hit@1) | MRR@10 A − B, 95% CI | sign-flip p (MRR) |", "|" + "---|" * 11]
    for k, v in gs.items():
        name, sub = k.split(" [")
        lines.append(f"| {name.replace('bm25_npr_rewrite','BM25+N_PR+rewrite').replace('bm25_npr','BM25+N_PR')} | {sub.rstrip(']')} | {v['n']} | {v['hit1_a']:.3f} | {v['hit1_b']:.3f} | {v['discordant_a_better']} / {v['discordant_b_better']} | {v['exact_mcnemar_p_two_sided']:.3g} | {v['cohens_g']:.2f} | {v['d_z_hit1']:.2f} | {v['mrr10_a']-v['mrr10_b']:+.3f} [{v['mrr10_diff_ci95'][0]:+.3f}, {v['mrr10_diff_ci95'][1]:+.3f}] | {v['mrr10_sign_flip_p']:.3g} |")
    out.append("\n".join(lines) + "\n\nThe repository's own benchmark script reproduces the same point estimates (sparse Hit@1 0.881, sparse + rewrite 0.952; spelling 0.583 → 0.833): `results/gold_library_benchmark_hashing.json`.\n")
    cap = (ROOT / "figures" / "captions.md").read_text().split("\n", 2)[2]
    out.append("\n## 5b. Figures\n\nFiles are in `figures/` (300-dpi PNG + vector PDF). Per-collection variants exist for Figures 4–6 (`_miracl`, `_miracl30k`, `_pquad`, `_gold`); the main-text versions should be `_miracl30k` (all four dense models) with `_miracl` and `_pquad` in the appendix.\n\n" + cap)
    out.append((P / "05_limitations.md").read_text())
    out.append((P / "06_revised_text.md").read_text())
    out.append((P / "08_references_intro.md").read_text())
    out.append("### 8.2 Full verified reference list (APA)\n\n" + (ROOT / "refs" / "references_apa.md").read_text())
    out.append("\n## 9. Release checklist for the camera-ready version\n\n- Publish `code/`, the perturbed query files (`data/*_queries_perturbed.jsonl`), the attested-variant list (`data/attested_variants.json`), the MIRACL pool document ids and all per-query result files as a versioned release, and archive it (e.g., Zenodo DOI).\n- State software versions and the seed (42) in the paper.\n- Replace the non-existent script names in the draft.\n- Add a data/code availability statement and the licences of MIRACL (Apache-2.0) and PQuAD (check the dataset card).\n")
    (ROOT / "revision_source.md").write_text("\n".join(out))
    print("written", len("\n".join(out)), "chars")


if __name__ == "__main__":
    main()
