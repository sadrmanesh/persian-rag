"""Schematic figures: pipeline architecture, perturbation workflow, taxonomy, dataset statistics.

Author: Mohammad Naser Sadrmanesh
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from figstyle import OKABE, fa, FA_FONT, save
from common import DATA, FIGS, read_jsonl
from perturb import perturb

C_IN, C_NORM, C_SP, C_DE, C_FUS, C_OUT = "#F2F2F2", "#FBE3B5", "#CFE6F5", "#C8EBDD", "#F3D3E5", "#E0E0E0"


def box(ax, x, y, w, h, text, fc, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#333333", lw=0.8))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", wrap=True)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=9,
                                 lw=0.9, color="#333333", linestyle=ls))


def architecture():
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    ax.set_xlim(0, 14.2); ax.set_ylim(0, 7.2); ax.axis("off"); ax.grid(False)
    box(ax, 0.1, 3.0, 1.9, 1.0, "Query q\n(possibly\nnoisy)", C_IN, fs=7)
    box(ax, 2.5, 4.6, 2.4, 1.1, "Query rewrite\n(spelling table,\northographic &\naffix variants)", C_NORM, fs=6.3)
    box(ax, 2.5, 2.4, 2.4, 1.6, "Normalisation N_PR\nNFKC · Arabic→Persian\nletters · digits ·\nZWNJ / half-space fix", C_NORM, fs=6.3)
    box(ax, 5.6, 4.4, 2.3, 1.2, "BM25\n(k1=1.5, b=0.75)\nPersian tokeniser", C_SP, fs=6.3)
    box(ax, 5.6, 2.4, 2.3, 1.2, "Dense encoder\nmE5 / BGE-M3 /\nMiniLM (cosine)", C_DE, fs=6.3)
    box(ax, 5.6, 0.5, 2.3, 1.2, "Char 2–4-gram\nhashing embedder\n(512-d)", C_DE, fs=6.3)
    box(ax, 8.7, 2.6, 2.2, 1.9, "Reciprocal Rank\nFusion\nΣ w_i / (k + r_i)\nk = 60", C_FUS, fs=6.5)
    box(ax, 11.6, 2.9, 2.2, 1.3, "Ranked\npassages (top-k)\n→ RAG / eval.", C_OUT, fs=6.5)
    box(ax, 5.3, 6.1, 2.9, 0.9, "Passage index\n(same N_PR on documents)", C_IN, fs=6.3)
    arrow(ax, 2.0, 3.5, 2.5, 3.2); arrow(ax, 2.0, 3.6, 2.5, 5.0)
    arrow(ax, 4.9, 5.1, 5.6, 5.0, ls="--")
    arrow(ax, 4.9, 3.4, 5.6, 4.7); arrow(ax, 4.9, 3.1, 5.6, 3.0); arrow(ax, 4.9, 2.8, 5.6, 1.2)
    arrow(ax, 6.75, 6.1, 6.75, 5.6)
    arrow(ax, 7.9, 5.0, 8.7, 4.1); arrow(ax, 7.9, 3.0, 8.7, 3.4); arrow(ax, 7.9, 1.1, 8.7, 2.9)
    arrow(ax, 10.9, 3.55, 11.6, 3.55)
    ax.text(3.7, 6.2, "variants fused by weighted RRF\n(original 2.0, variants share 1.0)", ha="center", fontsize=6, style="italic")
    ax.text(7, 0.05, "Solid: always applied · dashed: optional query rewriting", ha="center", fontsize=6.5, color="#444444")
    save(fig, FIGS / "fig1_architecture", "persian-rag retrieval pipeline")


def workflow():
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.6); ax.axis("off"); ax.grid(False)
    box(ax, 0.1, 2.6, 2.0, 1.3, "Clean query q\n+ query id", C_IN, fs=7)
    box(ax, 0.1, 0.4, 2.0, 1.4, "RNG seed =\nSHA-256(42 | id\n| operator | rate)", C_OUT, fs=6.3)
    box(ax, 2.9, 4.4, 2.6, 1.3, "T_char\nkaf, yeh → Arabic forms\nalef-madda → alef", C_NORM, fs=6.3)
    box(ax, 2.9, 2.6, 2.6, 1.3, "T_zwnj\neach ZWNJ: delete\n(p=.5) or → space (p=.5)", C_NORM, fs=6.3)
    box(ax, 2.9, 0.4, 2.6, 1.6, "T_typo (per token,\np = EditRate): insert ·\ndelete · transpose ·\nhomophone · ISIRI 9147\nadjacent-key subst.", C_NORM, fs=6.0)
    box(ax, 6.4, 2.6, 2.4, 1.3, "T_comp =\nT_typo ∘ T_zwnj\n∘ T_char", C_FUS, fs=6.5)
    box(ax, 6.4, 4.5, 2.4, 1.2, "T_real: attested\ncorpus variants\n(hamza, madda,\nArabic yeh/kaf, ZWNJ)", C_DE, fs=6.0)
    box(ax, 6.4, 0.5, 2.4, 1.3, "Dose–response\nEditRate ∈\n{.05, .10, .15,\n.20, .30}", C_SP, fs=6.3)
    box(ax, 9.6, 2.4, 2.0, 1.7, "Perturbed\nquery set\n(qid, cond, rate,\ntext, changed)", C_IN, fs=6.3)
    box(ax, 12.2, 2.4, 1.7, 1.7, "Retrieval,\nper-query\nmetrics,\npaired tests", C_OUT, fs=6.3)
    arrow(ax, 2.1, 3.25, 2.9, 5.0); arrow(ax, 2.1, 3.25, 2.9, 3.25); arrow(ax, 2.1, 3.1, 2.9, 1.2)
    arrow(ax, 2.1, 1.1, 2.9, 1.0, ls="--"); arrow(ax, 1.1, 1.8, 1.1, 2.6, ls="--")
    arrow(ax, 5.5, 5.0, 6.4, 3.6); arrow(ax, 5.5, 3.25, 6.4, 3.25); arrow(ax, 5.5, 1.2, 6.4, 2.9)
    arrow(ax, 5.5, 1.1, 6.4, 1.1)
    arrow(ax, 8.8, 5.1, 9.6, 3.9); arrow(ax, 8.8, 3.25, 9.6, 3.25); arrow(ax, 8.8, 1.2, 9.6, 2.6)
    arrow(ax, 11.6, 3.25, 12.2, 3.25)
    ax.text(7.6, 6.15, "Documents are never perturbed; only queries are.", ha="center", fontsize=7, style="italic")
    save(fig, FIGS / "fig2_perturbation_workflow", "Perturbation generation workflow")


def taxonomy():
    qs = {q["id"]: q for q in read_jsonl(DATA / "miracl_queries.jsonl")}
    pert = read_jsonl(DATA / "miracl_queries_perturbed.jsonl")
    by = {}
    for r in pert:
        by.setdefault(r["qid"], {})[(r["cond"], r["rate"])] = r
    # pick a query that contains ZWNJ and has attested variants (real examples from the run)
    pick = None
    for qid, d in by.items():
        if d[("zwnj", 0.15)]["changed"] and d[("real", 0.0)]["changed"] and d[("typo", 0.15)]["changed"] and len(qs[qid]["text"]) < 48 and "ک" in qs[qid]["text"] and "ی " in qs[qid]["text"]:
            pick = qid; break
    d = by[pick]
    rows = [("Clean query (MIRACL-fa dev)", d[("clean", 0.15)]["text"], "—"),
            ("T_char: Arabic-keyboard letters", d[("char", 0.15)]["text"], "kaf/yeh → Arabic; madda dropped"),
            ("T_zwnj: half-space loss", d[("zwnj", 0.15)]["text"], "ZWNJ deleted or → space"),
            ("T_typo (EditRate 0.15)", d[("typo", 0.15)]["text"], "keyboard/homophone/ins/del/transp."),
            ("T_comp = typo∘zwnj∘char", d[("comp", 0.15)]["text"], "all of the above"),
            ("T_real: attested variants", d[("real", 0.0)]["text"], "forms observed in the corpus")]
    lex = [("هیئت", "هیات", "hamza seat (yeh-hamza → alef)"), ("رئیس", "رییس", "hamza seat (yeh-hamza → yeh)"), ("مسئول", "مسوول", "hamza seat (yeh-hamza → waw)"),
           ("آمریکا", "امریکا", "alef madda → alef"), ("می‌شود", "می شود / میشود", "ZWNJ → space / joined"), ("کتاب", "كتاب", "Persian kaf → Arabic kaf")]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7.2, 5.4), gridspec_kw={"height_ratios": [1.25, 1]})
    for ax in (a1, a2):
        ax.axis("off"); ax.grid(False)
    a1.set_title("(a) Operators applied to one real query (outputs taken from the generated query set)", fontsize=9, loc="left")
    cols = [0.0, 0.33, 0.78]
    for j, h in enumerate(["Operator", "Query text", "Change"]):
        a1.text(cols[j], 0.97, h, fontweight="bold", fontsize=8, transform=a1.transAxes, va="top")
    for i, (name, text, note) in enumerate(rows):
        y = 0.84 - i * 0.145
        a1.add_patch(plt.Rectangle((0, y - 0.06), 1, 0.13, transform=a1.transAxes, color=OKABE[1 if i % 2 else 8], alpha=0.12, lw=0))
        a1.text(cols[0], y, name, fontsize=7.5, transform=a1.transAxes, va="center")
        a1.text(cols[2] - 0.02, y, fa(text), fontsize=9, transform=a1.transAxes, va="center", ha="right", fontproperties=FA_FONT)
        a1.text(cols[2], y, note, fontsize=7, transform=a1.transAxes, va="center", fontproperties=FA_FONT if any("\u0600" <= c <= "\u06ff" for c in note) else None)
    a2.set_title("(b) Attested orthographic alternations (each pair occurs ≥5× in the MIRACL-fa pool)", fontsize=9, loc="left")
    for j, h in enumerate(["Standard form", "Attested variant", "Alternation"]):
        a2.text([0.05, 0.38, 0.70][j], 0.97, h, fontweight="bold", fontsize=8, transform=a2.transAxes, va="top")
    for i, (a, b, n) in enumerate(lex):
        y = 0.82 - i * 0.14
        a2.text(0.12, y, fa(a), fontproperties=FA_FONT, fontsize=10, transform=a2.transAxes, va="center", ha="center")
        a2.text(0.47, y, fa(b) if "/" not in b else fa(b.split(" / ")[0]) + "  /  " + fa(b.split(" / ")[1]), fontproperties=FA_FONT, fontsize=10, transform=a2.transAxes, va="center", ha="center")
        a2.text(0.70, y, n.split(" ")[0] + " " + n.split(" ")[1] if "→" in n and any("\u0600" <= c <= "\u06ff" for c in n) else n, fontsize=7.5, transform=a2.transAxes, va="center")
    save(fig, FIGS / "fig3_perturbation_taxonomy", "Perturbation taxonomy with Persian examples")
    return pick


def dataset_stats():
    st = json.load(open(DATA.parent / "results" / "attested_variants_stats.json"))
    colls = {"MIRACL-fa": "miracl", "PQuAD-val": "pquad", "Gold": "gold"}
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), gridspec_kw={"width_ratios": [1, 1, 1.15]})
    # (a) query length
    for k, (lab, c) in enumerate(colls.items()):
        qs = read_jsonl(DATA / f"{c}_queries.jsonl")
        L = [len(q["text"].split()) for q in qs]
        axes[0].hist(L, bins=np.arange(0.5, 25.5, 1), histtype="step", lw=1.3, density=True, color=OKABE[[4, 5, 2][k]], label=f"{lab} (n={len(qs)})")
    axes[0].set_xlabel("query length (whitespace tokens)"); axes[0].set_ylabel("density"); axes[0].legend(fontsize=6, loc="upper right", frameon=False)
    axes[0].set_title("(a) Query length", fontsize=9)
    # (b) share of queries changed by each operator
    pert = {c: read_jsonl(DATA / f"{c}_queries_perturbed.jsonl") for c in colls.values()}
    ops = [("char", 0.15), ("zwnj", 0.15), ("typo", 0.15), ("comp", 0.15), ("real", 0.0)]
    w = 0.26
    for k, (lab, c) in enumerate(colls.items()):
        vals = []
        for op in ops:
            rs = [r for r in pert[c] if (r["cond"], r["rate"]) == op]
            vals.append(100 * sum(r["changed"] for r in rs) / len(rs))
        axes[1].bar(np.arange(len(ops)) + (k - 1) * w, vals, w, color=OKABE[[4, 5, 2][k]], label=lab)
    axes[1].set_xticks(range(len(ops))); axes[1].set_xticklabels(["T_" + o[0] for o in ops], rotation=30, fontsize=7)
    axes[1].set_ylabel("% queries changed"); axes[1].set_title("(b) Operator coverage", fontsize=9)
    axes[1].legend(fontsize=6.5, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.28), frameon=False)
    # (c) attested variants by rule
    br = st["by_rule"]
    keys = sorted(br, key=lambda k: -br[k])
    axes[2].barh(range(len(keys)), [br[k] for k in keys], color=OKABE[2])
    axes[2].set_yticks(range(len(keys)))
    LBL = {"zwnj->space": "ZWNJ → space", "zwnj->joined": "ZWNJ → joined", "آ->ا": "alef-madda → alef",
           "ئ->ی": "yeh-hamza → yeh", "ی->ي": "Persian → Arabic yeh", "أ->ا": "hamza-alef → alef",
           "ؤ->و": "waw-hamza → waw", "ئ->و": "yeh-hamza → waw", "ک->ك": "Persian → Arabic kaf",
           "ئ->ا": "yeh-hamza → alef", "ۀ->ه": "heh-yeh → heh", "ة->ه": "teh marbuta → heh", "إ->ا": "hamza-below → alef"}
    axes[2].set_yticklabels([LBL.get(k, k) for k in keys], fontsize=6.5)
    axes[2].invert_yaxis(); axes[2].set_xscale("log"); axes[2].set_xlabel("# words with attested variant")
    axes[2].set_title("(c) Attested alternations", fontsize=9)
    fig.tight_layout()
    save(fig, FIGS / "fig7_dataset_statistics", "Dataset and perturbation statistics")


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    architecture(); workflow(); print("taxonomy query:", taxonomy()); dataset_stats()
