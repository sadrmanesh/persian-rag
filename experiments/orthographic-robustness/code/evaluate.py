"""Per-query metrics and summary tables from cached runs.

Usage: python evaluate.py <collection> [<collection> ...]
Writes results/<coll>_per_query.csv.gz and results/<coll>_summary.csv

Delta definitions (used consistently everywhere):
  within-system degradation  Delta_s(T) = M_s(T(q)) - M_s(q)          (noisy - clean, same system)
  retention                  R_s(T)     = M_s(T(q)) / M_s(q)
  system gain                G_{A,B}(T) = M_A(T(q)) - M_B(T(q))       (two systems, same queries)

Author: Mohammad Naser Sadrmanesh
"""
import sys
import pandas as pd
from common import RESULTS, CACHE, metrics_for
from run_retrieval import load, load_run


def per_query(coll):
    docs, _, queries, pert = load(coll)
    pos = {d["id"]: i for i, d in enumerate(docs)}
    rel = {q["id"]: {pos[d] for d in q["rel"] if d in pos} for q in queries}
    cluster = {q["id"]: q.get("cluster", q["id"]) for q in queries}
    tags = {q["id"]: ",".join(q.get("tags", [])) for q in queries}
    changed = {(r["qid"], r["cond"], r["rate"]): r["changed"] for r in pert}
    rows = []
    for f in sorted((CACHE / "runs" / coll).glob("*.pkl")):
        system = f.stem
        run = load_run(coll, system)
        for (qid, cond, rate), ranked in run.items():
            m = metrics_for([int(x) for x in ranked if x >= 0], rel[qid])
            rows.append({"collection": coll, "system": system, "cond": cond, "rate": rate, "qid": qid,
                         "cluster": cluster[qid], "tags": tags[qid], "changed": changed[(qid, cond, rate)], **m})
    df = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    df.to_csv(RESULTS / f"{coll}_per_query.csv.gz", index=False)
    return df


def summary(df, coll):
    g = df.groupby(["system", "cond", "rate"])[["ndcg10", "mrr10", "hit1", "recall100"]].mean().reset_index()
    clean = g[g.cond == "clean"].set_index("system")
    for m in ["ndcg10", "mrr10", "hit1", "recall100"]:
        g[f"delta_{m}"] = g[m] - g.system.map(clean[m])
        g[f"retention_{m}"] = g[m] / g.system.map(clean[m])
    # changed-only subset (queries actually altered by the operator)
    ch = df[df.changed]
    cl = df[df.cond == "clean"].set_index(["system", "qid"])["ndcg10"]
    ch = ch.assign(clean_ndcg10=[cl[(s, q)] for s, q in zip(ch.system, ch.qid)])
    gc = ch.groupby(["system", "cond", "rate"]).agg(n_changed=("qid", "size"), ndcg10_changed=("ndcg10", "mean"),
                                                   clean_ndcg10_changed=("clean_ndcg10", "mean")).reset_index()
    gc["delta_ndcg10_changed"] = gc.ndcg10_changed - gc.clean_ndcg10_changed
    g = g.merge(gc, on=["system", "cond", "rate"], how="left")
    g.to_csv(RESULTS / f"{coll}_summary.csv", index=False, float_format="%.4f")
    return g


if __name__ == "__main__":
    for coll in sys.argv[1:]:
        df = per_query(coll)
        g = summary(df, coll)
        main = g[g.rate.isin([0.15]) | g.cond.str.startswith("real")]
        print(coll)
        print(main.pivot_table(index="system", columns="cond", values="ndcg10").round(4).to_string())
