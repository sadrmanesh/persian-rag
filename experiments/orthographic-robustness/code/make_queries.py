"""Generate perturbed query sets for every collection.

Conditions: clean, char, zwnj, typo(0.15), comp(0.15), dose-response typo/comp
at EditRate {0.05,0.10,0.20,0.30} (MIRACL and PQuAD), and 'real' (attested
corpus variants, see mine_variants.py). Seed 42.

Author: Mohammad Naser Sadrmanesh
"""
import json, re
from common import DATA, SEED, read_jsonl, write_jsonl
from perturb import perturb

RATES = [0.05, 0.10, 0.15, 0.20, 0.30]
_EDGE = "؟?!.,،؛;:«»\"'()"


REAL_GROUPS = {
    "real": None,                                            # every attested alternation
    "real_ortho": {"ئ->ی", "ئ->ا", "ئ->و", "أ->ا", "ؤ->و", "إ->ا", "آ->ا", "ۀ->ه", "ة->ه"},
    "real_arabic": {"ی->ي", "ک->ك"},
    "real_zwnj": {"zwnj->joined", "zwnj->space"},
}


def t_real(q, variants, rules=None):
    """Replace every token that has an attested variant by its most frequent variant."""
    out, changed = [], 0
    for tok in q.split(" "):
        core = tok.strip(_EDGE)
        alts = [a for a in variants.get(core, []) if rules is None or a[2] in rules]
        if alts:
            alt = alts[0][0]
            tok = tok.replace(core, alt)
            changed += 1
        out.append(tok)
    return " ".join(out), changed


def conditions(coll):
    conds = [("clean", 0.15), ("char", 0.15), ("zwnj", 0.15), ("typo", 0.15), ("comp", 0.15)]
    if coll in ("miracl", "pquad"):
        conds += [(op, r) for op in ("typo", "comp") for r in RATES if r != 0.15]
    return conds


def main():
    variants = json.load(open(DATA / "attested_variants.json", encoding="utf-8"))
    for coll in ("gold", "pquad", "miracl"):
        qs = read_jsonl(DATA / f"{coll}_queries.jsonl")
        rows = []
        for q in qs:
            for op, r in conditions(coll):
                t = perturb(q["text"], q["id"], op, SEED, r)
                rows.append({"qid": q["id"], "cond": op, "rate": r, "text": t, "changed": t != q["text"]})
            for name, rules in REAL_GROUPS.items():
                t, n = t_real(q["text"], variants, rules)
                rows.append({"qid": q["id"], "cond": name, "rate": 0.0, "text": t, "changed": n > 0})
        write_jsonl(DATA / f"{coll}_queries_perturbed.jsonl", rows)
        from collections import Counter
        c = Counter((r["cond"], r["rate"]) for r in rows if r["changed"])
        print(coll, len(qs), dict(c))


if __name__ == "__main__":
    main()
