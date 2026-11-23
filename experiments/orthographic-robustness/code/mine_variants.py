"""Mine attested orthographic variants from the MIRACL-fa passage collection.

A pair (w, w') is kept when w' is produced from w by one purely orthographic
alternation (no lexical change) and BOTH forms occur at least MIN_FREQ times
in the collection. Alternations:
  hamza seats   ئ->ی, ئ->ا, ئ->و, أ->ا, ؤ->و, إ->ا
  alef madda    آ->ا
  heh-yeh       ۀ->ه , ة->ه
  arabic forms  ی->ي , ک->ك  (occurring naturally in the collection)
  (substitution variants must also be rarer than the source form; ئ->ا is
   restricted to the ئت / ئله contexts, e.g. هیئت/هیات, مسئله/مساله)
  ZWNJ          w with ZWNJ -> joined form (unigram) or spaced form (bigram)
The rewriter's hand-written spelling table is NOT used, so the resulting
noise model is independent of the query-rewriting system under test.

Author: Mohammad Naser Sadrmanesh
"""
import json, re
from collections import Counter
from common import DATA, read_jsonl, RESULTS

MIN_FREQ = 5
ZWNJ = "\u200c"
TOK = re.compile(r"[\u0600-\u06FF\u200c]+")
SUBS = [("ئ", "ی"), ("ئ", "ا"), ("ئ", "و"), ("أ", "ا"), ("ؤ", "و"), ("إ", "ا"),
        ("آ", "ا"), ("ۀ", "ه"), ("ة", "ه"), ("ی", "ي"), ("ک", "ك")]


def main():
    uni, bi = Counter(), Counter()
    for d in read_jsonl(DATA / "miracl_corpus.jsonl"):
        toks = TOK.findall(d["text"])
        uni.update(toks)
        bi.update(zip(toks, toks[1:]))
    variants = {}
    stats = Counter()
    for w, n in uni.items():
        if n < MIN_FREQ:
            continue
        alts = []
        for a, b in SUBS:
            if a in w:
                if (a, b) == ("ئ", "ا") and not re.search(r"ئ(ت|له)", w):
                    continue  # elsewhere ئ->ا mostly yields a different word (سوئد/سواد)
                w2 = w.replace(a, b)
                # the variant must be attested AND rarer than the source form
                if MIN_FREQ <= uni.get(w2, 0) < n:
                    alts.append((w2, uni[w2], f"{a}->{b}"))
        if ZWNJ in w:
            parts = w.split(ZWNJ)
            joined = "".join(parts)
            if uni.get(joined, 0) >= MIN_FREQ:
                alts.append((joined, uni[joined], "zwnj->joined"))
            if len(parts) == 2 and bi.get(tuple(parts), 0) >= MIN_FREQ:
                alts.append((" ".join(parts), bi[tuple(parts)], "zwnj->space"))
        if alts:
            variants[w] = sorted(alts, key=lambda x: -x[1])
            for a in alts:
                stats[a[2]] += 1
    json.dump(variants, open(DATA / "attested_variants.json", "w", encoding="utf-8"), ensure_ascii=False)
    # overlap with the rewriter table
    from persian_rag.rewrite import DEFAULT_SPELLING_GROUPS
    table = {w for g in DEFAULT_SPELLING_GROUPS for w in g}
    overlap = sorted(w for w in variants if w in table)
    # natural noise already present in MIRACL / PQuAD queries
    nat = {}
    for coll in ("miracl", "pquad"):
        qs = read_jsonl(DATA / f"{coll}_queries.jsonl")
        nat[coll] = {
            "n": len(qs),
            "has_zwnj": sum(ZWNJ in q["text"] for q in qs),
            "has_spaced_mi": sum(bool(re.search(r"(^|\s)(ن?می)\s", q["text"])) for q in qs),
            "has_arabic_yeh_kaf": sum(("ي" in q["text"]) or ("ك" in q["text"]) for q in qs),
            "has_alef_madda": sum("آ" in q["text"] for q in qs),
            "has_hamza": sum(any(c in q["text"] for c in "ئأؤإ") for q in qs),
        }
    out = {"min_freq": MIN_FREQ, "n_words_with_variants": len(variants), "by_rule": dict(stats),
           "overlap_with_rewriter_table": overlap, "natural_query_noise": nat}
    RESULTS.mkdir(exist_ok=True)
    json.dump(out, open(RESULTS / "attested_variants_stats.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
