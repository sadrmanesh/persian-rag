"""Query perturbation operators: T_char, T_zwnj, T_typo, T_comp.

Every call is deterministic given (seed, query id, operator name, edit rate):
a private RNG is derived from these with SHA-256, so results do not depend on
the order in which queries are processed.

Author: Mohammad Naser Sadrmanesh
"""
from __future__ import annotations

import hashlib
import random

ZWNJ = "\u200c"
PERSIAN_LETTERS = set("ابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهیآ")

# ---- T_char: Arabic-keyboard / legacy encodings
CHAR_MAP = str.maketrans({"ک": "ك", "ی": "ي", "آ": "ا"})


def t_char(q: str, rng=None) -> str:
    return q.translate(CHAR_MAP)


# ---- T_zwnj: each ZWNJ deleted (p=.5) or replaced with a space (p=.5)
def t_zwnj(q: str, rng: random.Random) -> str:
    out = []
    for ch in q:
        if ch == ZWNJ:
            out.append("" if rng.random() < 0.5 else " ")
        else:
            out.append(ch)
    return "".join(out)


# ---- T_typo: ISIRI 9147 keyboard adjacency + homophones
ROWS = [
    ("ضصثقفغعهخحجچ", 0.0),
    ("شسیبلاتنمکگ", 0.25),
    ("ظطزرذدپو", 0.75),
]
_POS = {}
for r, (letters, off) in enumerate(ROWS):
    for i, ch in enumerate(letters):
        _POS[ch] = (r, i + off)
ADJ = {}
for a, (ra, xa) in _POS.items():
    ADJ[a] = [b for b, (rb, xb) in _POS.items()
              if b != a and abs(ra - rb) <= 1 and abs(xa - xb) <= 1.0]
ADJ["آ"] = ADJ.get("ا", [])
ADJ["ژ"] = ADJ.get("ز", [])
# Arabic forms produced by T_char remain editable inside T_comp
ADJ["ي"] = ADJ["ی"]
ADJ["ك"] = ADJ["ک"]
PERSIAN_LETTERS |= {"ي", "ك"}

HOMOPHONES = [set("تط"), set("سصث"), set("زذضظ"), set("هح"), set("قغ"), set("اع")]
HOMO = {}
for g in HOMOPHONES:
    for ch in g:
        HOMO[ch] = sorted(g - {ch})

OPS = ("insert", "delete", "transpose", "homophone", "keyboard")


def _edit_token(tok: str, rng: random.Random) -> str:
    idx = [i for i, c in enumerate(tok) if c in PERSIAN_LETTERS or c in ADJ]
    if not idx:
        return tok
    op = rng.choice(OPS)
    i = rng.choice(idx)
    c = tok[i]
    if op == "delete" and len(tok) > 1:
        return tok[:i] + tok[i + 1:]
    if op == "transpose" and len(tok) > 1:
        j = i + 1 if i + 1 < len(tok) else i - 1
        a, b = sorted((i, j))
        return tok[:a] + tok[b] + tok[a] + tok[b + 1:]
    if op == "homophone" and c in HOMO:
        return tok[:i] + rng.choice(HOMO[c]) + tok[i + 1:]
    if op == "insert" and ADJ.get(c):
        return tok[:i + 1] + rng.choice(ADJ[c]) + tok[i + 1:]
    # keyboard substitution (also the fallback when an op is not applicable)
    if ADJ.get(c):
        return tok[:i] + rng.choice(ADJ[c]) + tok[i + 1:]
    return tok


def t_typo(q: str, rng: random.Random, edit_rate: float = 0.15) -> str:
    toks = q.split(" ")
    out = []
    for t in toks:
        if len(t) >= 2 and any(c in PERSIAN_LETTERS for c in t) and rng.random() < edit_rate:
            out.append(_edit_token(t, rng))
        else:
            out.append(t)
    return " ".join(out)


def t_comp(q: str, rng: random.Random, edit_rate: float = 0.15) -> str:
    return t_typo(t_zwnj(t_char(q), rng), rng, edit_rate)


def rng_for(seed: int, qid: str, op: str, rate: float) -> random.Random:
    h = hashlib.sha256(f"{seed}|{qid}|{op}|{rate:.3f}".encode()).hexdigest()
    return random.Random(int(h[:16], 16))


def perturb(q: str, qid: str, op: str, seed: int = 42, edit_rate: float = 0.15) -> str:
    if op == "clean":
        return q
    rng = rng_for(seed, qid, op, edit_rate)
    if op == "char":
        return t_char(q)
    if op == "zwnj":
        return t_zwnj(q, rng)
    if op == "typo":
        return t_typo(q, rng, edit_rate)
    if op == "comp":
        return t_comp(q, rng, edit_rate)
    raise ValueError(op)
