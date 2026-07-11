#!/usr/bin/env python
"""Author data/arithmetic.json and data/logic.json in the unified op_core schema,
screening every answer against the Qwen3 tokenizer. CPU only.

Arithmetic: operators plus/times/minus over single-digit operand pairs, constrained
so every result is a single-digit (0-9) token and the three results are pairwise
first-token distinct (else the swap metric is silently null). Logic: comparison
operators greater/less/equal over number pairs, answers True/False (a large operand
space where the operator maps the same operand to different truth values, unlike
and/or over 2 booleans which is degenerate)."""

from __future__ import annotations

import json
from pathlib import Path

import transformers

ROOT = Path(__file__).resolve().parent.parent
tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")


def single(w, sp=True):
    p = " " if sp else ""
    return len(tok.encode(p + str(w).strip(), add_special_tokens=False)) == 1


def ft(w):
    return tok.encode(" " + str(w).strip(), add_special_tokens=False)[0]


def single_digit(v):
    # Qwen3 tokenizes bare digits as single tokens (" 3" is [space,3]); score bare.
    return len(tok.encode(str(v), add_special_tokens=False)) == 1


def build_arithmetic():
    ops = {"plus": "plus", "times": "times", "minus": "minus"}
    items = {}
    for a in range(1, 10):
        for b in range(0, a + 1):  # a >= b so minus is non-negative
            res = {"plus": a + b, "times": a * b, "minus": a - b}
            if any(v > 9 or v < 0 for v in res.values()):
                continue
            if len({res["plus"], res["times"], res["minus"]}) < 3:
                continue  # need all three answers distinct
            if not all(single_digit(v) for v in res.values()):
                continue
            key = f"{a},{b}"
            items[key] = {"args": {"a": str(a), "b": str(b)},
                          "answers": {k: str(v) for k, v in res.items()}}
    return {
        "meta": {
            "domain": "arithmetic",
            "template": "The result of {a} {op} {b} is",
            "operators": ops,
            "operand_slots": ["a", "b"],
            "answer_leading_space": False,
            "note": "single-digit results only (BPE splits multi-digit numbers); "
                    "operands with plus==times etc. dropped so every operator pair "
                    "is distinct; answers scored as bare digit tokens (Qwen3 splits "
                    "' 3' into [space,3])",
        },
        "items": items,
    }


def build_logic():
    # comparison operators over number pairs -> True/False. Large operand space;
    # the operator maps a fixed operand pair to different truth values.
    ops = {"greater": "greater than", "less": "less than", "equal": "equal to"}
    def ans(a, b):
        return {"greater": "True" if a > b else "False",
                "less": "True" if a < b else "False",
                "equal": "True" if a == b else "False"}
    items = {}
    pairs = [(5, 3), (3, 5), (7, 2), (2, 7), (4, 4), (8, 1), (1, 8),
             (6, 6), (9, 4), (4, 9), (2, 2), (3, 8)]
    for a, b in pairs:
        key = f"{a},{b}"
        items[key] = {"args": {"a": str(a), "b": str(b)}, "answers": ans(a, b)}
    return {
        "meta": {
            "domain": "logic",
            "template": "The claim that {a} is {op} {b} is",
            "operators": ops,
            "operand_slots": ["a", "b"],
            "note": "comparison operators over number pairs; answers True/False are "
                    "single distinct tokens. and/or over 2 booleans is degenerate "
                    "(operator entangled with answer), so we use comparisons instead.",
        },
        "items": items,
    }


def build_arith_addN():
    # Christ et al. (2510.26543) cut: 'add-N' as a RELATION, N the parameter, a
    # single number operand. Their arithmetic generalizes across held-out N -- but
    # here N is effectively an OPERAND (a number, linearly represented), not a
    # distinct operation, so 'generalizing across N' is number-line interpolation,
    # not operator structure. Contrast with our +/x/- cut, which varies the actual
    # function and does NOT generalize. Same space, different axis.
    Ns = [1, 2, 3, 4]
    ops = {str(n): str(n) for n in Ns}          # operator phrase is the addend N
    items = {}
    for a in range(1, 6):                        # operand; a + max(N) <= 9
        res = {str(n): a + n for n in Ns}
        if not all(single_digit(v) for v in res.values()):
            continue
        items[str(a)] = {"args": {"a": str(a)},
                         "answers": {k: str(v) for k, v in res.items()}}
    return {
        "meta": {
            "domain": "arith_addN",
            "template": "The result of {a} plus {op} is",
            "operators": ops,
            "operand_slots": ["a"],
            "answer_leading_space": False,
            "note": "add-N as a relation (Christ et al. 2510.26543 cut): the operator "
                    "IS the addend N, the operand is a single number. N is really a "
                    "linear numeric quantity, so swaps generalize -- unlike +/x/- over "
                    "two operands, which vary the function and do not.",
        },
        "items": items,
    }


def build_animals():
    """A non-geographic domain (reviewer round 2): animal -> taxonomic attribute.
    Curated for the same discipline as relations -- stable, 1-to-1, atemporal
    answers; ambiguous or many-to-many relations excluded. Four operators, twelve
    animals, three paraphrase frames mirroring relations.json. No desinence pair
    (animals have no natural syncretism).

    The screening GATE below refuses to write unless, for every operand, the four
    answers are pairwise first-token distinct (else that operand carries no swap
    signal), matching the arithmetic/logic discipline. Multi-token answers
    (mammal/reptile/carnivore ...) are allowed -- the efficacy metric scores the
    first token of a logit difference -- but depress the greedy exact-match ceiling,
    so we print single-token coverage for the paper to report (as with Gemma)."""
    ops = {"class": "class", "habitat": "habitat", "diet": "diet",
           "covering": "covering"}
    templates = [
        "The {op} of {a} is",
        "Q: What is the {op} of {a}? A: It is",
        "It is well known that the {op} of {a} is",
    ]
    grid = {
        "shark":   {"class": "fish",      "habitat": "ocean",    "diet": "carnivore",   "covering": "scales"},
        "eagle":   {"class": "bird",      "habitat": "mountain", "diet": "carnivore",   "covering": "feathers"},
        "frog":    {"class": "amphibian", "habitat": "pond",     "diet": "insectivore", "covering": "skin"},
        "snake":   {"class": "reptile",   "habitat": "desert",   "diet": "carnivore",   "covering": "scales"},
        "bee":     {"class": "insect",    "habitat": "hive",     "diet": "herbivore",   "covering": "hair"},
        "whale":   {"class": "mammal",    "habitat": "ocean",    "diet": "carnivore",   "covering": "skin"},
        "owl":     {"class": "bird",      "habitat": "forest",   "diet": "carnivore",   "covering": "feathers"},
        "lizard":  {"class": "reptile",   "habitat": "desert",   "diet": "insectivore", "covering": "scales"},
        "cow":     {"class": "mammal",    "habitat": "farm",     "diet": "herbivore",   "covering": "hair"},
        "salmon":  {"class": "fish",      "habitat": "river",    "diet": "carnivore",   "covering": "scales"},
        "penguin": {"class": "bird",      "habitat": "ice",      "diet": "carnivore",   "covering": "feathers"},
        "deer":    {"class": "mammal",    "habitat": "forest",   "diet": "herbivore",   "covering": "fur"},
    }
    op_keys = list(ops)
    # GATE 1: within each operand, the four answers first-token distinct.
    bad = []
    for a, ans in grid.items():
        fts = {k: ft(ans[k]) for k in op_keys}
        if len(set(fts.values())) < len(op_keys):
            bad.append((a, {k: ans[k] for k in op_keys}))
    if bad:
        raise SystemExit(f"animals screening FAILED (first-token collisions): {bad}")
    # GATE 2: every operator pair has signal on >= 8 of 12 operands.
    import itertools
    weak = []
    for ka, kb in itertools.combinations(op_keys, 2):
        n = sum(1 for a in grid if ft(grid[a][ka]) != ft(grid[a][kb]))
        if n < 8:
            weak.append((ka, kb, n))
    if weak:
        raise SystemExit(f"animals screening FAILED (weak operator pairs): {weak}")
    items = {a: {"args": {"a": a}, "answers": grid[a]} for a in grid}
    return {
        "meta": {
            "domain": "animals",
            "template": templates[0],
            "templates": templates,
            "operators": ops,
            "operand_slots": ["a"],
            "note": "animal -> taxonomic attribute (class/habitat/diet/covering); "
                    "curated for stable atemporal 1-to-1 answers, ambiguous relations "
                    "excluded; multi-token answers allowed (first-token logit metric), "
                    "single-token coverage reported for the greedy ceiling.",
        },
        "items": items,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None,
                    help="subset of {arithmetic,logic,arith_addN,animals} to (re)write; "
                         "default writes all EXCEPT animals to avoid clobbering")
    args = ap.parse_args()
    assert single("True") and single("False") and ft("True") != ft("False"), \
        "True/False must be single distinct tokens"
    builders = {"arithmetic": build_arithmetic, "logic": build_logic,
                "arith_addN": build_arith_addN, "animals": build_animals}
    default = ["arithmetic", "logic", "arith_addN"]  # animals only on explicit --only
    names = args.only if args.only else default
    for name in names:
        obj = builders[name]()
        path = ROOT / "data" / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2))
        n = len(obj["items"])
        # single-token coverage across the whole grid (the greedy-ceiling caveat)
        cells = [(a, k) for a in obj["items"] for k in obj["meta"]["operators"]]
        cov = sum(1 for a, k in cells
                  if single(obj["items"][a]["answers"][k])) / len(cells)
        print(f"wrote {path.relative_to(ROOT)}: {n} operands, "
              f"{len(obj['meta']['operators'])} operators, "
              f"single-token coverage {cov:.2f}")
        for k in list(obj["items"])[:3]:
            print(f"    {k}: {obj['items'][k]['answers']}")


if __name__ == "__main__":
    main()
