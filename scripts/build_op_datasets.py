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


def main():
    assert single("True") and single("False") and ft("True") != ft("False"), \
        "True/False must be single distinct tokens"
    for name, obj in (("arithmetic", build_arithmetic()), ("logic", build_logic()),
                      ("arith_addN", build_arith_addN())):
        path = ROOT / "data" / f"{name}.json"
        path.write_text(json.dumps(obj, indent=2))
        n = len(obj["items"])
        print(f"wrote {path.relative_to(ROOT)}: {n} operands, "
              f"{len(obj['meta']['operators'])} operators")
        # show a couple
        for k in list(obj["items"])[:3]:
            print(f"    {k}: {obj['items'][k]['answers']}")


if __name__ == "__main__":
    main()
