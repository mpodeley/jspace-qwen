#!/usr/bin/env python
"""Lens-quality eval: does the lens surface the hidden bridge entity?

Uses the paper's shipped multihop set (jacobian-lens/data/evaluations). For each
two-hop prompt we read out at the token before `target` across all fitted layers
and ask whether each `intermediate` (the unstated bridge entity, e.g. "Brazil")
appears at lens rank <= k for its best layer. pass@k = mean over items of the
fraction of intermediates that hit. Reported for the J-lens and the logit-lens
baseline, so the gap is the J-lens' advantage — and comparing across model sizes
is the scale story.

    python scripts/lens_eval.py 1.7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import MODELS, load_model, resolve_tag
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "jacobian-lens" / "data" / "evaluations"


def first_token(tok, word: str) -> int:
    ids = tok.encode(" " + word.strip(), add_special_tokens=False)
    return ids[0]


@torch.no_grad()
def ranks_at_last_pos(model, lens, prompt, *, use_jacobian):
    """min-over-layers rank of every vocab token at the last prompt position."""
    lens_logits, _, _ = lens.apply(
        model, prompt, layers=lens.source_layers, positions=[-1],
        use_jacobian=use_jacobian,
    )
    # stack layers -> [n_layers, vocab]; rank = # tokens strictly greater
    stack = torch.stack([lens_logits[l][0] for l in lens.source_layers])  # [L, V]
    # per-layer rank of each token, then min over layers
    order = stack.argsort(dim=-1, descending=True)  # [L, V]
    ranks = torch.empty_like(order)
    ar = torch.arange(stack.shape[-1])
    for i in range(stack.shape[0]):
        ranks[i, order[i]] = ar
    return ranks.min(dim=0).values  # [V] best (lowest) rank across layers


def evaluate(model, lens, items, ks=(1, 5, 10)):
    tok = model.tokenizer
    out = {f"jlens_pass@{k}": [] for k in ks}
    out.update({f"logit_pass@{k}": [] for k in ks})
    for it in items:
        for use_j, pre in ((True, "jlens"), (False, "logit")):
            best = ranks_at_last_pos(model, lens, it["prompt"], use_jacobian=use_j)
            fracs = {k: [] for k in ks}
            for w in it["intermediates"]:
                r = int(best[first_token(tok, w)])
                for k in ks:
                    fracs[k].append(1.0 if r < k else 0.0)
            for k in ks:
                out[f"{pre}_pass@{k}"].append(sum(fracs[k]) / max(len(fracs[k]), 1))
    return {m: sum(v) / max(len(v), 1) for m, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--set", default="lens-eval-multihop")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    items = json.loads((DATA / f"{args.set}.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(key)
    else:
        model = load_model(key)
    lens = JacobianLens.from_pretrained(str(lens_path))

    res = evaluate(model, lens, items)
    res["model"] = tag
    res["n_items"] = len(items)
    print(pd.Series(res).to_string())

    out = ROOT / "results" / "metrics" / f"{tag}_lenseval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
