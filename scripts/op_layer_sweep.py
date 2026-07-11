#!/usr/bin/env python
"""Where does the operator direction live, and where does it act?

The paper talks about a workspace *band*. That conflates two questions a reviewer
is right to separate:

  decodability  at which layer is the operator most READABLE? Measured two ways:
                (a) leave-one-operand-out nearest-direction classification -- build
                    v(op) from 11 operands at layer l, classify the held-out
                    operand's cells by cosine to those directions (chance = 1/n_ops);
                (b) the ANOVA operator variance share at layer l (the `case` term).
  causal        at which layer does injecting it most MOVE the model? The all-pairs
                swap contrast run with a SINGLE layer, at every layer.

If the two curves peaked together, "the operator lives in the workspace band" would
be the whole story. A divergence -- most decodable early, most causal late, or vice
versa -- is a result in its own right: readability and control are different
properties of the same direction, which is the paper's central distinction between
representation and behavioral influence, seen along the depth axis.

    python scripts/op_layer_sweep.py 1.7b --domain relations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import (band_layers, depth_percent, evenly_spaced_layers, load_model,
                     resolve_tag)
import op_core

ROOT = Path(__file__).resolve().parent.parent


def decodability(R, layer, dom, d_model):
    """Leave-one-operand-out operator classification at one layer, from the cached
    residual grid. For each held-out operand, build v(op) from the others and label
    each of its cells by the nearest (max cosine) operator direction, after removing
    that cell's mean over operators -- the same centering the direction is built
    with, so the classifier sees the operator contrast, not the operand identity."""
    ops, operands = dom.op_keys, dom.operand_keys
    hit = tot = 0
    for held in operands:
        rest = [o for o in operands if o != held]
        v = op_core.dirs_from_resids(R, [layer], ops, rest, [0], d_model)
        mean_l = sum(R[(held, k, 0)][layer] for k in ops) / len(ops)
        for k in ops:
            x = R[(held, k, 0)][layer] - mean_l
            pred = max(ops, key=lambda kk: op_core._cos(x, v[kk][layer]))
            hit += (pred == k)
            tot += 1
    return hit / tot


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(layers, model.n_layers)["workspace"]
    ops = dom.op_keys
    chance = 1.0 / len(ops)

    print(f"[{tag} / {args.domain}] layer sweep over {len(layers)} source layers "
          f"(workspace band {ws[0]}..{ws[-1]}); chance decodability {chance:.2f}")

    # one residual grid over ALL layers: decodability and the per-layer directions
    # both come from it, so the sweep costs one pass, not one pass per layer.
    R = op_core.op_resids(model, layers, dom, dom.operand_keys, [0])

    rows = []
    for l in layers:
        dec = decodability(R, l, dom, model.d_model)
        fac = op_core.factorize(model, None, l, -1, dom, False)
        v = op_core.dirs_from_resids(R, [l], ops, dom.operand_keys, [0], model.d_model)
        ldf = op_core.measure_swaps_long(model, [l], dom, tok, seed=args.seed,
                                         alpha=args.alpha, dirs=v)
        fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
        rows.append({"layer": int(l),
                     "depth": round(depth_percent(l, model.n_layers), 1),
                     "in_workspace": l in ws,
                     "decodability": dec,
                     "case_share": fac["case"],
                     "stem_share": fac["stem"],
                     "interaction_share": fac["interaction"],
                     "contrast": fam["contrast_mean"],
                     "contrast_lo": fam["contrast_lo"],
                     "contrast_hi": fam["contrast_hi"],
                     "flip_frac": fam["flip_frac"],
                     "swap": float(ldf["swap"].mean()),
                     "clean": float(ldf["clean"].mean())})
        r = rows[-1]
        print(f"  L{l:>2} ({r['depth']:>5.1f}%)  decode {dec:>5.1%}  "
              f"case {r['case_share']:>5.1%}  contrast {r['contrast']:>+7.2f} "
              f"[{r['contrast_lo']:+.1f},{r['contrast_hi']:+.1f}]  "
              f"flips {r['flip_frac']:.2f}{'  [ws]' if r['in_workspace'] else ''}")

    df = pd.DataFrame(rows)
    best_dec = df.loc[df["decodability"].idxmax()]
    best_case = df.loc[df["case_share"].idxmax()]
    best_cau = df.loc[df["contrast"].idxmax()]
    print(f"\n  peak decodability  L{int(best_dec['layer'])} "
          f"({best_dec['depth']:.1f}% depth, {best_dec['decodability']:.1%})")
    print(f"  peak case share    L{int(best_case['layer'])} "
          f"({best_case['depth']:.1f}% depth, {best_case['case_share']:.1%})")
    print(f"  peak causal effect L{int(best_cau['layer'])} "
          f"({best_cau['depth']:.1f}% depth, contrast {best_cau['contrast']:+.2f})")
    gap = float(best_cau["depth"] - best_dec["depth"])
    print(f"  decodability -> causal peak gap: {gap:+.1f} points of depth")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_layersweep.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    meta = {"tag": tag, "domain": args.domain, "alpha": args.alpha,
            "n_layers": int(model.n_layers), "chance_decodability": chance,
            "workspace": [int(l) for l in ws],
            "peak_decodability": {"layer": int(best_dec["layer"]),
                                  "depth": float(best_dec["depth"]),
                                  "value": float(best_dec["decodability"])},
            "peak_case_share": {"layer": int(best_case["layer"]),
                                "depth": float(best_case["depth"]),
                                "value": float(best_case["case_share"])},
            "peak_causal": {"layer": int(best_cau["layer"]),
                            "depth": float(best_cau["depth"]),
                            "value": float(best_cau["contrast"])},
            "depth_gap": gap,
            "bootstrap_unit": "operator (dyadic node), operands nested"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {out} (+ .json)")


if __name__ == "__main__":
    main()
