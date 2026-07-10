#!/usr/bin/env python
"""Factorize a domain's operator/operand representation.

(A) two-way factorization H[operand, operator] = mu + operand + operator + interaction
    (variance shares + principal angles), raw residual and J-space, at the query and
    (control) operand reading positions.
(B) pure desinence: an exponent-free operator marker from a syncretic pair
    (relations only -- language/demonym emit the same word).
(C) held-out generalization: build v(op) from half the operands, swap on the other
    half. Generalization => a real operator, not interpolation.

    python scripts/operator_factorize.py 1.7b --domain relations
    python scripts/operator_factorize.py 8b  --domain logic
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from _common import band_layers, load_model, resolve_tag
from jlens import JacobianLens
import op_core

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    dom = op_core.load_domain(args.domain)
    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    ws = band_layers(lens.source_layers, model.n_layers)["workspace"]
    L = ws[len(ws) // 2]

    rep = op_core.guard_tokenization(dom, tok)
    print(f"[{args.domain}] {rep['operands_total']} operands, single-token frac "
          f"{rep['single_token_frac']:.2f}, min operator-pair signal {rep['min_pair_signal']}")

    def show(name, r):
        print(f"  {name:<22} stem={r['stem']:5.1%}  case={r['case']:5.1%}  "
              f"interaction={r['interaction']:5.1%}  angles={r['angles']}")

    print(f"\n(A) two-way factorization  [{tag}, {args.domain}, layer {L}]")
    print("  query position:")
    show("raw residual", op_core.factorize(model, lens, L, -1, dom, False))
    show("J-space", op_core.factorize(model, lens, L, -1, dom, True))
    print("  reading-position control (operand token, -2):")
    show("raw @ operand tok", op_core.factorize(model, lens, L, -2, dom, False))

    des = op_core.pure_desinence(model, ws, dom, tok)
    if des:
        print(f"\n(B) pure desinence {des['pair']} vs {des['other']} "
              f"(n_same={des['n_same']}): clean={des['clean']:+.2f} -> "
              f"+desinence={des['desinence']:+.2f}")

    print("\n(C) held-out generalization: build v(op) on half the operands, swap the other half")
    build, test, dfg = op_core.held_out_generalization(model, ws, dom, tok, seed=args.seed)
    print(f"  build on {len(build)} operands, test on {len(test)} held-out")
    print(f"  held-out swaps flipped: {(dfg['swap'] > 0).sum()}/{len(dfg)}; "
          f"mean swap={dfg['swap'].mean():+.2f} vs random={dfg['random'].mean():+.2f}")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_heldout.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    dfg.to_parquet(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
