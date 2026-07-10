#!/usr/bin/env python
"""Operator paradigm: are the operators of a domain a structured set of causally
manipulable directions over the operands?

(A) all-pairs operator swap efficacy with a matched-norm random control, and
(B) paradigm geometry (case-direction cosines) in the J-lens vs logit-lens readout.

Domain-general via op_core; pick the domain (relations / arithmetic / logic).

    python scripts/operator_paradigm.py 1.7b --domain relations
    python scripts/operator_paradigm.py 8b  --domain arithmetic
"""

from __future__ import annotations

import argparse
import statistics as st
from pathlib import Path

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
from jlens import JacobianLens
import op_core

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--geometry", action="store_true",
                    help="also compute J-space vs logit-space readout geometry (needs a J-lens)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    dom = op_core.load_domain(args.domain)

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    tok = model.tokenizer

    # The causal core (swaps, factorization) is lens-free; only the readout
    # geometry needs a fitted J-lens. This lets a second model run without a fit.
    lens = None
    if args.geometry:
        lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
        lens = JacobianLens.from_pretrained(str(lens_path))
    source_layers = lens.source_layers if lens else evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]

    rep = op_core.guard_tokenization(dom, tok)
    print(f"[{args.domain}] {rep['operands_total']} operands, single-token frac "
          f"{rep['single_token_frac']:.2f}, min operator-pair signal {rep['min_pair_signal']} operands")

    print(f"\n(A) operator swap  [{tag}, {args.domain}]  clean<0; working swap>0; random~0")
    df = op_core.measure_swaps(model, ws, dom, tok, seed=args.seed)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    valid = df.dropna(subset=["swap"])
    print(f"  swaps that flipped sign: {(valid['swap'] > 0).sum()}/{len(valid)}; "
          f"random flipped: {(valid['random'] > 0).sum()}/{len(valid)}; "
          f"mean swap={valid['swap'].mean():+.2f} vs mean random={valid['random'].mean():+.2f}")

    if lens is not None:
        print("\n(B) paradigm geometry: mean |off-diagonal cosine| (lower = more distinct)")
        for name, uj in (("J-space (unembed(J h))", True), ("logit-space (unembed(h))", False)):
            _, offs = op_core.measure_geometry(model, lens, ws, dom, uj)
            mabs = st.mean(abs(o[2]) for o in offs)
            syn = max(offs, key=lambda o: o[2])
            print(f"  {name:<26} mean|off|={mabs:.3f}   most-syncretic: "
                  f"{syn[0]}~{syn[1]} cos={syn[2]:+.2f}")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_operator_swap.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
