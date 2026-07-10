#!/usr/bin/env python
"""Injector test: does the country->currency manifold reroute under injection?

Baseline: the model does "boot-shaped country" -> Italy -> euro. We then INJECT
a different country at the intermediate ("Italy") node, across the workspace
band, and read the model's actual output. If the J-space is a relational
manifold, injecting "United States" should route down the same country->currency
channel and produce "dollar"; "Japan" -> "yen"; etc.

    python scripts/inject.py 1.7b --from Italy --to "United States" Japan France China
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _common import MODELS, depth_percent, load_model, resolve_tag
from causal_swap import SwapHooks, first_token, jlens_dir
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent
PROMPT = "Fact: The currency used in the country shaped like a boot is"


@torch.no_grad()
def top_out(model, prompt, k=6):
    ids = model.encode(prompt, max_length=128)
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0, -1]
    logits = model.unembed(h[None])[0]
    return [model.tokenizer.decode([t]).strip() for t in logits.topk(k).indices.tolist()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--from", dest="src", default="Italy")
    ap.add_argument("--to", nargs="+", default=["United States", "Japan", "France"])
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--band", default="workspace", choices=["early", "workspace", "late"])
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    model = (__import__("int8_model").load_int8_model(key) if args.int8
             else load_model(key))
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    W_U = model._lm_head.weight
    dev = W_U.device
    from causal_swap import BANDS
    lo, hi = BANDS[args.band]
    layers = [l for l in lens.source_layers if lo <= depth_percent(l, model.n_layers) <= hi]

    print(f"prompt: {args.prompt!r}\nband: {args.band} ({len(layers)} layers)\n")
    print(f"{'injection':>22} | model output (top-6)")
    print("-" * 78)
    print(f"{'(baseline)':>22} | {', '.join(top_out(model, args.prompt))}")

    ti = first_token(tok, args.src)
    for target in args.to:
        tt = first_token(tok, target)
        dirs = {l: (jlens_dir(lens, W_U, l, ti, dev), jlens_dir(lens, W_U, l, tt, dev))
                for l in layers}
        with SwapHooks(model, dirs):
            out = top_out(model, args.prompt)
        print(f"{args.src + ' -> ' + target:>22} | {', '.join(out)}")


if __name__ == "__main__":
    main()
