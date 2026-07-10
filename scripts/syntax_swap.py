#!/usr/bin/env python
"""Causal test of the pointer-node hypothesis: rerouting the workspace node
reroutes the DOWNSTREAM morphological realization.

The a/an test (run_aan). English "a"/"an" is a purely phonological consequence of
the *next* noun's onset, so the determiner can only be chosen AFTER the concept
pointer is dereferenced into a specific noun. If the workspace holds a
morphology-free lemma pointer, then rewriting that pointer (cat -> elephant) in
the workspace band should flip the determiner the model is about to emit
(" a" -> " an") -- even though we never touched the determiner or any late layer.

This reuses the exact machinery of the semantic swap:
  jlens_dir  (causal_swap.py) : the pointer v_{l,t} = normalize(J_l^T W_U[t])
  SwapHooks  (causal_swap.py) : project out v_concept, add v_swap over a band
  greedy_next(causal_swap.py) : the model's actual next token (for screening)

Controls:
  - behavioural screen: only score items whose clean next token is the determiner
  - matched-norm random swap: v_to is a random unit direction, same projection
    magnitude, so a determiner flip cannot be a generic perturbation effect
  - verify the swap took: under the concept swap, the continuation should move
    toward swap_to (checked by forcing the determiner and reading the noun)

    python scripts/syntax_swap.py 1.7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import BANDS, band_layers, first_token, load_model, resolve_tag
from causal_swap import SwapHooks, greedy_next, jlens_dir
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent
MORPH = ROOT / "data" / "morphosyntax"


@torch.no_grad()
def final_logits_last(model, prompt, hooks=None):
    """Model's own final-layer logits at the last position, optionally under a
    forward-hook intervention. This is the real next-token distribution, not a
    lens readout -- the claim is about what the model does."""
    ids = model.encode(prompt, max_length=128)
    final = model.n_layers - 1
    ctx = hooks if hooks is not None else _null_ctx()
    with ctx, ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0, -1]
    return model.unembed(h[None])[0]


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def norm_logit_diff(logits, id_a: int, id_b: int) -> float:
    la, lb = float(logits[id_a]), float(logits[id_b])
    return (la - lb) / (abs(la) + abs(lb) + 1e-9)


def build_swap(lens, W_U, layers, tok_from: int, tok_to: int, dev):
    """{layer: (v_from, v_to)} pointer directions over a band."""
    return {l: (jlens_dir(lens, W_U, l, tok_from, dev),
                jlens_dir(lens, W_U, l, tok_to, dev)) for l in layers}


def build_ctrl(lens, W_U, layers, tok_from: int, dev, seed):
    """Matched-norm control: same v_from, but v_to is a random unit direction."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l in layers:
        vf = jlens_dir(lens, W_U, l, tok_from, dev)
        r = torch.randn(vf.shape, generator=g).to(dev)
        out[l] = (vf, r / (r.norm() + 1e-8))
    return out


@torch.no_grad()
def run_aan(model, lens, items, band="workspace", seed=0):
    tok = model.tokenizer
    W_U = model._lm_head.weight
    dev = W_U.device
    layers = band_layers(lens.source_layers, model.n_layers)[band]
    id_a = first_token(tok, "a")
    id_an = first_token(tok, "an")

    rows = []
    n_screened = 0
    for it in items:
        # 1. behavioural screen: clean next token must be the determiner
        clean = greedy_next(model, it["prompt"])
        det_id = first_token(tok, it["det"])
        if clean != det_id:
            continue
        n_screened += 1

        tok_from = first_token(tok, it["concept"])
        tok_to = first_token(tok, it["swap_to"])
        swap = build_swap(lens, W_U, layers, tok_from, tok_to, dev)
        ctrl = build_ctrl(lens, W_U, layers, tok_from, dev, seed)

        # a>an is positive when the model favors " a"; the swap should push it the
        # other way (concept "a"-onset -> swap_to "an"-onset means d should drop).
        d_clean = norm_logit_diff(final_logits_last(model, it["prompt"]), id_a, id_an)
        with SwapHooks(model, swap):
            d_swap = norm_logit_diff(final_logits_last(model, it["prompt"]), id_a, id_an)
            g_swap = greedy_next(model, it["prompt"])
        with SwapHooks(model, ctrl):
            d_ctrl = norm_logit_diff(final_logits_last(model, it["prompt"]), id_a, id_an)
            g_ctrl = greedy_next(model, it["prompt"])

        # did the determiner flip to the swap partner's determiner?
        swap_det_id = first_token(tok, it["swap_det"])
        rows.append({
            "name": it["name"], "det": it["det"], "swap_det": it["swap_det"],
            "d_clean": d_clean, "d_swap": d_swap, "d_ctrl": d_ctrl,
            "flip_swap": int(g_swap == swap_det_id),
            "flip_ctrl": int(g_ctrl == swap_det_id),
            "moved_swap": int(g_swap != clean),
            "moved_ctrl": int(g_ctrl != clean),
        })
    return pd.DataFrame(rows), n_screened


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--band", default="workspace", choices=list(BANDS))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    items = json.loads((MORPH / "aan-determiner.json").read_text())["items"]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    lens = JacobianLens.from_pretrained(str(lens_path))

    df, n_screened = run_aan(model, lens, items, band=args.band, seed=args.seed)
    n_total = len(items)
    print(f"\na/an test  [{tag}, band={args.band}, lens={lens_path.stem}]")
    print(f"  screened clean: {n_screened}/{n_total}  scored: {len(df)}")
    if len(df):
        # the determiner should flip under the pointer swap, but not the control
        print(f"  determiner flip rate  swap={df['flip_swap'].mean():.3f}  "
              f"control={df['flip_ctrl'].mean():.3f}")
        print(f"  a-vs-an logit diff     clean={df['d_clean'].mean():+.3f}  "
              f"swap={df['d_swap'].mean():+.3f}  control={df['d_ctrl'].mean():+.3f}")
        print(f"  (swap moved next tok: {df['moved_swap'].mean():.2f}, "
              f"control moved: {df['moved_ctrl'].mean():.2f})")

    out = ROOT / "results" / "ablation" / f"{tag}_aan_{args.band}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
