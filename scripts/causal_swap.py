#!/usr/bin/env python
"""Causal test: is the J-space representation actually steering the answer?

Reuses the paper's probe-swap set (90 two-hop prompts). Each item's model
should do a hidden two-hop: prompt -> intermediate (e.g. "Brazil") -> answer
("Portuguese"). We intervene in the residual stream across a layer *band*:
at every position we move the component lying along the intermediate's J-lens
direction onto the swap entity's J-lens direction (intermediate -> swap_to,
e.g. Brazil -> Mexico), then check whether the greedy answer flips to
`swap_answer` ("Spanish").

The J-lens direction for token t at layer l is v_{l,t} = J_l^T W_U[t] (the
residual direction that most raises t's lens readout). Conditions:
  clean   no intervention          -> baseline two-hop accuracy (== answer)
  swap    intermediate -> swap_to  -> flip rate (== swap_answer)
  control intermediate -> random   -> same norm change, random target (should NOT flip)

Run over early / workspace / late bands. The workspace band flipping (while
control does not, and while early/late flip less) is the causal signature of
the J-space. Comparing across model sizes is the scale story.

    python scripts/causal_swap.py 1.7b
    python scripts/causal_swap.py 32b --int8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import MODELS, band_layers, first_token, load_model, resolve_tag
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "jacobian-lens" / "data" / "experiments"


def jlens_dir(lens: JacobianLens, W_U: torch.Tensor, layer: int, tok_id: int, dev):
    """Unit residual direction v_{l,t} = J_l^T W_U[t]."""
    u = W_U[tok_id].float().to(dev)
    v = lens.jacobians[layer].to(dev).t() @ u
    return v / (v.norm() + 1e-8)


class SwapHooks:
    """Forward hooks that swap intermediate->target J-lens direction over a band."""

    def __init__(self, model, dirs: dict[int, tuple[torch.Tensor, torch.Tensor]]):
        self.model, self.dirs, self.handles = model, dirs, []

    def __enter__(self):
        for layer, (v_from, v_to) in self.dirs.items():
            def mk(vf, vt):
                def hook(mod, inp, out):
                    h = out[0] if isinstance(out, tuple) else out
                    proj = h @ vf.to(h.dtype)  # [b, s]
                    h = h - proj[..., None] * vf.to(h.dtype) + proj[..., None] * vt.to(h.dtype)
                    return (h, *out[1:]) if isinstance(out, tuple) else h
                return hook
            self.handles.append(self.model.layers[layer].register_forward_hook(mk(v_from, v_to)))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def greedy_next(model, prompt: str) -> int:
    ids = model.encode(prompt, max_length=128)
    from jlens import ActivationRecorder
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0, -1]
    return int(model.unembed(h[None]).argmax(-1))


@torch.no_grad()
def run(model, lens, items, seed_dir):
    tok = model.tokenizer
    W_U = model._lm_head.weight  # [vocab, d]
    dev = W_U.device
    layers_by_band = band_layers(lens.source_layers, model.n_layers)
    rows = []
    for band, layers in layers_by_band.items():
        if not layers:
            continue
        n_base = n_swap = n_ctrl = n_eval = 0
        for it in items:
            ans = first_token(tok, it["answer"])
            swp = first_token(tok, it["swap_answer"])
            if greedy_next(model, it["prompt"]) != ans:
                continue  # only score items the model gets right cleanly
            n_eval += 1
            ti = first_token(tok, it["intermediate"])
            tt = first_token(tok, it["swap_to"])
            dirs, ctrl = {}, {}
            for l in layers:
                vf = jlens_dir(lens, W_U, l, ti, dev)
                vt = jlens_dir(lens, W_U, l, tt, dev)
                g = torch.randn(vf.shape, generator=seed_dir, device="cpu").to(dev)
                g = g / g.norm()
                dirs[l] = (vf, vt)
                ctrl[l] = (vf, g)
            with SwapHooks(model, dirs):
                if greedy_next(model, it["prompt"]) == swp:
                    n_swap += 1
            with SwapHooks(model, ctrl):
                if greedy_next(model, it["prompt"]) == swp:
                    n_ctrl += 1
            n_base += 1
        rows.append(dict(
            band=band, layers=len(layers), n_clean_correct=n_eval,
            swap_flip_rate=n_swap / max(n_base, 1),
            control_flip_rate=n_ctrl / max(n_base, 1),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    items = json.loads((DATA / "probe-swap.json").read_text())["items"]
    if args.limit:
        items = items[: args.limit]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(key)
    else:
        model = load_model(key)
    lens = JacobianLens.from_pretrained(str(lens_path))
    seed = torch.Generator().manual_seed(0)

    df = run(model, lens, items, seed)
    df.insert(0, "model", tag)
    out = ROOT / "results" / "ablation" / f"{tag}_swap.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
