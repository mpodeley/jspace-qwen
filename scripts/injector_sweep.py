#!/usr/bin/env python
"""Animated injector sweep: propagate a concept through the network.

Reservoir 'injector well' view. We inject a country at the intermediate node by
interpolating the residual from Italy's J-lens direction toward another
country's, across the workspace band, at strength alpha in [0, 1]. For each
alpha we read two saturation fields over the (layer x position) grid:

  - country plane: saturation of the injected country (the advancing 'front');
  - currency plane: a diverging map sat(dollar) - sat(euro), showing the
    `currency` channel flipping from euro (blue) to dollar (red) as the front
    arrives.

Sweeping alpha and stitching the frames gives an animation of the concept
propagating and the currency channel switching euro -> dollar. Output: an
animated GIF plus the final frame as a PNG (both embed in the docs).

    python scripts/injector_sweep.py 1.7b --to "United States" --currency-to dollar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation

from _common import MODELS, depth_percent, load_model, resolve_tag
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 9, "figure.dpi": 120,
})
WORKSPACE = (38, 92)


def first_token(tok, w):
    return tok.encode(" " + w.strip(), add_special_tokens=False)[0]


def jdir(lens, W_U, layer, tid, dev):
    v = lens.jacobians[layer].to(dev).t() @ W_U[tid].to(dev)
    return v / (v.norm() + 1e-8)


class InterpSwap:
    """Move fraction alpha of the 'from' component onto 'to', per band layer."""
    def __init__(self, model, dirs, alpha):
        self.model, self.dirs, self.alpha, self.h = model, dirs, alpha, []

    def __enter__(self):
        for l, (vf, vt) in self.dirs.items():
            def mk(vf, vt, a):
                def hook(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    p = (h @ vf.to(h.dtype))[..., None]
                    h = h - a * p * vf.to(h.dtype) + a * p * vt.to(h.dtype)
                    return (h, *o[1:]) if isinstance(o, tuple) else h
                return hook
            self.h.append(self.model.layers[l].register_forward_hook(mk(vf, vt, self.alpha)))
        return self

    def __exit__(self, *a):
        for x in self.h:
            x.remove()


@torch.no_grad()
def fields(model, lens, prompt, tok_ids, dirs, alpha):
    with InterpSwap(model, dirs, alpha):
        ll, _, ids = lens.apply(model, prompt, layers=lens.source_layers, positions=None)
    out = {name: np.stack([ll[l][:, t].numpy() for l in lens.source_layers])
           for name, t in tok_ids.items()}
    return out, ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--from", dest="src", default="Italy")
    ap.add_argument("--to", default="United States")
    ap.add_argument("--currency-from", default="euro")
    ap.add_argument("--currency-to", default="dollar")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is the"))
    ap.add_argument("--frames", type=int, default=11)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    model = (__import__("int8_model").load_int8_model(key) if args.int8
             else load_model(key))
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    W_U = model._lm_head.weight.float()
    dev = W_U.device
    layers = lens.source_layers
    depths = np.array([depth_percent(l, model.n_layers) for l in layers])
    band = [l for l in layers if WORKSPACE[0] <= depth_percent(l, model.n_layers) <= WORKSPACE[1]]

    ti, tt = first_token(tok, args.src), first_token(tok, args.to)
    dirs = {l: (jdir(lens, W_U, l, ti, dev), jdir(lens, W_U, l, tt, dev)) for l in band}
    tok_ids = {"country": tt, args.currency_from: first_token(tok, args.currency_from),
               args.currency_to: first_token(tok, args.currency_to)}

    alphas = np.linspace(0, 1, args.frames)
    frames = [fields(model, lens, args.prompt, tok_ids, dirs, a)[0] for a in alphas]
    _, ids = fields(model, lens, args.prompt, tok_ids, dirs, 0.0)
    toks = [tok.decode([t]).strip()[:8] or "·" for t in ids.tolist()]
    nP = len(toks)

    cmax = max(np.abs(f["country"]).max() for f in frames)
    dmax = max(np.abs(f[args.currency_to] - f[args.currency_from]).max() for f in frames)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ext = [0, nP, depths[0], depths[-1]]
    im1 = a1.imshow(frames[0]["country"], aspect="auto", origin="lower", cmap="viridis",
                    extent=ext, vmin=0, vmax=cmax)
    im2 = a2.imshow(frames[0][args.currency_to] - frames[0][args.currency_from],
                    aspect="auto", origin="lower", cmap="RdBu_r", extent=ext,
                    vmin=-dmax, vmax=dmax)
    for ax in (a1, a2):
        ax.axhspan(*WORKSPACE, color="white", alpha=0.08, lw=0)
        ax.set_xticks(np.arange(nP) + 0.5); ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_ylabel("depth (0–100)")
    a1.set_title(f'country front: "{args.to}"', fontsize=10)
    a2.set_title(f'currency channel: {args.currency_from} (blue) → {args.currency_to} (red)',
                 fontsize=10)
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)
    sup = fig.suptitle("", fontsize=12, x=0.01, ha="left")

    def update(k):
        im1.set_data(frames[k]["country"])
        im2.set_data(frames[k][args.currency_to] - frames[k][args.currency_from])
        sup.set_text(f"{tag}: inject {args.src}→{args.to}   (strength α = {alphas[k]:.2f})")
        return im1, im2, sup

    anim = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)
    FIGS.mkdir(parents=True, exist_ok=True)
    gif = FIGS / f"inject_{tag}.gif"
    anim.save(gif, writer=animation.PillowWriter(fps=3))
    update(len(frames) - 1)
    fig.savefig(FIGS / f"inject_{tag}.png", bbox_inches="tight")
    print("wrote", gif, "and", FIGS / f"inject_{tag}.png")


if __name__ == "__main__":
    main()
