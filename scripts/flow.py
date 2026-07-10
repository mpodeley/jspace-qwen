#!/usr/bin/env python
"""Concept-flow visualization: streamlines from workspace to output.

Two views of how a concept moves through the network, both forward-only:

1. Concept-plane trajectory. Pick two concept tokens (e.g. the two-hop
   intermediate "Italy" and the answer "euro"). Their unembedding rows define a
   2D plane. At each layer we transport the residual into the final basis with
   the J-lens (J_l h) and project onto that plane, drawing the layer-by-layer
   path. The logit-lens path (no transport) is drawn alongside. The trajectory
   crossing from the intermediate axis to the answer axis *inside the workspace
   band* is the visual "channel" from workspace to output. Several token
   positions give several streamlines = a flow field.

2. Read-channel profile. The component of the residual along a concept's J-lens
   read direction v_{l,t} = J_l^T W_U[t], as a function of depth — i.e. which
   layers build up that concept on its way to the output.

NB: attention mixes positions, so this is a projected trajectory, not an
autonomous vector field; "streamline" is a visual metaphor for the path.

    python scripts/flow.py 1.7b --intermediate Italy --answer euro \
        --prompt "Fact: The currency used in the country shaped like a boot is"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from _common import MODELS, depth_percent, load_model, resolve_tag
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
C_J, C_L = "#2a78d6", "#eb6834"  # J-lens blue, logit-lens orange (validated slots)

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "text.color": INK, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})


def first_token(tok, w):
    return tok.encode(" " + w.strip(), add_special_tokens=False)[0]


@torch.no_grad()
def residuals_at(model, prompt, layers, pos):
    ids = model.encode(prompt, max_length=128)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
        return {l: rec.activations[l][0, pos].float() for l in layers}, ids


def concept_plane(model, lens, prompt, ti, ta, pos):
    """Return per-layer (x=intermediate, y=answer) coords for J-lens & logit-lens."""
    W_U = model._lm_head.weight.float()
    dev = W_U.device
    # orthonormal 2D concept plane from the two unembedding rows
    a = W_U[ti].to(dev); b = W_U[ta].to(dev)
    e1 = a / a.norm()
    b_perp = b - (b @ e1) * e1
    e2 = b_perp / (b_perp.norm() + 1e-8)
    layers = lens.source_layers
    res, _ = residuals_at(model, prompt, layers, pos)
    jx, jy, lx, ly = [], [], [], []
    for l in layers:
        h = res[l].to(dev)
        t = lens.transport(h, l)               # J_l h  (final basis)
        for vec, xs, ys in ((t, jx, jy), (h, lx, ly)):
            n = model._final_norm(vec.to(model._lm_head.weight.dtype)).float()
            xs.append(float(n @ e1)); ys.append(float(n @ e2))
    return layers, (np.array(jx), np.array(jy)), (np.array(lx), np.array(ly))


def _stream(ax, xs, ys, depths, color, label):
    pts = np.column_stack([xs, ys]).reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, colors=color, linewidths=2, alpha=0.9)
    ax.add_collection(lc)
    # arrowheads every few layers to show direction of flow
    for i in range(0, len(xs) - 1, max(len(xs) // 6, 1)):
        ax.annotate("", xy=(xs[i + 1], ys[i + 1]), xytext=(xs[i], ys[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.4))
    ax.scatter(xs[-1], ys[-1], color=color, s=28, zorder=5)
    ax.annotate(label, (xs[-1], ys[-1]), color=color, fontsize=9,
                xytext=(4, 4), textcoords="offset points")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--intermediate", default="Italy")
    ap.add_argument("--answer", default="euro")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is"))
    ap.add_argument("--position", type=int, default=-1)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    model = (__import__("int8_model").load_int8_model(key) if args.int8
             else load_model(key))
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    ti, ta = first_token(tok, args.intermediate), first_token(tok, args.answer)

    layers, (jx, jy), (lx, ly) = concept_plane(model, lens, args.prompt, ti, ta,
                                               args.position)
    depths = [depth_percent(l, model.n_layers) for l in layers]

    fig, (axf, axc) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    # left: concept-plane streamlines
    _stream(axf, jx, jy, depths, C_J, "J-lens")
    _stream(axf, lx, ly, depths, C_L, "logit-lens")
    axf.set_xlabel(f'"{args.intermediate}" axis  (intermediate)')
    axf.set_ylabel(f'"{args.answer}" axis  (output)')
    axf.set_title("Concept-plane flow: workspace → output", fontsize=11)
    axf.autoscale()

    # right: read-channel profile — build-up of the answer concept along depth
    comp = jy / (np.abs(jy).max() + 1e-9)
    axc.plot(depths, comp, color=C_J, lw=2, marker="o", ms=3.5)
    axc.axvspan(38, 92, color="#000000", alpha=0.05, lw=0)
    axc.set_xlabel("depth (0–100)")
    axc.set_ylabel(f'"{args.answer}" build-up (normalized)')
    axc.set_title("Read-channel profile (which layers build the output)", fontsize=11)

    fig.suptitle(f"{tag}: concept flow for “{args.intermediate}→{args.answer}”",
                 fontsize=12, x=0.01, ha="left")
    out = FIGS / f"flow_{tag}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
