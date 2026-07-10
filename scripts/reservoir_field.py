#!/usr/bin/env python
"""Concept 'saturation' field over the (layer x position) grid.

Reservoir-simulation view of the model: the grid cells are activations
h[layer, position]; the scalar we plot is how strongly a concept token is read
out of each cell by the J-lens, C[l,p] = <J_l h[l,p], W_U[token]>. This is the
concept's 'saturation map' across the network — where and when it lights up on
its way from the source token (injector) to the output (producer).

The logit-lens field (no transport) is shown alongside; it stays dim in the
early/mid cells because it reads the residual in the wrong basis.

    python scripts/reservoir_field.py 1.7b --concepts Italy euro \
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

from _common import MODELS, depth_percent, load_model, resolve_tag
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
INK, MUTED = "#0b0b0b", "#898781"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "text.color": INK, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 9, "figure.dpi": 130,
})


def first_token(tok, w):
    return tok.encode(" " + w.strip(), add_special_tokens=False)[0]


@torch.no_grad()
def concept_field(model, lens, prompt, tok_id, *, use_jacobian):
    """C[l, p] = concept logit read from cell (layer l, position p)."""
    layers = lens.source_layers
    lens_logits, _, ids = lens.apply(
        model, prompt, layers=layers, positions=None, use_jacobian=use_jacobian
    )
    field = np.stack([lens_logits[l][:, tok_id].numpy() for l in layers])  # [L, seq]
    return field, layers, ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--concepts", nargs="+", default=["Italy", "euro"])
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is"))
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    model = (__import__("int8_model").load_int8_model(key) if args.int8
             else load_model(key))
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer

    # panels: J-lens field for each concept, plus logit-lens field for concept[-1]
    panels = [(c, True) for c in args.concepts] + [(args.concepts[-1], False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.4),
                             constrained_layout=True, squeeze=False)
    for ax, (concept, use_j) in zip(axes[0], panels):
        tid = first_token(tok, concept)
        field, layers, ids = concept_field(model, lens, args.prompt, tid,
                                            use_jacobian=use_j)
        depths = [depth_percent(l, model.n_layers) for l in layers]
        toks = [tok.decode([t]).strip()[:8] or "·" for t in ids.tolist()]
        im = ax.imshow(field, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0, len(toks), depths[0], depths[-1]])
        ax.axhspan(38, 92, color="white", alpha=0.10, lw=0)  # workspace band hint
        ax.set_xticks(np.arange(len(toks)) + 0.5)
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_ylabel("depth (0–100)")
        kind = "J-lens" if use_j else "logit-lens"
        ax.set_title(f'"{concept}"  ({kind})', fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"{tag}: concept saturation field over (layer × position)",
                 fontsize=12, x=0.01, ha="left")
    out = FIGS / f"field_{tag}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
