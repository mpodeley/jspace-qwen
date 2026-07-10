#!/usr/bin/env python
"""High-permeability channel from workspace to output (reservoir view).

Building on the saturation field C[l,p] = <J_l h[l,p], W_U[answer]> (how much of
the answer concept each cell holds), this identifies the *channel*:

- Write-rate (recharge) field dC/dl: how much each cell *adds* the concept to
  the residual highway as depth increases. Bright cells = high-permeability
  sources feeding the output — the reservoir "recharge".
- Channel trace: at each layer, the position carrying the most of the concept;
  connected across depth this is the dominant route from the source token
  (injector) up through the workspace to the output position (producer).

Forward-only. attention-level transmissibility (position->position edges) is the
next fidelity level; see notes in the docs.

    python scripts/permeability.py 1.7b --answer euro
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
    "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 9, "figure.dpi": 130,
})


def first_token(tok, w):
    return tok.encode(" " + w.strip(), add_special_tokens=False)[0]


@torch.no_grad()
def saturation(model, lens, prompt, tok_id):
    layers = lens.source_layers
    ll, _, ids = lens.apply(model, prompt, layers=layers, positions=None)
    C = np.stack([ll[l][:, tok_id].numpy() for l in layers])  # [L, seq]
    return C, layers, ids[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--answer", default="euro")
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

    C, layers, ids = saturation(model, lens, args.prompt, first_token(tok, args.answer))
    depths = np.array([depth_percent(l, model.n_layers) for l in layers])
    toks = [tok.decode([t]).strip()[:8] or "·" for t in ids.tolist()]
    nL, nP = C.shape

    # recharge (write-rate): concept added per unit depth
    dC = np.vstack([C[1:] - C[:-1], (C[-1] - C[-2])[None]])
    # channel trace: most-saturated position per layer (the route)
    path = C.argmax(axis=1)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)

    im1 = a1.imshow(C, aspect="auto", origin="lower", cmap="viridis",
                    extent=[0, nP, depths[0], depths[-1]])
    a1.plot(path + 0.5, depths, color="#eb6834", lw=2.2, marker="o", ms=3,
            label="dominant channel")
    a1.axhspan(38, 92, color="white", alpha=0.10, lw=0)
    a1.set_title(f'"{args.answer}" saturation + channel', fontsize=10)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    fig.colorbar(im1, ax=a1, fraction=0.046, pad=0.02)

    im2 = a2.imshow(dC, aspect="auto", origin="lower", cmap="magma",
                    extent=[0, nP, depths[0], depths[-1]])
    a2.axhspan(38, 92, color="white", alpha=0.10, lw=0)
    a2.set_title("recharge dC/d(depth): where the concept is written", fontsize=10)
    fig.colorbar(im2, ax=a2, fraction=0.046, pad=0.02)

    for ax in (a1, a2):
        ax.set_xticks(np.arange(nP) + 0.5)
        ax.set_xticklabels(toks, rotation=90, fontsize=6)
        ax.set_ylabel("depth (0–100)")
    fig.suptitle(f"{tag}: permeability channel to output — “{args.answer}”",
                 fontsize=12, x=0.01, ha="left")
    out = FIGS / f"perm_{tag}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    # also report the channel as text
    print("channel (depth% -> token):")
    for d, p in zip(depths[::3], path[::3]):
        print(f"  {d:5.0f}  {toks[p]!r}")
    print("wrote", out)


if __name__ == "__main__":
    main()
