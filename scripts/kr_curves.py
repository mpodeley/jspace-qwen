#!/usr/bin/env python
"""Relative-permeability curves k_r(S) for concepts sharing the same weights.

Multiphase-flow analogy: concepts (phases) share the residual stream (pore
space) and a layer's finite write bandwidth. For a set of concepts we measure,
at each cell (layer in the workspace band, position) over a corpus:

  S_c  = share of the current concept 'saturation' held by concept c
         = |<J_l h, W_U[c]>| normalized across the concept set  (sums to 1)
  kr_c = share of the layer's *write* (recharge dC) carried by concept c
         = |<J_{l+1} dh, W_U[c]>| normalized across the set     (sums to 1)

Plotting kr_c against S_c gives the relative-permeability curve: how much of a
layer's output bandwidth a concept captures as a function of how present it
already is. Competition/interference shows up as kr_A falling while another
phase rises; an 'irreducible saturation' as kr_c ~ 0 below a threshold S_c.

This is an operationalization of the analogy, exploratory by nature.

    python scripts/kr_curves.py 8b --concepts euro dollar yen pound franc
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import MODELS, depth_percent, get_corpus, load_model, resolve_tag
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
# categorical (validated slots), enough for ~6 concepts
PAL = ["#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948", "#e87ba4"]
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb", "text.color": INK,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "font.size": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})
WORKSPACE = (38, 92)


def first_token(tok, w):
    return tok.encode(" " + w.strip(), add_special_tokens=False)[0]


@torch.no_grad()
def collect(model, lens, prompts, tok_ids):
    """Gather (S, kr) shares per concept over workspace-band cells."""
    W_U = model._lm_head.weight.float()
    U = torch.stack([W_U[t] for t in tok_ids]).to("cuda")  # [C, d]
    layers = [l for l in lens.source_layers
              if WORKSPACE[0] <= depth_percent(l, model.n_layers) <= WORKSPACE[1]]
    S_all, K_all = [], []
    for prompt in prompts:
        ids = model.encode(prompt, max_length=128)
        need = sorted(set(layers) | {l + 1 for l in layers if l + 1 < model.n_layers})
        need = [l for l in need if l < model.n_layers]
        with ActivationRecorder(model.layers, at=need) as rec:
            model.forward(ids)
            acts = {l: rec.activations[l][0].float() for l in need}
        for l in layers:
            if l + 1 not in acts:
                continue
            h = acts[l][16:-1]           # valid positions
            dh = acts[l + 1][16:-1] - h
            Jt = lens.jacobians[l].to("cuda").t()      # fitted transport at layer l
            # concept logit (saturation) and write-flux, per position x concept
            sat = (h @ (Jt @ U.t())).abs()    # [P, C]
            flux = (dh @ (Jt @ U.t())).abs()  # [P, C]  (write transported with J_l)
            S = sat / (sat.sum(1, keepdim=True) + 1e-9)
            K = flux / (flux.sum(1, keepdim=True) + 1e-9)
            S_all.append(S.cpu().numpy()); K_all.append(K.cpu().numpy())
    return np.concatenate(S_all), np.concatenate(K_all)  # [N, C] each


def binned(x, y, nb=10):
    edges = np.linspace(0, x.max() + 1e-9, nb + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nb - 1)
    xs, ys = [], []
    for b in range(nb):
        m = idx == b
        if m.sum() >= 5:
            xs.append(x[m].mean()); ys.append(y[m].mean())
    return np.array(xs), np.array(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--concepts", nargs="+",
                    default=["euro", "dollar", "yen", "pound", "franc"])
    ap.add_argument("--n-prompts", type=int, default=24)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    model = (__import__("int8_model").load_int8_model(key) if args.int8
             else load_model(key))
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    tids = [first_token(tok, c) for c in args.concepts]

    # prompts where these concepts vary: multihop currency set + some web text
    import json
    mh = json.loads((ROOT / "jacobian-lens" / "data" / "evaluations"
                     / "lens-eval-multihop.json").read_text())["items"]
    prompts = [it["prompt"] for it in mh[:args.n_prompts]] + get_corpus(8)
    S, K = collect(model, lens, prompts, tids)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for i, c in enumerate(args.concepts):
        a1.scatter(S[:, i], K[:, i], s=4, alpha=0.15, color=PAL[i])
        bx, by = binned(S[:, i], K[:, i])
        a1.plot(bx, by, color=PAL[i], lw=2.2, marker="o", ms=4, label=c)
    a1.plot([0, 1], [0, 1], color=MUTED, ls=":", lw=1)  # kr = S reference
    a1.set_xlabel("concept saturation $S_c$ (share)")
    a1.set_ylabel("relative permeability $k_r^c$ (write share)")
    a1.set_title(f"{tag}: $k_r(S)$ per concept (workspace band)", fontsize=11)
    a1.legend(frameon=False, fontsize=8)

    # competition: kr of concept 0 vs saturation of concept 1 (interference)
    bx, by = binned(S[:, 1], K[:, 0])
    a2.plot(bx, by, color=PAL[0], lw=2.2, marker="o", ms=4,
            label=f"$k_r$({args.concepts[0]}) vs $S$({args.concepts[1]})")
    a2.set_xlabel(f"saturation of {args.concepts[1]}")
    a2.set_ylabel(f"relative perm of {args.concepts[0]}")
    a2.set_title("Interference (competing phase)", fontsize=11)
    a2.legend(frameon=False, fontsize=8)

    out = FIGS / f"kr_{tag}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
