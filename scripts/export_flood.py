#!/usr/bin/env python
"""Export Concept-Flooding data as JSON for the web piece.

Two modes:
  --synthetic : physically-plausible mock data (numpy only, no GPU) so the viz
                can be built while the GPU is busy. Same schema as the real path.
  (default)   : real data from the fitted J-lens on a model (needs the GPU free).

Schema (data/flood_<tag>.json):
  meta: model, prompt, tokens[], depths[] (0-100), workspace_band[2],
        injector_pos, producer_pos, concepts{}, pair{a,b,cosine}, caveats[]
  alphas: [0..1]
  frames[]: per alpha -> country_field[L][P], diverging[L][P] in [-1,1],
        breakthrough{concept:share}, condensed[], neighbors[{token,cos,conc}]
  streamlines[]: concept-flow paths (final alpha)

The injected concept diffuses (miscible), and at the producer either condenses
into a word or stays latent (the featured latent<->verbalized phase signal).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data"

CAVEATS = [
    "Multicomponent, not two-phase: ~150k components (tokens); this is a low-dim slice.",
    "No discrete phases: concept directions overlap (superposition); 'saturation' is concentration and does not physically sum to 1.",
    "No conserved mass: LayerNorm rescales (sources/sinks everywhere).",
    "'Permeability' (attention) is content-dependent and non-local — not a fixed rock property.",
    "Deterministic, causal-in-position, one-way in depth — not pressure-driven reversible flow.",
    "The Jacobian is a linearization; real transport is nonlinear.",
]
DEFAULT_TOKENS = ["Fact", ":", "The", "currency", "used", "in", "the", "country",
                  "shaped", "like", "a", "boot", "is", "the"]


def _smooth_front(depths, positions, alpha, chan_pos, rng):
    """A concentration field: rises with depth, channels at chan_pos, front
    advances (deeper) as alpha grows. Returns [L][P] in ~[0, 1]."""
    L, P = len(depths), len(positions)
    d = np.array(depths) / 100.0
    field = np.zeros((L, P))
    front = 0.35 + 0.55 * alpha  # deeper reach with injection strength
    for j in range(P):
        col = 0.15 + 0.85 * np.clip((d - (1 - front)) / 0.4, 0, 1)  # rises past front
        chan = math.exp(-((j - chan_pos) ** 2) / 6.0)  # high-perm channel
        field[:, j] = col * (0.3 + 0.7 * chan)
    field += 0.04 * rng.standard_normal((L, P))
    return np.clip(field, 0, None)


def synthetic(tag: str, seed_terms):
    rng = np.random.default_rng(0)
    tokens = DEFAULT_TOKENS
    P = len(tokens)
    depths = [round(x, 1) for x in np.linspace(0, 100, 25)]
    L = len(depths)
    inj, prod = tokens.index("currency"), P - 1
    alphas = [round(a, 3) for a in np.linspace(0, 1, 13)]
    a_conc, b_conc = "euro", "dollar"
    frames = []
    for a in alphas:
        country = _smooth_front(depths, range(P), a, inj, rng)  # injected country front
        # currency: euro fades, dollar grows with alpha, concentrated at prod/inj cols
        base = _smooth_front(depths, range(P), 0.8, prod, rng)
        euro = base * (1 - a) + 0.05
        dollar = base * a + 0.05
        div = (dollar - euro)
        div = np.clip(div / (np.abs(div).max() + 1e-9), -1, 1)
        # breakthrough at producer (top of grid): shares
        bt_e = float(euro[-1, prod]); bt_d = float(dollar[-1, prod])
        s = bt_e + bt_d + 1e-9
        breakthrough = {a_conc: bt_e / s, b_conc: bt_d / s}
        condensed = [c for c, v in breakthrough.items() if v > 0.5]
        # semantic diffusion: near-neighbors that light up (mock)
        nbrs = [{"token": t, "cos": round(c, 2),
                 "conc": round(float(0.6 * a * c + 0.05 * rng.random()), 3)}
                for t, c in [("Dollar", 0.82), ("USD", 0.71), ("dollars", 0.68),
                             ("peso", 0.44), ("cent", 0.39)]]
        frames.append(dict(alpha=a,
                           country_field=np.round(country, 3).tolist(),
                           diverging=np.round(div, 3).tolist(),
                           breakthrough={k: round(v, 3) for k, v in breakthrough.items()},
                           condensed=condensed, neighbors=nbrs))
    # a couple of streamlines up the channel to the producer
    streamlines = []
    for j0 in (inj, inj + 1, prod):
        path = [[j0 + 0.5 + 0.4 * math.sin(i / 3), depths[i]] for i in range(L)]
        streamlines.append([[round(x, 2), round(y, 1)] for x, y in path])
    return dict(
        meta=dict(model=f"synthetic-{tag}", prompt=" ".join(tokens),
                  tokens=tokens, depths=depths, workspace_band=[38, 92],
                  injector_pos=inj, producer_pos=prod,
                  concepts={a_conc: {}, b_conc: {}},
                  pair=dict(a=a_conc, b=b_conc, cosine=0.42), caveats=CAVEATS,
                  synthetic=True),
        alphas=alphas, frames=frames, streamlines=streamlines)


def real(tag, model_key, int8, src, to, cur_from, cur_to, prompt, frames_n):
    """Real J-lens export (needs GPU free). Reuses the injector/field logic."""
    import torch

    from _common import depth_percent, load_model, resolve_tag
    from causal_swap import first_token
    from injector_sweep import InterpSwap, jdir
    from jlens import JacobianLens

    tag = resolve_tag(model_key, int8=int8)
    model = (__import__("int8_model").load_int8_model(model_key) if int8
             else load_model(model_key))
    lens = JacobianLens.from_pretrained(str(ROOT / "out" / "lenses" / f"{tag}.pt"))
    tok = model.tokenizer
    W_U = model._lm_head.weight.float()
    dev = W_U.device
    layers = lens.source_layers
    depths = [round(depth_percent(l, model.n_layers), 1) for l in layers]
    ids0 = model.encode(prompt, max_length=128)[0].tolist()
    tokens = [tok.decode([t]).strip()[:10] or "·" for t in ids0]
    band = [l for l in layers if 38 <= depth_percent(l, model.n_layers) <= 92]
    ti, tt = first_token(tok, src), first_token(tok, to)
    ce, cd = first_token(tok, cur_from), first_token(tok, cur_to)
    dirs = {l: (jdir(lens, W_U, l, ti, dev), jdir(lens, W_U, l, tt, dev)) for l in band}

    def field_of(tid, alpha):
        with InterpSwap(model, dirs, alpha):
            ll, ml, _ = lens.apply(model, prompt, layers=layers, positions=None)
        return (np.stack([ll[l][:, tid].numpy() for l in layers]),
                ml[:, tid].numpy())  # (lens field [L,P], model logits at final [P])

    # semantic neighbors of the injected country (cosine over vocab)
    inj_dir = W_U[tt] / W_U[tt].norm()
    cos_all = (W_U @ inj_dir) / (W_U.norm(dim=1) + 1e-9)
    nbr_ids = cos_all.topk(8).indices.tolist()[1:6]
    pair_cos = float(torch.cosine_similarity(W_U[ce][None], W_U[cd][None])[0])

    alphas = [round(a, 3) for a in np.linspace(0, 1, frames_n)]
    prod = len(tokens) - 1
    frames = []
    for a in alphas:
        country, _ = field_of(tt, a)
        euro, euro_out = field_of(ce, a)
        dollar, doll_out = field_of(cd, a)
        def norm01(x):
            x = x - x.min(); return x / (x.max() + 1e-9)
        div = dollar - euro
        div = np.clip(div / (np.abs(div).max() + 1e-9), -1, 1)
        s = abs(euro_out[prod]) + abs(doll_out[prod]) + 1e-9
        breakthrough = {cur_from: abs(float(euro_out[prod])) / s,
                        cur_to: abs(float(doll_out[prod])) / s}
        condensed = [c for c, v in breakthrough.items() if v > 0.5]
        nbrs = []
        for nid in nbr_ids:
            nf, _ = field_of(nid, a)
            nbrs.append({"token": tok.decode([nid]).strip()[:10],
                         "cos": round(float(cos_all[nid]), 2),
                         "conc": round(float(norm01(nf)[len(layers)//2].mean()), 3)})
        frames.append(dict(alpha=a, country_field=np.round(norm01(country), 3).tolist(),
                           diverging=np.round(div, 3).tolist(),
                           breakthrough={k: round(v, 3) for k, v in breakthrough.items()},
                           condensed=condensed, neighbors=nbrs))
    return dict(meta=dict(model=tag, prompt=prompt, tokens=tokens, depths=depths,
                          workspace_band=[38, 92], injector_pos=0, producer_pos=prod,
                          concepts={cur_from: {}, cur_to: {}},
                          pair=dict(a=cur_from, b=cur_to, cosine=round(pair_cos, 3)),
                          caveats=CAVEATS, synthetic=False),
                alphas=alphas, frames=frames, streamlines=[])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="8b")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--from", dest="src", default="Italy")
    ap.add_argument("--to", default="United States")
    ap.add_argument("--currency-from", default="euro")
    ap.add_argument("--currency-to", default="dollar")
    ap.add_argument("--prompt", default=(
        "Fact: The currency used in the country shaped like a boot is the"))
    ap.add_argument("--frames", type=int, default=13)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.synthetic:
        data = synthetic("dev", [args.currency_from, args.currency_to])
        path = OUT / "flood_sample.json"
    else:
        data = real("", args.model, args.int8, args.src, args.to,
                    args.currency_from, args.currency_to, args.prompt, args.frames)
        path = OUT / f"flood_{data['meta']['model']}.json"
    path.write_text(json.dumps(data))
    print("wrote", path, f"({path.stat().st_size//1024} KB, {len(data['frames'])} frames)")


if __name__ == "__main__":
    main()
