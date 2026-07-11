#!/usr/bin/env python
"""A competitive null battery for the operator swap.

The matched-norm Gaussian used elsewhere is a weak null: in a high-dimensional
residual stream almost any random direction is nearly orthogonal to everything
that matters, so beating it is cheap. This script runs the nulls that actually
share the structure of the real intervention, and are therefore hard to beat:

  permuted (per-operand)  rebuild v(op) after independently permuting the operator
                          labels within each operand's cells. Same residuals, same
                          averaging, same norms, same everything -- only the intended
                          semantics is destroyed. THE decisive control: if the effect
                          survives label permutation, the direction is not carrying
                          the relation.
  permuted (global)       one permutation for all operands: the directions stay
                          internally coherent but are mis-assigned (v'(k) = v(pi(k))).
                          Separates "label alignment matters" from "coherence matters".
  operator subspace       a random direction INSIDE the span of the operator
                          directions, norm-matched per layer. Much stronger than a
                          full-space Gaussian: it has the right home, wrong content.
  wrong layer             the correct direction injected into the early band instead
                          of the workspace band.
  other relation          v(other) - v(from) for a relation other than the target,
                          norm-matched, still scored toward the target answer.
  shuffled answers        the real intervention, scored against a permuted
                          operand -> answer map (target = another operand's answer for
                          the same relation). Tests specificity to the true answer
                          rather than a generic "attribute of a country" boost.
  random (Gaussian)       the paper's existing full-space control, for reference.

    python scripts/op_nulls.py 1.7b --domain relations
    python scripts/op_nulls.py 1.7b --domain relations --seeds 20
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core

ROOT = Path(__file__).resolve().parent.parent


def swap_rows(model, layers, dom, tok, dirs, alpha, gen, dev, d_model,
              dir_layers=None, answer_perm=None, other_dirs=None):
    """One long-form swap frame in op_core's schema. Generalizes
    `op_core.measure_swaps_long` along the axes the nulls need:

      dir_layers   layers the direction vectors are INDEXED by, if different from
                   the layers they are INJECTED into (the wrong-layer null).
      answer_perm  operand -> operand map for scoring: the target token becomes
                   another operand's answer for the same relation (shuffled answers).
      other_dirs   per-pair override of the 'to' endpoint: dv = other[frm][to] - v[frm]
                   (the other-relation null).

    With all three at their defaults this is measure_swaps_long exactly (asserted
    against it in main), so the nulls and the reference share one code path."""
    dls = dir_layers or layers
    ops = dom.op_keys
    rows = []
    for frm, to in [(a, b) for a in ops for b in ops if a != b]:
        end = other_dirs[(frm, to)] if other_dirs else dirs[to]
        dv = {l: alpha * (end[dl] - dirs[frm][dl]) for l, dl in zip(layers, dls)}
        norm = torch.cat([dv[l] for l in layers]).norm()
        rv = {l: torch.randn(d_model, generator=gen).to(dev) for l in layers}
        rn = torch.cat([rv[l] for l in layers]).norm()
        rv = {l: rv[l] / rn * norm for l in layers}
        for o in dom.operand_keys:
            score_o = answer_perm[o] if answer_perm else o
            af = dom.answer_tok(tok, o, frm)
            at = dom.answer_tok(tok, score_o, to)
            if af == at:
                continue
            p = dom.render(o, frm)
            L0 = op_core.final_logits(model, p)
            Ls = op_core.final_logits(model, p, dv)
            Lr = op_core.final_logits(model, p, rv)
            rows.append({"from": frm, "to": to, "operand": o,
                         "clean": float(L0[at] - L0[af]),
                         "swap": float(Ls[at] - Ls[af]),
                         "random": float(Lr[at] - Lr[af])})
    return pd.DataFrame(rows)


def summarize(name, ldf, seed):
    fam = op_core.bootstrap_family_ci(ldf, seed=seed)
    return {"null": name, "n": int(len(ldf)),
            "clean": float(ldf["clean"].mean()),
            "swap": float(ldf["swap"].mean()),
            "random": float(ldf["random"].mean()),
            "delta_margin": float((ldf["swap"] - ldf["clean"]).mean()),
            "contrast": fam["contrast_mean"],
            "contrast_lo": fam["contrast_lo"], "contrast_hi": fam["contrast_hi"],
            "flip_frac": fam["flip_frac"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--seeds", type=int, default=20,
                    help="redraws for the stochastic nulls (permutations, subspace)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    bands = band_layers(source_layers, model.n_layers)
    ws, early = bands["workspace"], bands["early"]
    ops = dom.op_keys
    d, dev = model.d_model, None

    print(f"[{tag} / {args.domain}] null battery, alpha={args.alpha}, "
          f"ws={ws[0]}..{ws[-1]}, early={early[0]}..{early[-1]}, "
          f"{args.seeds} redraws per stochastic null")

    R = op_core.op_resids(model, ws, dom, dom.operand_keys, [0])
    v = op_core.dirs_from_resids(R, ws, ops, dom.operand_keys, [0], d)
    dev = v[ops[0]][ws[0]].device

    # --- reference: the real swap, and a proof this code path == op_core's -------
    g = torch.Generator().manual_seed(args.seed)
    real = swap_rows(model, ws, dom, tok, v, args.alpha, g, dev, d)
    ref = op_core.measure_swaps_long(model, ws, dom, tok, seed=args.seed,
                                     alpha=args.alpha, dirs=v)
    m = real.merge(ref, on=["from", "to", "operand"], suffixes=("", "_ref"))
    drift = max(float((m[c] - m[f"{c}_ref"]).abs().max()) for c in ("swap", "random"))
    assert len(m) == len(ref) and drift < 1e-3, f"null code path drifted: {drift}"
    print(f"  [check] reference path == op_core.measure_swaps_long (max Δ {drift:.1e})")

    rows = [summarize("real swap", real, args.seed)]
    long_frames = {"real swap": real}

    # --- permuted operator labels (the decisive null) ---------------------------
    for label, per_op in (("permuted labels (per-operand)", True),
                          ("permuted labels (global)", False)):
        per_seed, frames = [], []
        for s in range(args.seeds):
            pv = op_core.permuted_op_dirs(model, ws, dom, seed=1000 + s,
                                          per_operand=per_op, R=R)
            gg = torch.Generator().manual_seed(args.seed)
            ldf = swap_rows(model, ws, dom, tok, pv, args.alpha, gg, dev, d)
            per_seed.append(float((ldf["swap"] - ldf["random"]).mean()))
            frames.append(ldf)
        pooled = pd.concat(frames, ignore_index=True)
        rec = summarize(label, pooled, args.seed)
        rec["seed_mean"] = float(np.mean(per_seed))
        rec["seed_lo"] = float(np.percentile(per_seed, 2.5))
        rec["seed_hi"] = float(np.percentile(per_seed, 97.5))
        rec["n_seeds"] = args.seeds
        rows.append(rec)
        long_frames[label] = pooled

    # --- random inside the operator subspace ------------------------------------
    Q = op_core.operator_subspace_basis(v, ws)
    frames, per_seed = [], []
    for s in range(args.seeds):
        gs = torch.Generator().manual_seed(2000 + s)
        # a fixed random subspace direction plays the role of every dv this round:
        # build a dirs-like dict whose differences are subspace-random, norm-matched
        # per pair to the real dv.
        sub = {}
        for frm, to in [(a, b) for a in ops for b in ops if a != b]:
            norms = {l: (args.alpha * (v[to][l] - v[frm][l])).norm() for l in ws}
            sub[(frm, to)] = op_core.random_in_subspace(Q, norms, gs)
        gg = torch.Generator().manual_seed(args.seed)
        # inject sub[(frm,to)] directly: dirs[to] - dirs[frm] must equal it/alpha,
        # so pass zeros for 'from' and sub/alpha for 'to'.
        zero = {k: {l: torch.zeros(d, device=dev) for l in ws} for k in ops}
        other = {(frm, to): {l: sub[(frm, to)][l] / args.alpha for l in ws}
                 for frm, to in sub}
        ldf = swap_rows(model, ws, dom, tok, zero, args.alpha, gg, dev, d,
                        other_dirs=other)
        per_seed.append(float((ldf["swap"] - ldf["random"]).mean()))
        frames.append(ldf)
    pooled = pd.concat(frames, ignore_index=True)
    rec = summarize("random in operator subspace", pooled, args.seed)
    rec["seed_mean"] = float(np.mean(per_seed))
    rec["seed_lo"] = float(np.percentile(per_seed, 2.5))
    rec["seed_hi"] = float(np.percentile(per_seed, 97.5))
    rec["n_seeds"] = args.seeds
    rows.append(rec)
    long_frames["random in operator subspace"] = pooled

    # --- correct direction, wrong layer (early band) ----------------------------
    n = min(len(ws), len(early))
    g = torch.Generator().manual_seed(args.seed)
    ldf = swap_rows(model, early[:n], dom, tok, v, args.alpha, g, dev, d,
                    dir_layers=ws[:n])
    rows.append(summarize("wrong layer (early band)", ldf, args.seed))
    long_frames["wrong layer (early band)"] = ldf

    # --- other-relation direction, norm-matched ---------------------------------
    g = torch.Generator().manual_seed(args.seed)
    rng = _random.Random(args.seed)
    other = {}
    for frm, to in [(a, b) for a in ops for b in ops if a != b]:
        cands = [k for k in ops if k not in (frm, to)]
        oth = rng.choice(cands)
        real_norm = {l: (v[to][l] - v[frm][l]).norm() for l in ws}
        dv_o = {l: v[oth][l] - v[frm][l] for l in ws}
        # norm-match the *difference*, then re-express as an endpoint for swap_rows
        other[(frm, to)] = {l: v[frm][l] + dv_o[l] / (dv_o[l].norm() + 1e-9)
                            * real_norm[l] for l in ws}
    ldf = swap_rows(model, ws, dom, tok, v, args.alpha, g, dev, d, other_dirs=other)
    rows.append(summarize("other-relation direction", ldf, args.seed))
    long_frames["other-relation direction"] = ldf

    # --- shuffled answers -------------------------------------------------------
    per_seed, frames = [], []
    for s in range(args.seeds):
        rr = _random.Random(3000 + s)
        keys = dom.operand_keys[:]
        shuf = keys[:]
        for _ in range(50):  # derangement: nobody keeps their own answer
            rr.shuffle(shuf)
            if all(a != b for a, b in zip(keys, shuf)):
                break
        perm = dict(zip(keys, shuf))
        gg = torch.Generator().manual_seed(args.seed)
        ldf = swap_rows(model, ws, dom, tok, v, args.alpha, gg, dev, d,
                        answer_perm=perm)
        per_seed.append(float((ldf["swap"] - ldf["random"]).mean()))
        frames.append(ldf)
    pooled = pd.concat(frames, ignore_index=True)
    rec = summarize("shuffled answers", pooled, args.seed)
    rec["seed_mean"] = float(np.mean(per_seed))
    rec["seed_lo"] = float(np.percentile(per_seed, 2.5))
    rec["seed_hi"] = float(np.percentile(per_seed, 97.5))
    rec["n_seeds"] = args.seeds
    rows.append(rec)
    long_frames["shuffled answers"] = pooled

    # --- report -----------------------------------------------------------------
    print(f"\n{'null':<32} {'n':>5} {'swap':>8} {'Δmargin':>9} "
          f"{'contrast [95% CI]':>24} {'flips':>6}")
    for r in rows:
        ci = f"{r['contrast']:+.2f} [{r['contrast_lo']:+.1f},{r['contrast_hi']:+.1f}]"
        print(f"{r['null']:<32} {r['n']:>5} {r['swap']:>+8.2f} "
              f"{r['delta_margin']:>+9.2f} {ci:>24} {r['flip_frac']:>6.2f}")

    df = pd.concat([f.assign(null=k) for k, f in long_frames.items()],
                   ignore_index=True)
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_nulls.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    meta = {"tag": tag, "domain": args.domain, "alpha": args.alpha,
            "ws": [int(l) for l in ws], "early": [int(l) for l in early],
            "n_seeds": args.seeds,
            "bootstrap_unit": "operator (dyadic node), operands nested; "
                              "stochastic nulls also report a percentile interval "
                              "across redraws (seed_lo/seed_hi)",
            "nulls": rows}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"\nsaved {out} (+ .json)")


if __name__ == "__main__":
    main()
