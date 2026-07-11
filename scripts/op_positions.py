#!/usr/bin/env python
"""Where does the operator intervention have to land -- and is the effect a
localized edit of the queried relation, or a global perturbation of processing?

The all-position additive injection used elsewhere is open to a serious objection:
adding a vector at EVERY token of every layer in a band could move the margin by
perturbing processing broadly rather than by modifying the relation being queried.
This script separates those hypotheses by crossing WHERE with HOW WIDE:

  positions   all      every prompt token (the default used in the paper)
              query    the last prompt token only ("is") -- where v(op) is read
              operand  the entity token (-2) -- where the operand main effect lives
              wrong    the sentence-initial token (0) -- structurally irrelevant to
                       the relation, but still upstream of the answer (so it CAN
                       perturb; a null here is informative, not trivial)
  layers      band     the whole workspace band
              single   one mid-workspace layer

For every condition we report the reviewer's full metric set rather than flips
alone: absolute and normalized margin change, target rank before/after, top-1,
greedy exact match, on-task KL, off-task KL on unrelated text, and the fraction of
cells whose sign is correct -- with cluster-bootstrap CIs (operators as the
top-level unit, operands nested).

    python scripts/op_positions.py 1.7b --domain relations
    python scripts/op_positions.py 8b --domain animals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, get_corpus, load_model, resolve_tag
import op_core
import op_dose

ROOT = Path(__file__).resolve().parent.parent

# name -> prompt indices for the additive hook (None = every position)
POSITIONS = {"all": None, "query": [-1], "operand": [-2], "wrong": [0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--n-corpus", type=int, default=8,
                    help="unrelated WikiText prompts for the off-task KL")
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--dose-alphas", nargs="*", type=float,
                    default=[0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                    help="alphas for the dose x position exact-match sweep. The small "
                         "end matters: injecting into a 13-layer band ACCUMULATES down "
                         "the residual stream, so the on-manifold dose for a band is "
                         "far below the on-manifold dose for a single layer (~1).")
    ap.add_argument("--no-dose", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]
    scopes = {"band": ws, "single": [ws[len(ws) // 2]]}
    ops = dom.op_keys
    pairs = [(a, b) for a in ops for b in ops if a != b]

    print(f"[{tag} / {args.domain}] alpha={args.alpha}; band={ws[0]}..{ws[-1]} "
          f"({len(ws)} layers), single layer={scopes['single'][0]}")

    v = op_core.op_dirs(model, ws, dom)

    rows = []
    for scope, layers in scopes.items():
        for pos_name, pos in POSITIONS.items():
            for frm, to in pairs:
                dv = {l: args.alpha * (v[to][l] - v[frm][l]) for l in layers}
                for o in dom.operand_keys:
                    af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
                    if af == at:
                        continue
                    mb = op_core.metric_bundle(model, dom, tok, o, frm, to,
                                               add=dv, add_positions=pos)
                    rows.append({"scope": scope, "position": pos_name,
                                 "from": frm, "to": to, "operand": o, **mb})

    df = pd.DataFrame(rows)

    # off-task KL: the same hook (same layers, same position restriction) run on
    # unrelated text, averaged over 3 representative pairs.
    kl_off = {}
    if not args.no_kl:
        corpus = [p[:600] for p in get_corpus(args.n_corpus)]
        for scope, layers in scopes.items():
            for pos_name, pos in POSITIONS.items():
                tot = 0.0
                for frm, to in pairs[:3]:
                    dv = {l: args.alpha * (v[to][l] - v[frm][l]) for l in layers}
                    k, _ = op_dose.collateral(model, corpus, dv, positions=pos)
                    tot += k / 3
                kl_off[(scope, pos_name)] = tot

    print(f"\n{'scope':<7} {'position':<9} {'Δmargin':>18} {'norm':>7} {'rank(to)':>16} "
          f"{'top-1':>6} {'exact':>6} {'sign+':>6} {'KL on':>7} {'KL off':>7}")
    summary = {}
    for scope in scopes:
        for pos_name in POSITIONS:
            s = df[(df["scope"] == scope) & (df["position"] == pos_name)]
            # bootstrap the Δmargin (swap vs its own clean baseline) with the
            # family bootstrap: operators top-level, operands nested.
            ldf = pd.DataFrame({"from": s["from"], "to": s["to"],
                                "operand": s["operand"],
                                "swap": s["margin"], "random": s["margin_clean"]})
            fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
            rec = {
                "n": int(len(s)),
                "delta_margin": float(s["delta_margin"].mean()),
                "delta_margin_lo": fam["contrast_lo"],
                "delta_margin_hi": fam["contrast_hi"],
                # median: the per-cell ratio explodes when |clean margin| ~ 0
                "norm_margin": float(s["norm_margin"].median()),
                "rank_to_clean": float(s["rank_to_clean"].median()),
                "rank_to": float(s["rank_to"].median()),
                "top1": float(s["top1"].mean()),
                "exact_match": float(s["exact_match"].mean()),
                "sign_correct": float((s["margin"] > 0).mean()),
                "flip_frac": fam["flip_frac"],
                "kl_ontask": float(s["kl_ontask"].mean()),
                "kl_offtask": kl_off.get((scope, pos_name)),
            }
            summary[f"{scope}/{pos_name}"] = rec
            ko = f"{rec['kl_offtask']:.2f}" if rec["kl_offtask"] is not None else "  -  "
            print(f"{scope:<7} {pos_name:<9} "
                  f"{rec['delta_margin']:>+7.2f} [{rec['delta_margin_lo']:+.1f},"
                  f"{rec['delta_margin_hi']:+.1f}] {rec['norm_margin']:>+7.2f} "
                  f"{rec['rank_to_clean']:>7.0f}->{rec['rank_to']:<7.0f} "
                  f"{rec['top1']:>5.0%} {rec['exact_match']:>6.1%} "
                  f"{rec['sign_correct']:>5.0%} {rec['kl_ontask']:>7.2f} {ko:>7}")
    print()

    # --- dose x position, on GENERATION ---------------------------------------
    # The paper reported ~0% exact match for additive steering at alpha=4. But the
    # operator direction IS the ANOVA operator main effect -- op_dirs computes
    # mean_o[h(o,k) - mean_k h(o,k)] = mean_o h(o,k) - mu = case[k], identically. So
    # adding alpha*(case[to] - case[frm]) at the query position of ONE layer gives
    #
    #     h(o,frm) + alpha*(case[to] - case[frm])
    #       = mu + stem[o] + (1-alpha)*case[frm] + alpha*case[to] + inter[(o,frm)]
    #
    # which at alpha=1 is the composed target state up to the (wrong) interaction
    # term -- and op_patch_decomp shows that composed state generates. At alpha=4 it
    # is 4*case[to] - 3*case[frm]: a quadruple-magnitude, off-manifold state. So the
    # generation failure may be a DOSE artifact, not an information deficit.
    # Injecting into the whole band accumulates the offset down the residual stream
    # (~one addition per layer), so the band's on-manifold dose is far below 1 --
    # hence the small alphas in the default sweep. This is the experiment that
    # decides whether "influence without sufficiency" is a fact about the
    # representation or about how we intervened on it.
    dose_rows = []
    if not args.no_dose:
        conds = [("band", "all"), ("band", "query"), ("single", "query")]
        for scope, pos_name in conds:
            layers, pos = scopes[scope], POSITIONS[pos_name]
            for a in args.dose_alphas:
                hits = margins = n = 0
                for frm, to in pairs:
                    dv = {l: a * (v[to][l] - v[frm][l]) for l in layers}
                    for o in dom.operand_keys:
                        af, at = (dom.answer_tok(tok, o, frm),
                                  dom.answer_tok(tok, o, to))
                        if af == at:
                            continue
                        mb = op_core.metric_bundle(model, dom, tok, o, frm, to,
                                                   add=dv, add_positions=pos)
                        hits += mb["exact_match"]
                        margins += mb["delta_margin"]
                        n += 1
                dose_rows.append({"scope": scope, "position": pos_name, "alpha": a,
                                  "exact_match": hits / n, "delta_margin": margins / n,
                                  "n": n})
                r = dose_rows[-1]
                print(f"  dose {scope:>6}/{pos_name:<5} α={a:>4.1f}  "
                      f"exact match {r['exact_match']:>6.1%}  "
                      f"Δmargin {r['delta_margin']:>+7.2f}")
        summary["_dose"] = dose_rows

    summary["_meta"] = {
        "tag": tag, "domain": args.domain, "alpha": args.alpha,
        "ws": [int(l) for l in ws], "single_layer": int(scopes["single"][0]),
        "positions": {k: (v_ if v_ else "all") for k, v_ in POSITIONS.items()},
        "bootstrap_unit": "operator (dyadic node), operands nested",
        "note": "rank columns are medians; Δmargin CIs are 95% cluster bootstrap",
    }
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_positions.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    if dose_rows:
        pd.DataFrame(dose_rows).to_parquet(
            out.with_name(f"{tag}_{args.domain}_posdose.parquet"))
    print(f"\nsaved {out} (+ .json)")


if __name__ == "__main__":
    main()
