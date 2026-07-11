#!/usr/bin/env python
"""What does the donor activation carry that the operator direction does not?

The paper's central tension: adding the averaged operator direction v(to)-v(frm)
robustly flips the target-vs-source logit MARGIN, but essentially never makes the
model EMIT the target answer; replacing the query-position residual with a real
donor activation does (op_minimal). "The patch has more information" is not an
explanation until we say WHICH information. So decompose the donor.

At the query position, the two-way factorization is exact by construction:

    H[(operand, op)] = mu + stem[operand] + case[op] + inter[(operand, op)]

Patch partial reconstructions back in and ask which of them restores generation:

    operator only        mu + case[to]                       (relation, no entity)
    operand only         mu + stem[o]                        (entity, no relation)
    operator + operand   mu + stem[o] + case[to]             (purely additive)
    interaction only     mu + inter[(o,to)]                  (the fusion term alone)
    full (== donor)      mu + stem[o] + case[to] + inter     (identity: reproduces
                                                              op_minimal's patch)
    magnitude control    mu + random, norm-matched to full   (is it just size?)

mu is included in EVERY variant: the hook REPLACES the residual, so a variant must
be a plausible full-magnitude state -- dropping the grand mean would ablate the
shared structure every real activation has and confound the comparison.

Reading:
  * if operator+operand does not generate but full does, the missing ingredient is
    the non-additive interaction term -- influence without behavioral sufficiency,
    and the additive picture is incomplete exactly where behavior is decided;
  * if operator-only already generates, the operator subspace IS behaviorally
    sufficient at the query position, and the additive-steering null was an artifact
    of averaging/offset, not of missing information;
  * if only the magnitude control matters, the effect is norm, not content.

    python scripts/op_patch_decomp.py 1.7b --domain relations
    python scripts/op_patch_decomp.py 8b --domain animals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core
import op_minimal

ROOT = Path(__file__).resolve().parent.parent

VARIANTS = ["operator only", "operand only", "operator + operand",
            "operator + operand (held-out cell)", "operator + wrong operand",
            "interaction only", "full (donor)", "magnitude control"]


def build_variants(comp, loo, ws, o, to, o_other, g, dev, d_model):
    """{variant: {layer: replacement residual}} for one (operand, target op) cell.

    `loo` holds the leave-one-cell-out components: the additive reconstruction of
    cell (o,to) built WITHOUT ever looking at cell (o,to). That variant is the
    decisive one -- the plain "operator + operand" reconstruction is contaminated,
    because stem[o] averages over o's cells (which include (o,to)) and case[to]
    averages over to's cells (likewise). If a state composed from parts that never
    saw the cell still makes the model emit that cell's answer, the composition is
    doing the work, not a leaked memory of the target."""
    mu, stem, case, inter = comp["mu"], comp["stem"], comp["case"], comp["inter"]
    V = {
        "operator only": {l: mu[l] + case[l][to] for l in ws},
        "operand only": {l: mu[l] + stem[l][o] for l in ws},
        "operator + operand": {l: mu[l] + stem[l][o] + case[l][to] for l in ws},
        "operator + operand (held-out cell)": {
            l: loo["mu"][l] + loo["stem"][l] + loo["case"][l] for l in ws},
        # specificity: the same operator component with ANOTHER operand's stem --
        # should install the relation on the wrong entity (and say ITS answer).
        "operator + wrong operand": {l: mu[l] + stem[l][o_other] + case[l][to]
                                     for l in ws},
        "interaction only": {l: mu[l] + inter[l][(o, to)] for l in ws},
        "full (donor)": {l: mu[l] + stem[l][o] + case[l][to] + inter[l][(o, to)]
                         for l in ws},
    }
    # magnitude control: mu + a random direction whose norm equals the full patch's
    # deviation from mu, per layer -- same size of departure from the grand mean,
    # no operator/operand/fusion content.
    V["magnitude control"] = {}
    for l in ws:
        dev_norm = (V["full (donor)"][l] - mu[l]).norm()
        r = torch.randn(d_model, generator=g).to(dev)
        V["magnitude control"][l] = mu[l] + r / (r.norm() + 1e-9) * dev_norm
    return V


def loo_components(H, ws, ops, operands, o, to):
    """Two-way main effects of the grid with cell (o,to) REMOVED. Unbalanced by one
    cell, so the main effects are taken over the surviving cells of that row/column
    and mu over the surviving grid -- the standard leave-one-out reconstruction."""
    keep = [(a, k) for a in operands for k in ops if not (a == o and k == to)]
    out = {"mu": {}, "stem": {}, "case": {}}
    for l in ws:
        mu = torch.stack([H[(a, k)][l] for a, k in keep]).mean(0)
        row = [H[(o, k)][l] for k in ops if k != to]            # o's other operators
        col = [H[(a, to)][l] for a in operands if a != o]       # to's other operands
        out["mu"][l] = mu
        out["stem"][l] = torch.stack(row).mean(0) - mu
        out["case"][l] = torch.stack(col).mean(0) - mu
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--k", type=int, default=3, help="greedy tokens for exact match")
    ap.add_argument("--scope", choices=["band", "single"], default="band",
                    help="write the composed state into the whole workspace band, or "
                         "into a single mid-workspace layer (the minimal version)")
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    band = band_layers(source_layers, model.n_layers)["workspace"]
    ws = band if args.scope == "band" else [band[len(band) // 2]]
    ops = dom.op_keys
    pairs = [(a, b) for a in ops for b in ops if a != b]
    g = torch.Generator().manual_seed(args.seed)

    print(f"[{tag} / {args.domain}] donor decomposition at the query position, "
          f"scope={args.scope}: layers {ws[0]}..{ws[-1]} ({len(ws)})")

    # component vectors of the exact two-way factorization, at every ws layer
    comp = op_core.factorize_components(model, ws, dom, pos=-1)
    dev = comp["mu"][ws[0]].device
    # the raw grid, for the leave-one-cell-out reconstruction
    H = {(o, k): op_core.resid(model, ws, dom.render(o, k), -1)
         for o in dom.operand_keys for k in ops}

    rows = []
    for frm, to in pairs:
        for i, o in enumerate(dom.operand_keys):
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            p = dom.render(o, frm)
            ansA, ansB = str(dom.answer(o, frm)), str(dom.answer(o, to))
            # a fixed other operand for the specificity control (next one round-robin)
            o_other = dom.operand_keys[(i + 1) % len(dom.operand_keys)]
            ansOther = str(dom.answer(o_other, to))
            L0 = op_core.final_logits(model, p)
            clean_txt = op_minimal.greedy(model, p, k=args.k)
            loo = loo_components(H, ws, ops, dom.operand_keys, o, to)
            V = build_variants(comp, loo, ws, o, to, o_other, g, dev, model.d_model)
            base = {"from": frm, "to": to, "operand": o, "other_operand": o_other,
                    "clean_margin": float(L0[at] - L0[af]),
                    "clean_rank_to": int((L0 > L0[at]).sum()),
                    "clean_exact_from": op_minimal.hit(clean_txt, ansA),
                    "clean_exact_to": op_minimal.hit(clean_txt, ansB)}
            for name in VARIANTS:
                mb = op_core.metric_bundle(model, dom, tok, o, frm, to,
                                           patch=V[name], clean_logits=L0, k=args.k)
                # what did it actually SAY? target, source, or the other operand's
                # answer for the target relation (the specificity readout)
                txt = mb["text"]
                rows.append({**base, "variant": name,
                             "margin": mb["margin"],
                             "delta_margin": mb["delta_margin"],
                             "norm_margin": mb["norm_margin"],
                             "rank_to": mb["rank_to"],
                             "top1": mb["top1"],
                             "kl_ontask": mb["kl_ontask"],
                             "exact_match": mb["exact_match"],
                             "says_source": op_minimal.hit(txt, ansA),
                             "says_other_operand": op_minimal.hit(txt, ansOther)})

    df = pd.DataFrame(rows)
    n_cells = df.groupby("variant").size().iloc[0]
    clean_em = df.drop_duplicates(["from", "to", "operand"])["clean_exact_from"].mean()

    print(f"\n{n_cells} (pair, operand) cells; clean greedy says the SOURCE answer "
          f"in {clean_em:.0%} (the model's own competence ceiling)\n")
    print(f"{'variant':<36} {'says target':>12} {'says source':>12} {'says other':>11} "
          f"{'Δmargin':>9} {'rank(to)':>9} {'top-1':>7}")
    summary = {}
    for name in VARIANTS:
        s = df[df["variant"] == name]
        ldf = pd.DataFrame({"from": s["from"], "to": s["to"], "operand": s["operand"],
                            "swap": s["margin"], "random": s["clean_margin"]})
        fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
        summary[name] = {
            "n": int(len(s)),
            "exact_match": float(s["exact_match"].mean()),
            "says_source": float(s["says_source"].mean()),
            "says_other_operand": float(s["says_other_operand"].mean()),
            "margin": float(s["margin"].mean()),
            "delta_margin": float(s["delta_margin"].mean()),
            "delta_margin_lo": fam["contrast_lo"], "delta_margin_hi": fam["contrast_hi"],
            "rank_to": float(s["rank_to"].mean()),
            "rank_to_median": float(s["rank_to"].median()),
            "top1": float(s["top1"].mean()),
            "kl_ontask": float(s["kl_ontask"].mean()),
            "flip_frac": fam["flip_frac"],
        }
        r = summary[name]
        print(f"{name:<36} {r['exact_match']:>11.1%} {r['says_source']:>11.1%} "
              f"{r['says_other_operand']:>10.1%} {r['delta_margin']:>+9.2f} "
              f"{r['rank_to']:>9.1f} {r['top1']:>6.1%}")

    summary["_meta"] = {
        "tag": tag, "domain": args.domain, "scope": args.scope,
        "ws": [int(l) for l in ws],
        "n_cells": int(n_cells), "clean_exact_from": float(clean_em),
        "clean_exact_to": float(df.drop_duplicates(["from", "to", "operand"])
                                 ["clean_exact_to"].mean()),
        "clean_margin": float(df["clean_margin"].mean()),
        "clean_rank_to": float(df.drop_duplicates(["from", "to", "operand"])
                                 ["clean_rank_to"].mean()),
        "bootstrap_unit": "operator (dyadic node), operands nested; CIs on Δmargin",
    }

    sfx = "" if args.scope == "band" else f"_{args.scope}"
    out = (ROOT / "results" / "ablation"
           / f"{tag}_{args.domain}_patch_decomp{sfx}.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ .json)")


if __name__ == "__main__":
    main()
