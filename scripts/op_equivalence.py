#!/usr/bin/env python
"""Equivalence, not eyeballing — and how much hangs on any one operator.

Two statistical upgrades, both pure recomputes from persisted artifacts (no GPU):

(1) PAIRED EQUIVALENCE for the composition claims. "Composed matches the donor"
    and "adding the interaction term does not help" were read off nearby
    percentages. Here: per-cell paired differences of the generation indicator
    (short-decode containment), a hierarchical bootstrap CI on the mean difference
    (operators resampled dyadically, operands nested — the paper's estimator), and
    a TOST equivalence call against a PRE-SPECIFIED margin of +/-5 percentage
    points (equivalent iff the 90% CI of the paired difference lies inside it;
    equivalently the 95% percentile bounds used one-sidedly). Comparisons:
      composed vs donor;  full(=composed+interaction) vs composed;
      composed vs clean ceiling (unpaired across conditions in the clean case --
      clean accuracy is a per-(pair,operand) indicator too, so it pairs by cell).

(2) LEAVE-ONE-OPERATOR-OUT for the swap contrast. The family bootstrap has only
    five operator clusters; dropping each operator in turn and recomputing the
    contrast shows no single operator carries the paradigm.

    python scripts/op_equivalence.py 1.7b
    python scripts/op_equivalence.py 8b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import op_core

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "results" / "ablation"
MARGIN = 0.05  # pre-specified equivalence margin: +/-5 percentage points


def paired_boot(df_a, df_b, col="exact_match", n_boot=10_000, seed=0):
    """Hierarchical bootstrap of mean(a - b) over paired cells: operators resampled
    dyadically (a pair enters with multiplicity count(from)*count(to)), operands
    resampled within pairs -- op_core.bootstrap_family_ci's scheme applied to a
    paired per-cell difference."""
    rng = np.random.default_rng(seed)
    m = df_a.merge(df_b, on=["from", "to", "operand"], suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_a"].astype(float) - m[f"{col}_b"].astype(float)
    ops = list(dict.fromkeys(m["from"]))
    groups = {k: g["d"].to_numpy() for k, g in m.groupby(["from", "to"], sort=False)}
    reps = np.empty(n_boot)
    for b in range(n_boot):
        cnt = {k: 0 for k in ops}
        for k in rng.choice(ops, size=len(ops), replace=True):
            cnt[k] += 1
        tot_w = tot = 0.0
        for (f, t), d in groups.items():
            w = cnt[f] * cnt[t]
            if w == 0:
                continue
            idx = rng.integers(0, len(d), size=len(d))
            tot_w += w
            tot += w * d[idx].mean()
        reps[b] = tot / tot_w if tot_w else np.nan
    obs = float(m["d"].mean())
    lo95, hi95 = np.nanpercentile(reps, [2.5, 97.5])
    lo90, hi90 = np.nanpercentile(reps, [5.0, 95.0])
    return {"n_cells": int(len(m)), "diff": obs,
            "ci95": [float(lo95), float(hi95)],
            "ci90": [float(lo90), float(hi90)],
            "tost_equivalent_5pp": bool(-MARGIN < lo90 and hi90 < MARGIN)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    tag = args.model

    # ---------- (1) paired equivalence on the composition ladder ----------
    p = ABL / f"{tag}_{args.domain}_patch_decomp.parquet"
    df = pd.read_parquet(p)
    keys = ["from", "to", "operand"]
    pick = lambda v: df[df["variant"] == v][keys + ["exact_match"]]
    clean = df.drop_duplicates(keys)[keys + ["clean_exact_from"]].rename(
        columns={"clean_exact_from": "exact_match"})

    comparisons = {
        "composed - donor": (pick("operator + operand"), pick("full (donor)")),
        "full - composed (interaction adds?)": (pick("full (donor)"),
                                                pick("operator + operand")),
        "composed - clean ceiling": (pick("operator + operand"), clean),
        "held-out composed - donor": (pick("operator + operand (held-out cell)"),
                                      pick("full (donor)")),
    }
    out = {"margin_pp": MARGIN * 100}
    print(f"[{tag} / {args.domain}] paired equivalence "
          f"(margin +/-{MARGIN:.0%}, hierarchical bootstrap):")
    for name, (a, b) in comparisons.items():
        r = paired_boot(a, b, seed=args.seed)
        out[name] = r
        verdict = "EQUIVALENT" if r["tost_equivalent_5pp"] else "not shown equivalent"
        print(f"  {name:<38} diff {r['diff']:+.3f} "
              f"95% CI [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
              f"90% CI [{r['ci90'][0]:+.3f}, {r['ci90'][1]:+.3f}]  -> {verdict}")

    # ---------- (2) leave-one-operator-out on the swap contrast ----------
    p = ABL / f"{tag}_{args.domain}_operator_swap_long.parquet"
    ldf = pd.read_parquet(p)
    ops = list(dict.fromkeys(ldf["from"]))
    loo = {}
    print(f"\nleave-one-operator-out swap contrast "
          f"(full paradigm: {op_core.bootstrap_family_ci(ldf, seed=args.seed)['contrast_mean']:+.2f}):")
    for k in ops:
        sub = ldf[(ldf["from"] != k) & (ldf["to"] != k)]
        fam = op_core.bootstrap_family_ci(sub, seed=args.seed)
        loo[k] = {"contrast": fam["contrast_mean"],
                  "lo": fam["contrast_lo"], "hi": fam["contrast_hi"],
                  "flip_frac": fam["flip_frac"], "n_pairs": fam["n_pairs"]}
        print(f"  drop {k:<10} contrast {fam['contrast_mean']:+.2f} "
              f"[{fam['contrast_lo']:+.2f}, {fam['contrast_hi']:+.2f}] "
              f"flips {fam['flip_frac']:.2f} ({fam['n_pairs']} pairs)")
    vals = [v["contrast"] for v in loo.values()]
    out["loo_operator"] = {**loo, "_range": [min(vals), max(vals)]}
    print(f"  range across drops: [{min(vals):+.2f}, {max(vals):+.2f}]")

    dst = ABL / f"{tag}_{args.domain}_equivalence.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {dst}")


if __name__ == "__main__":
    main()
