#!/usr/bin/env python
"""Export the showcase data bundle for the assembled-thought site.

Joins the persisted artifacts (no torch, no model, no GPU) into one JS blob that
drives the "Assemble a thought" interactive scene:

  - real k=8 generations per cell (clean / composed / overdose) from the audit,
  - ladder stats (exact match, margins) from the patch decomposition,
  - 2D coordinates in the SAME frame as declension.data.js's query-token cloud,
    so both explorers share one geometry: the projection is recovered by exact
    least squares from the saved workspace-mean grid Hq onto the saved coords_q
    (never re-run SVD -- sign/rotation stability across numpy versions is not
    guaranteed; the saved coords are the ground truth).

Composed points are mu + stem(o) + case(k) of the workspace-mean grid; overdose
points are the actual overdosed state Hq[o,from] + 4*(case(to) - case(from)).
Because the projection is linear, projecting the high-dim sums is exact.

    .venv/bin/python scripts/export_showcase_data.py 1.7b --domain relations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRIM = 64  # chars per stored generation

# source prompt per target relation for the scene's default (a fixed derangement
# that never lands on a syncretic missing cell)
DEFAULT_FROM = {"capital": "currency", "currency": "capital",
                "language": "continent", "demonym": "capital",
                "continent": "currency"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="1.7b")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--dest", default=None)
    args = ap.parse_args()
    tag = args.model

    audit = pd.read_parquet(ROOT / f"results/ablation/{tag}_{args.domain}_audit.parquet")
    patch = pd.read_parquet(ROOT / f"results/ablation/{tag}_{args.domain}_patch_decomp.parquet")
    pj = json.loads((ROOT / f"results/ablation/{tag}_{args.domain}_patch_decomp.json").read_text())
    aj = json.loads((ROOT / f"results/ablation/{tag}_{args.domain}_audit.json").read_text())
    z = np.load(ROOT / f"results/geometry/{tag}_{args.domain}.npz")
    geo = json.loads((ROOT / "docs/interactive/declension.data.js")
                     .read_text().split("window.DATA = ", 1)[1].rstrip(";\n"))
    rel = json.loads((ROOT / f"data/{args.domain}.json").read_text())

    operands, ops = geo["operands"], geo["operators"]
    Hq, coords_q = z["Hq"].astype(np.float64), z["coords_q"].astype(np.float64)
    assert Hq.shape[0] == len(operands) * len(ops)

    # --- exact linear projection onto the saved 2D frame ----------------------
    mean = Hq.mean(0)
    X = Hq - mean
    W, res, rank, _ = np.linalg.lstsq(X, coords_q, rcond=None)
    recon = X @ W
    proj_err = float(np.abs(recon - coords_q).max())
    assert proj_err < 1e-6, f"projection does not reproduce saved coords: {proj_err}"
    proj = lambda v: ((np.asarray(v, dtype=np.float64) - mean) @ W).tolist()

    # --- grid components (workspace-mean) -------------------------------------
    row = {(o, k): Hq[i * len(ops) + j]
           for i, o in enumerate(operands) for j, k in enumerate(ops)}
    mu = Hq.mean(0)
    stem = {o: np.mean([row[(o, k)] for k in ops], axis=0) - mu for o in operands}
    case = {k: np.mean([row[(o, k)] for o in operands], axis=0) - mu for k in ops}

    asm2d = {f"{o}|{k}": proj(mu + stem[o] + case[k]) for o in operands for k in ops}
    # sanity: >=90% of assembled points land nearest their own operator centroid
    cent = {k: np.array(geo["centroids_query"][k]) for k in ops}
    ok = 0
    for o in operands:
        for k in ops:
            p2 = np.array(asm2d[f"{o}|{k}"])
            nearest = min(ops, key=lambda kk: float(np.linalg.norm(p2 - cent[kk])))
            ok += (nearest == k)
    frac = ok / (len(operands) * len(ops))
    assert frac >= 0.9, f"assembled points off-cluster: only {frac:.0%} nearest own centroid"

    # --- per-cell texts + stats ------------------------------------------------
    aud = {(r["from"], r["to"], r["operand"], r["condition"]): r
           for r in audit.to_dict("records")}
    pat = {(r["from"], r["to"], r["operand"], r["variant"]): r
           for r in patch.to_dict("records")}

    cells = {}
    keys = sorted({(r["from"], r["to"], r["operand"])
                   for r in audit.to_dict("records")})
    for frm, to, o in keys:
        a_comp = aud[(frm, to, o, "patch composed")]
        a_od = aud[(frm, to, o, "add band a=4")]
        p_comp = pat[(frm, to, o, "operator + operand")]
        od_state = row[(o, frm)] + 4.0 * (case[to] - case[frm])
        cells[f"{frm}|{to}|{o}"] = {
            "clean": aud[(frm, to, o, "clean")]["text"][:TRIM],
            "composed": a_comp["text"][:TRIM],
            "cls": a_comp["class"],
            "od": a_od["text"][:TRIM],
            "od_fc": bool(a_od["forced_choice_target"]),
            "od_says": a_od["forced_choice_says"],
            "em": bool(p_comp["exact_match"]),
            "m": [round(float(p_comp["clean_margin"]), 1),
                  round(float(p_comp["margin"]), 1)],
            "od2d": [round(x, 2) for x in proj(od_state)],
        }
    assert len(cells) == 224, len(cells)
    for k in ops:
        for o in operands:
            has = any(kk.endswith(f"|{k}|{o}") for kk in cells)
            if not has:
                # syncretic gaps are per-(from,to); every (to, operand) must still
                # be reachable from at least one source
                raise AssertionError(f"no source for ({o},{k})")
    for to, frm in DEFAULT_FROM.items():
        n = sum(1 for kk in cells if kk.startswith(f"{frm}|{to}|"))
        assert n == 12, f"default_from {frm}->{to} covers {n}/12 operands"

    stats = {
        "composed": round(pj["operator + operand"]["exact_match"], 3),
        "donor": round(pj["full (donor)"]["exact_match"], 3),
        "ceiling": round(pj["_meta"]["clean_exact_from"], 3),
        "heldout_cell": round(pj["operator + operand (held-out cell)"]["exact_match"], 3),
        "wrong_operand": round(pj["operator + wrong operand"]["says_other_operand"], 3),
        "wrong_operand_base": round(pj["operator + operand"]["says_other_operand"], 3),
        "od_fc": round(aj["add band a=4"]["forced_choice_target"], 3),
        "od_degraded": round(aj["add band a=4"]["degraded"], 3),
        "calibrated": round(aj["add band a=0.1"]["target"], 3),
        "interaction_share": 0.09,
    }

    data = {
        "model": tag, "domain": args.domain,
        "template": rel["meta"]["template"],
        "operators": ops, "operands": operands,
        "answers": {o: rel["items"][o]["answers"] for o in operands},
        "cloud": geo["cloud_query"],
        "centroids": geo["centroids_query"],
        "mu2d": [round(x, 2) for x in proj(mu)],
        "stem2d": {o: [round(x, 2) for x in proj(mu + stem[o])] for o in operands},
        "asm2d": {k: [round(x, 2) for x in v] for k, v in asm2d.items()},
        "real2d": {f"{o}|{k}": [round(x, 2) for x in proj(row[(o, k)])]
                   for o in operands for k in ops},
        "default_from": DEFAULT_FROM,
        "cells": cells,
        "stats": stats,
    }

    dest = Path(args.dest) if args.dest else ROOT / "results/showcase/composition.data.js"
    dest.parent.mkdir(parents=True, exist_ok=True)
    blob = "window.COMP = " + json.dumps(data, separators=(",", ":")) + ";\n"
    dest.write_text(blob)
    kb = len(blob) / 1024
    assert kb < 100, f"blob too large: {kb:.0f}KB"
    print(f"wrote {dest} ({kb:.0f}KB): {len(cells)} cells, proj err {proj_err:.1e}, "
          f"{frac:.0%} assembled points on-cluster")
    print("copy into the site with:")
    print(f"  cp {dest} /var/home/matias/Projects/assembled-thought/docs/interactive/")


if __name__ == "__main__":
    main()
