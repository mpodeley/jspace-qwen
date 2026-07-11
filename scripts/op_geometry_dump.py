#!/usr/bin/env python
"""Dump the operator/operand GEOMETRY to disk for the paper figures and the
interactive explorer. Everything here is print-only in operator_factorize.py /
operator_paradigm.py; this script persists it so plots.py and the in-repo HTML
explorer can render without re-running the model.

Per (model, domain) it computes and saves:
  - factorize() at the query token (-1) and the operand/country token (-2):
    stem/case/interaction variance shares + principal angles.
  - a 2-D PCA point cloud of the 60 workspace vectors H[operand, operator] at
    BOTH read positions (query-token cloud Procrustes-aligned to the country-token
    cloud so the two layouts morph smoothly) -- this is the "declension" figure.
  - the operator-direction cosine matrix (op_dirs) -- the syncretism geometry.
  - pure_desinence() (relations only).
  - the all-pairs swap matrix (read back from the existing *_operator_swap.parquet).

Outputs:
  results/geometry/{tag}_{domain}.json        (all numbers, for plots.py)
  results/geometry/{tag}_{domain}.npz         (raw H grids + coords)
  docs/interactive/declension.data.js         (window.DATA = {...} for the explorer;
                                               written for the relations/1.7b run)

    python scripts/op_geometry_dump.py 1.7b --domain relations
    python scripts/op_geometry_dump.py 8b  --domain relations
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core

ROOT = Path(__file__).resolve().parent.parent


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def workspace_mean_grid(model, ws, dom) -> tuple[list, list, np.ndarray, np.ndarray]:
    """H[operand, operator] as the workspace-mean residual, at the query token (-1)
    and the operand/country token (-2). Returns (operands, operators, Hq, Hc) where
    the H arrays are (n_operand*n_operator, d_model), row-major over (operand, op)."""
    operands, ops = dom.operand_keys, dom.op_keys
    rows_q, rows_c = [], []
    for o in operands:
        for k in ops:
            p = dom.render(o, k)
            rq = op_core.resid(model, ws, p, -1)
            rc = op_core.resid(model, ws, p, -2)
            rows_q.append(torch.stack([rq[l] for l in ws]).mean(0).cpu().numpy())
            rows_c.append(torch.stack([rc[l] for l in ws]).mean(0).cpu().numpy())
    return operands, ops, np.stack(rows_q), np.stack(rows_c)


def pca2(X: np.ndarray) -> np.ndarray:
    """Top-2 PCA coordinates of rows of X (centered)."""
    Xc = X - X.mean(0, keepdims=True)
    U, S, _Vt = np.linalg.svd(Xc, full_matrices=False)
    return U[:, :2] * S[:2]


def procrustes_align(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Orthogonal R (rotation+reflection) so B@R best matches A (both 2-D, centered).
    Keeps the two clouds in a common frame so the morph doesn't spin/flip."""
    Bc, Ac = B - B.mean(0), A - A.mean(0)
    U, _S, Vt = np.linalg.svd(Bc.T @ Ac)
    return Bc @ (U @ Vt)


def op_direction_cosines(model, ws, dom) -> tuple[list, np.ndarray]:
    """Per-operator direction (workspace-mean of op_dirs) and their cosine matrix."""
    v = op_core.op_dirs(model, ws, dom)
    ops = dom.op_keys
    D = {k: torch.stack([v[k][l] for l in ws]).mean(0).cpu().numpy() for k in ops}
    M = np.eye(len(ops))
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            M[i, j] = _cos(D[a], D[b])
    return ops, M


def read_swap_matrix(tag: str, domain: str) -> dict:
    """The all-pairs swap from the existing parquet, as {from:{to:{clean,swap,random,n}}}."""
    import pandas as pd
    p = ROOT / "results" / "ablation" / f"{tag}_{domain}_operator_swap.parquet"
    if not p.exists():
        print(f"  (no swap parquet at {p}; run operator_paradigm.py first)")
        return {}
    df = pd.read_parquet(p)
    out: dict = {}
    for _, r in df.iterrows():
        out.setdefault(r["from"], {})[r["to"]] = {
            "clean": float(r["clean"]), "swap": float(r["swap"]),
            "random": float(r["random"]), "n": int(r["n"])}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--no-explorer", action="store_true",
                    help="skip writing docs/interactive/declension.data.js")
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    dom = op_core.load_domain(args.domain)
    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)

    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]
    L = ws[len(ws) // 2]

    print(f"[{tag} / {args.domain}] workspace layers {ws[0]}..{ws[-1]}, factorize layer {L}")

    # (A) variance factorization at query and country positions
    fac_q = op_core.factorize(model, None, L, -1, dom, False)
    fac_c = op_core.factorize(model, None, L, -2, dom, False)
    print(f"  query  : stem={fac_q['stem']:.1%} case={fac_q['case']:.1%} "
          f"inter={fac_q['interaction']:.1%}")
    print(f"  country: stem={fac_c['stem']:.1%} case={fac_c['case']:.1%} "
          f"inter={fac_c['interaction']:.1%}")

    # (B) PCA point clouds (declension morph), query aligned onto country frame
    operands, ops, Hq, Hc = workspace_mean_grid(model, ws, dom)
    coords_c = pca2(Hc)
    coords_q = procrustes_align(pca2(Hq), coords_c)

    # (C) operator-direction cosines (syncretism)
    cos_ops, cosM = op_direction_cosines(model, ws, dom)

    # (D) pure desinence (relations only)
    des = op_core.pure_desinence(model, ws, dom, model.tokenizer)

    # (E) swap matrix from the persisted parquet
    swap = read_swap_matrix(tag, args.domain)

    # --- persist -------------------------------------------------------------
    gdir = ROOT / "results" / "geometry"
    gdir.mkdir(parents=True, exist_ok=True)

    def cloud(coords):
        return [{"operand": operands[i // len(ops)], "operator": ops[i % len(ops)],
                 "x": float(coords[i, 0]), "y": float(coords[i, 1])}
                for i in range(len(coords))]

    # operator centroids at the query token (for the injection animation)
    centroids_q = {}
    for j, k in enumerate(ops):
        idx = [i for i in range(len(coords_q)) if i % len(ops) == j]
        centroids_q[k] = [float(coords_q[idx, 0].mean()), float(coords_q[idx, 1].mean())]

    data = {
        "model": tag, "domain": args.domain,
        "operands": operands, "operators": ops,
        "variance": {"query": fac_q, "country": fac_c},
        "cloud_country": cloud(coords_c),
        "cloud_query": cloud(coords_q),
        "centroids_query": centroids_q,
        "cos": {"labels": cos_ops, "matrix": cosM.tolist()},
        "desinence": des,
        "swap": swap,
    }
    (gdir / f"{tag}_{args.domain}.json").write_text(json.dumps(data, indent=2))
    np.savez(gdir / f"{tag}_{args.domain}.npz", Hq=Hq, Hc=Hc,
             coords_q=coords_q, coords_c=coords_c)
    print(f"  saved results/geometry/{tag}_{args.domain}.json (+ .npz)")

    # explorer bundle: written for the canonical relations run
    if not args.no_explorer and args.domain == "relations":
        idir = ROOT / "docs" / "interactive"
        idir.mkdir(parents=True, exist_ok=True)
        js = "window.DATA = " + json.dumps(data) + ";\n"
        (idir / "declension.data.js").write_text(js)
        print(f"  saved docs/interactive/declension.data.js")


if __name__ == "__main__":
    main()
