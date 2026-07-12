#!/usr/bin/env python
"""Is there a TRANSFERABLE on-manifold dose — or did we pick alpha post hoc?

The dose x position sweep found generation peaking at alpha ~ 0.1 (13-layer band)
and alpha ~ 1 (single layer). A fair objection: those alphas were selected on the
SAME 224 cells where the peak is reported. This script pre-registers the honest
protocol:

  for each of k fixed partitions of the operands into calibration/evaluation halves:
    1. sweep alpha on the CALIBRATION cells only; pick alpha* = argmax generation
       (long-decode containment, k=8 — the audit's readout);
    2. freeze alpha*; report generation on the EVALUATION cells (operands the
       calibration never saw), at alpha* and, for the curve's shape, at every alpha.

If alpha* is stable across partitions and evaluation generation at alpha* is at the
calibration level, the on-manifold dose is a transferable property of the model and
intervention — not an artifact of peeking. Directions are built from ALL operands
(they are the paper's standard directions; what is being validated out-of-sample is
the DOSE, not the direction — held-out direction transfer is its own experiment).

    python scripts/op_dose_oos.py 1.7b --domain relations
"""

from __future__ import annotations

import argparse
import json
import random as _random
from pathlib import Path

import pandas as pd

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core
import op_minimal

ROOT = Path(__file__).resolve().parent.parent

ALPHAS = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 4.0, 8.0]


def gen_rate(model, dom, tok, v, layers, positions, alpha, operands, k):
    """Long-decode containment rate over the (pair, operand) cells of `operands`."""
    ops = dom.op_keys
    hits = n = 0
    for frm, to in [(a, b) for a in ops for b in ops if a != b]:
        dv = {l: alpha * (v[to][l] - v[frm][l]) for l in layers}
        for o in operands:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            text = op_minimal.greedy(model, dom.render(o, frm), k=k, add=dv,
                                     add_positions=positions)
            hits += op_minimal.hit(text, str(dom.answer(o, to)))
            n += 1
    return hits / n, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--k", type=int, default=8, help="greedy tokens (audit readout)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    ws = band_layers(evenly_spaced_layers(model.n_layers), model.n_layers)["workspace"]
    conds = {"band/all": (ws, None), "single/query": ([ws[len(ws) // 2]], [-1])}

    print(f"[{tag} / {args.domain}] out-of-sample dose: {args.partitions} partitions, "
          f"alphas {ALPHAS}, k={args.k}")

    v = op_core.op_dirs(model, ws, dom)
    operands = dom.operand_keys
    half = len(operands) // 2

    rows, summary = [], {}
    for cname, (layers, positions) in conds.items():
        stars, eval_at_star, eval_at_cal_curve = [], [], []
        for p in range(args.partitions):
            rng = _random.Random(100 + p)
            perm = operands[:]
            rng.shuffle(perm)
            cal, ev = perm[:half], perm[half:]
            # calibration sweep
            cal_curve = {}
            for a in ALPHAS:
                r, n = gen_rate(model, dom, tok, v, layers, positions, a, cal, args.k)
                cal_curve[a] = r
            a_star = max(cal_curve, key=cal_curve.get)
            # frozen evaluation
            ev_curve = {}
            for a in ALPHAS:
                r, n_ev = gen_rate(model, dom, tok, v, layers, positions, a, ev, args.k)
                ev_curve[a] = r
            stars.append(a_star)
            eval_at_star.append(ev_curve[a_star])
            rows.append({"condition": cname, "partition": p, "alpha_star": a_star,
                         "cal_at_star": cal_curve[a_star],
                         "eval_at_star": ev_curve[a_star],
                         "eval_best": max(ev_curve.values()),
                         "eval_best_alpha": max(ev_curve, key=ev_curve.get),
                         **{f"cal_{a}": cal_curve[a] for a in ALPHAS},
                         **{f"eval_{a}": ev_curve[a] for a in ALPHAS}})
            print(f"  {cname:>13} p{p}: cal={sorted(cal)[:2]}..., alpha*={a_star} "
                  f"(cal {cal_curve[a_star]:.1%}) -> eval {ev_curve[a_star]:.1%} "
                  f"(eval best {max(ev_curve.values()):.1%} at "
                  f"{max(ev_curve, key=ev_curve.get)})")
        import statistics as st
        summary[cname] = {
            "alpha_stars": stars,
            "alpha_star_mode": max(set(stars), key=stars.count),
            "eval_at_star_mean": st.mean(eval_at_star),
            "eval_at_star_min": min(eval_at_star),
            "eval_at_star_max": max(eval_at_star),
            "n_partitions": args.partitions,
        }
        s = summary[cname]
        print(f"  => {cname}: alpha* mode {s['alpha_star_mode']} "
              f"(all: {stars}); eval generation at frozen alpha* "
              f"{s['eval_at_star_mean']:.1%} [{s['eval_at_star_min']:.1%}, "
              f"{s['eval_at_star_max']:.1%}]")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_dose_oos.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    summary["_meta"] = {"tag": tag, "domain": args.domain, "alphas": ALPHAS,
                        "k": args.k, "protocol": "calibrate alpha* on half the "
                        "operands (argmax long-decode containment), freeze, evaluate "
                        "on the disjoint half; directions built on all operands (the "
                        "dose, not the direction, is what is validated here)"}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ .json)")


if __name__ == "__main__":
    main()
