#!/usr/bin/env python
"""Held-out generalization beyond one lucky split: multiple partitions + LOO.

The paper's held-out result used one fixed half/half split (insertion order). With
twelve operands it is cheap to close the "favorable partition" objection: run k
shuffled partitions AND full leave-one-operand-out, and report the spread.

    python scripts/op_heldout_parts.py 1.7b --domain relations --partitions 5
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import pandas as pd

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--partitions", type=int, default=5)
    ap.add_argument("--no-loo", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    ws = band_layers(evenly_spaced_layers(model.n_layers), model.n_layers)["workspace"]
    operands = dom.operand_keys

    rows = []

    def run(label, **kw):
        build, test, dfg, ldf = op_core.held_out_generalization(
            model, ws, dom, tok, seed=args.seed, **kw)
        fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
        n = len(dfg.dropna(subset=["swap"]))
        flips = int((dfg["swap"] > 0).sum())
        rows.append({"label": label, "build_n": len(build), "test_n": len(test),
                     "test": ",".join(test), "flips": flips, "n_pairs": n,
                     "contrast": fam["contrast_mean"],
                     "contrast_lo": fam["contrast_lo"],
                     "contrast_hi": fam["contrast_hi"],
                     "flip_frac": fam["flip_frac"]})
        print(f"  {label:<16} flips {flips}/{n}  contrast "
              f"{fam['contrast_mean']:+.2f} [{fam['contrast_lo']:+.2f}, "
              f"{fam['contrast_hi']:+.2f}]")

    print(f"[{tag} / {args.domain}] held-out robustness")
    run("original")
    for p in range(args.partitions):
        run(f"partition {p}", split_seed=200 + p)
    if not args.no_loo:
        for o in operands:
            run(f"LOO {o}", build=[x for x in operands if x != o], test=[o])

    df = pd.DataFrame(rows)
    parts = df[df["label"].str.startswith(("original", "partition"))]
    loo = df[df["label"].str.startswith("LOO")]
    summary = {
        "partitions": {
            "n": int(len(parts)),
            "contrast_mean": st.mean(parts["contrast"]),
            "contrast_min": float(parts["contrast"].min()),
            "contrast_max": float(parts["contrast"].max()),
            "all_flip_all": bool((parts["flips"] == parts["n_pairs"]).all()),
        },
    }
    if len(loo):
        summary["loo"] = {
            "n": int(len(loo)),
            "contrast_mean": st.mean(loo["contrast"]),
            "contrast_min": float(loo["contrast"].min()),
            "contrast_max": float(loo["contrast"].max()),
            "total_flips": int(loo["flips"].sum()),
            "total_pairs": int(loo["n_pairs"].sum()),
        }
    print(f"\npartitions: mean {summary['partitions']['contrast_mean']:+.2f} "
          f"range [{summary['partitions']['contrast_min']:+.2f}, "
          f"{summary['partitions']['contrast_max']:+.2f}], "
          f"all pairs flip in every partition: {summary['partitions']['all_flip_all']}")
    if len(loo):
        print(f"LOO: mean {summary['loo']['contrast_mean']:+.2f} range "
              f"[{summary['loo']['contrast_min']:+.2f}, "
              f"{summary['loo']['contrast_max']:+.2f}], flips "
              f"{summary['loo']['total_flips']}/{summary['loo']['total_pairs']}")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_heldout_parts.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    summary["_meta"] = {"tag": tag, "domain": args.domain,
                        "note": "partitions = shuffled 6/6 splits (seeds 200+p); "
                                "LOO = build on 11 operands, test the swap on the "
                                "single held-out operand"}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"saved {out} (+ .json)")


if __name__ == "__main__":
    main()
