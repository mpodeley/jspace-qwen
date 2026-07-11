#!/usr/bin/env python
"""Cross-template transfer: is the operator direction a property of the RELATION or
of the PROMPT FRAME? Build v(op) on paraphrase frame i, run the all-pairs swap on
frame j. Diagonal combos (i==j) measure paraphrase robustness within each frame
(0,0 reproduces the legacy single-template result); off-diagonal combos are the
transfer test that rules out template echo.

    python scripts/operator_templates.py 1.7b
    python scripts/operator_templates.py 8b --combos 0:0 0:1 1:0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--combos", nargs="*", default=None,
                    help="build:test template-index pairs, e.g. 0:1 1:0 (default: all)")
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    n_tpl = len(dom.templates)
    combos = ([tuple(map(int, c.split(":"))) for c in args.combos] if args.combos
              else [(i, j) for i in range(n_tpl) for j in range(n_tpl)])

    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]

    print(f"[{tag} / {args.domain}] {n_tpl} paraphrase frames:")
    for t, tpl in enumerate(dom.templates):
        print(f"  T{t}: {tpl!r}")

    # cache operator directions per build frame; reuse across test frames
    dirs = {}
    summary, longs = [], []
    for i, j in combos:
        if i not in dirs:
            dirs[i] = op_core.op_dirs(model, ws, dom, templates=[i])
        ldf = op_core.measure_swaps_long(model, ws, dom, tok, seed=args.seed,
                                         templates=[j], dirs=dirs[i])
        ldf["build_tpl"], ldf["test_tpl"] = i, j
        longs.append(ldf)
        wide = op_core.measure_swaps(model, ws, dom, tok, long_df=ldf)
        fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
        valid = wide.dropna(subset=["swap"])
        summary.append({
            "build_tpl": i, "test_tpl": j, "transfer": i != j,
            "flips": int((valid["swap"] > 0).sum()), "pairs": len(valid),
            "contrast": fam["contrast_mean"],
            "lo": fam["contrast_lo"], "hi": fam["contrast_hi"],
            "flip_frac": fam["flip_frac"],
            "flip_lo": fam["flip_lo"], "flip_hi": fam["flip_hi"]})
        s = summary[-1]
        print(f"  build T{i} -> test T{j}{'  (transfer)' if i != j else '           '}"
              f"  flips {s['flips']}/{s['pairs']}  contrast "
              f"{s['contrast']:+.2f} [{s['lo']:+.2f}, {s['hi']:+.2f}]")

    sdf = pd.DataFrame(summary)
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_templates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_parquet(out)
    pd.concat(longs, ignore_index=True).to_parquet(
        out.with_name(out.name.replace(".parquet", "_long.parquet")))
    print(f"\nsaved {out} (+ _long)")

    off = sdf[sdf["transfer"]]
    if len(off):
        print(f"cross-frame transfer: {off['flips'].sum()}/{off['pairs'].sum()} flips, "
              f"mean contrast {off['contrast'].mean():+.2f}")


if __name__ == "__main__":
    main()
