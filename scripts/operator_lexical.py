#!/usr/bin/env python
"""Cross-LEXICALIZATION transfer: is the operator direction a property of the
RELATION or of its WORDING? Unlike operator_templates.py (which varies the frame
around a fixed '{op} of {a}' unit), this test replaces the unit itself:

    A (relations.json):      "The currency of France is"
    B (relations_lex.json):  "The money used in France is"

Build v(op) on formulation A, run the all-pairs swap on formulation B (and the
reverse, plus each formulation's within-set baseline). Transfer across lexical
paraphrases rules out the surface-wording confound entirely.

    python scripts/operator_lexical.py 1.7b
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
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    domA = op_core.load_domain("relations")
    domB = op_core.load_domain("relations_lex")
    assert domA.op_keys == domB.op_keys and domA.operand_keys == domB.operand_keys

    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]

    for name, dom in (("A (of-phrasing)", domA), ("B (lexical)", domB)):
        rep = op_core.guard_tokenization(dom, tok)
        print(f"[{name}] single-token frac {rep['single_token_frac']:.2f}, "
              f"min pair signal {rep['min_pair_signal']}")
        print(f"  sample: {dom.render(dom.operand_keys[0], dom.op_keys[0])!r}")

    dirs = {"A": op_core.op_dirs(model, ws, domA),
            "B": op_core.op_dirs(model, ws, domB)}
    doms = {"A": domA, "B": domB}

    print(f"\n[{tag}] cross-lexicalization swaps  (build -> test)")
    rows, longs = [], []
    for build in ("A", "B"):
        for test in ("A", "B"):
            ldf = op_core.measure_swaps_long(model, ws, doms[test], tok,
                                             seed=args.seed, dirs=dirs[build])
            ldf["build_set"], ldf["test_set"] = build, test
            longs.append(ldf)
            wide = op_core.measure_swaps(model, ws, doms[test], tok, long_df=ldf)
            fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
            valid = wide.dropna(subset=["swap"])
            rows.append({"build": build, "test": test, "transfer": build != test,
                         "flips": int((valid["swap"] > 0).sum()), "pairs": len(valid),
                         "clean": float(valid["clean"].mean()),
                         "contrast": fam["contrast_mean"],
                         "lo": fam["contrast_lo"], "hi": fam["contrast_hi"],
                         "flip_frac": fam["flip_frac"],
                         "flip_lo": fam["flip_lo"], "flip_hi": fam["flip_hi"]})
            r = rows[-1]
            mark = "  (lexical transfer)" if build != test else ""
            print(f"  {build} -> {test}{mark}  flips {r['flips']}/{r['pairs']}  "
                  f"clean {r['clean']:+.2f}  contrast {r['contrast']:+.2f} "
                  f"[{r['lo']:+.2f}, {r['hi']:+.2f}]")

    sdf = pd.DataFrame(rows)
    out = ROOT / "results" / "ablation" / f"{tag}_relations_lexical.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    sdf.to_parquet(out)
    pd.concat(longs, ignore_index=True).to_parquet(
        out.with_name(out.name.replace(".parquet", "_long.parquet")))
    print(f"\nsaved {out} (+ _long)")

    off = sdf[sdf["transfer"]]
    print(f"lexical transfer total: {off['flips'].sum()}/{off['pairs'].sum()} flips, "
          f"mean contrast {off['contrast'].mean():+.2f}")


if __name__ == "__main__":
    main()
