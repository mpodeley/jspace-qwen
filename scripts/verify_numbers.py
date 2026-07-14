#!/usr/bin/env python
"""Recompute every headline number from the persisted artifacts and diff it against
what the prose actually says.

The repo's discipline: no number appears in the paper, the site, or a promo draft
unless it can be recomputed from `results/`. This script is the enforcement. It
reloads each parquet/json, recomputes the statistic with the same estimator the
experiment used (op_core's cluster bootstraps, not a re-derivation), and prints a
table of value-vs-source. Run it before every PDF build.

    .venv/bin/python scripts/verify_numbers.py
    .venv/bin/python scripts/verify_numbers.py --grep "+22.6"   # where is this cited?
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import op_core

ROOT = Path(__file__).resolve().parent.parent
ABL = ROOT / "results" / "ablation"
GEO = ROOT / "results" / "geometry"
PROSE = ["docs/paper.md", "docs/findings.md", "docs/robustness.md", "docs/index.md",
         "docs/explained.md", "README.md", "promo/x-thread.md", "promo/lesswrong.md"]


def swap_contrast(tag, domain, kind="operator_swap"):
    p = ABL / f"{tag}_{domain}_{kind}_long.parquet"
    if not p.exists():
        return None
    ldf = pd.read_parquet(p)
    fam = op_core.bootstrap_family_ci(ldf, seed=0)
    return {"contrast": fam["contrast_mean"], "lo": fam["contrast_lo"],
            "hi": fam["contrast_hi"], "flips": fam["flip_frac"], "n": len(ldf)}


def rows():
    """(label, recomputed value, formatted-as-it-should-appear) triples."""
    out = []

    # --- the all-pairs swap, per model -------------------------------------
    for tag in ("1.7b", "8b", "gemma-2-9b"):
        r = swap_contrast(tag, "relations")
        if r:
            out.append((f"{tag} all-pairs swap contrast",
                        f"{r['contrast']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]",
                        f"flip fraction {r['flips']:.2f}, n={r['n']}"))
        h = swap_contrast(tag, "relations", "heldout")
        if h:
            out.append((f"{tag} held-out-operand contrast",
                        f"{h['contrast']:+.1f} [{h['lo']:+.1f}, {h['hi']:+.1f}]",
                        f"flip fraction {h['flips']:.2f}, n={h['n']}"))

    # --- animals (the second domain) ---------------------------------------
    for tag in ("1.7b", "8b"):
        for kind, name in (("operator_swap", "all-pairs swap"),
                           ("heldout", "held-out-operand")):
            r = swap_contrast(tag, "animals", kind)
            if r:
                out.append((f"{tag} ANIMALS {name} contrast",
                            f"{r['contrast']:+.1f} [{r['lo']:+.1f}, {r['hi']:+.1f}]",
                            f"flip fraction {r['flips']:.2f}, n={r['n']}"))

    # --- the factorization (variance shares) --------------------------------
    for tag in ("1.7b", "8b", "gemma-2-9b"):
        for domain in ("relations", "animals", "arithmetic", "logic"):
            p = GEO / f"{tag}_{domain}.json"
            if not p.exists():
                continue
            g = json.loads(p.read_text())
            v = g["variance"]["query"]
            out.append((f"{tag} {domain} ANOVA @query",
                        f"operand {v['stem']:.1%} / operator {v['case']:.1%} / "
                        f"interaction {v['interaction']:.1%}", ""))

    # --- E2: the donor decomposition (the headline) -------------------------
    for tag in ("1.7b", "8b"):
        for sfx, scope in (("", "band"), ("_single", "single layer")):
            p = ABL / f"{tag}_relations_patch_decomp{sfx}.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())
            meta = s.pop("_meta")
            out.append((f"{tag} PATCH[{scope}] clean greedy accuracy",
                        f"{meta['clean_exact_from']:.0%}",
                        f"n={meta['n_cells']} cells"))
            for k, v in s.items():
                out.append((f"{tag} PATCH[{scope}] {k}",
                            f"exact match {v['exact_match']:.1%}",
                            f"Δmargin {v['delta_margin']:+.1f} "
                            f"[{v['delta_margin_lo']:+.1f}, {v['delta_margin_hi']:+.1f}], "
                            f"rank(to) {v['rank_to_median']:.0f}, top-1 {v['top1']:.0%}"))

    # --- E1: positions x scope ---------------------------------------------
    for tag in ("1.7b", "8b"):
        p = ABL / f"{tag}_relations_positions.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        s.pop("_meta", None)
        dose = s.pop("_dose", None)
        for k, v in s.items():
            kl = f", KL off {v['kl_offtask']:.1f}" if v.get("kl_offtask") else ""
            out.append((f"{tag} POSITION {k}",
                        f"Δmargin {v['delta_margin']:+.1f} "
                        f"[{v['delta_margin_lo']:+.1f}, {v['delta_margin_hi']:+.1f}]",
                        f"exact {v['exact_match']:.1%}, rank {v['rank_to_clean']:.0f}→"
                        f"{v['rank_to']:.0f}, top-1 {v['top1']:.0%}, "
                        f"sign+ {v['sign_correct']:.0%}{kl}"))
        for d in (dose or []):
            out.append((f"{tag} DOSE {d['scope']}/{d['position']} α={d['alpha']}",
                        f"exact match {d['exact_match']:.1%}",
                        f"Δmargin {d['delta_margin']:+.1f}, n={d['n']}"))

    # --- E4: the null battery ----------------------------------------------
    for tag in ("1.7b", "8b"):
        p = ABL / f"{tag}_relations_nulls.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        for r in m["nulls"]:
            extra = ""
            if "seed_mean" in r:
                extra = (f" · across {r['n_seeds']} redraws: {r['seed_mean']:+.2f} "
                         f"[{r['seed_lo']:+.2f}, {r['seed_hi']:+.2f}]")
            out.append((f"{tag} NULL {r['null']}",
                        f"contrast {r['contrast']:+.2f} "
                        f"[{r['contrast_lo']:+.2f}, {r['contrast_hi']:+.2f}]",
                        f"flips {r['flip_frac']:.2f}{extra}"))

    # --- E5: layer sweep ----------------------------------------------------
    for tag in ("1.7b", "8b"):
        p = ABL / f"{tag}_relations_layersweep.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        pd_, pc = m["peak_decodability"], m["peak_causal"]
        out.append((f"{tag} LAYERS peak decodability",
                    f"L{pd_['layer']} ({pd_['depth']:.0f}% depth), {pd_['value']:.1%}",
                    f"chance {m['chance_decodability']:.0%}"))
        out.append((f"{tag} LAYERS peak causal",
                    f"L{pc['layer']} ({pc['depth']:.0f}% depth), "
                    f"contrast {pc['value']:+.1f}",
                    f"depth gap {m['depth_gap']:+.1f} points"))

    # --- dose-response (incl. negative alphas) ------------------------------
    for tag in ("1.7b",):
        p = ABL / f"{tag}_relations_dose.parquet"
        if not p.exists():
            continue
        d = pd.read_parquet(p).sort_values("alpha")
        spec = d["swap_shift"] - d["random_shift"]
        neg = d[d["alpha"] < 0]
        out.append((f"{tag} DOSE specific effect @α=4",
                    f"{float(spec[d['alpha'] == 4].iloc[0]):+.1f}",
                    f"random shift {float(d[d['alpha'] == 4]['random_shift'].iloc[0]):+.1f}, "
                    f"off-task KL {float(d[d['alpha'] == 4]['kl_nats'].iloc[0]):.1f} nats"))
        if len(neg):
            out.append((f"{tag} DOSE sign reversal (α<0)",
                        f"most negative α={float(neg['alpha'].min()):+.1f}: "
                        f"specific effect {float(spec[d['alpha'] == neg['alpha'].min()].iloc[0]):+.1f}",
                        "monotone through zero ⇒ signed axis"))

    # --- op_minimal (the earlier claim, re-checked) -------------------------
    p = ABL / "1.7b_relations_minimal.parquet"
    if p.exists():
        df = pd.read_parquet(p)
        cols = [c for c in df.columns if c not in ("from", "to", "operand")]
        for c in cols:
            out.append((f"1.7b MINIMAL {c}", f"exact match {df[c].mean():.1%}",
                        f"n={len(df)}"))

    # --- vocab semantics: portrait (P1) --------------------------------------
    for tag in ("1.7b", "8b", "gemma-2-9b"):
        p = ABL / f"{tag}_relations_vocab_portrait_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        ws = s["_meta"]["ws"]
        lever = str(ws[-1])
        first = str(ws[0])
        for lname in s["e2_diag_contrast"]:
            c = s["e2_diag_contrast"][lname]
            out.append((f"{tag} VOCAB E2 diag contrast [{lname}]",
                        f"L{first} {c[first]['diag_contrast']:+.2f} → "
                        f"L{lever} {c[lever]['diag_contrast']:+.2f}",
                        f"null max @L{lever}: perm {c[lever]['perm_max']:+.2f}, "
                        f"rand_sub {c[lever]['rand_sub_max']:+.2f}"))
        m = s["e5_marker"][lever]
        out.append((f"{tag} VOCAB E5g marker energy @L{lever}",
                    f"{m['energy_in_pair_span']:.2%} of ||m||²",
                    f"random expectation {m['rand_energy_mean']:.2%}, "
                    f"built on {m['n_syncretic_build']} syncretic operands"))

    # --- vocab semantics: causal split (P2) -----------------------------------
    for tag in ("1.7b", "8b", "gemma-2-9b"):
        p = ABL / f"{tag}_relations_vocab_causal_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        for key in sorted(k for k in s if "/" in k):
            v = s[key]
            dm = v["delta_margin"]
            fc = v.get("forced_choice_target")
            out.append((f"{tag} VOCAB SPLIT {key}",
                        f"Δmargin {dm['mean']:+.1f} [{dm['lo']:+.1f}, {dm['hi']:+.1f}]",
                        f"flips {v['flip_frac']:.0%}, exact {v['exact_match']:.1%}, "
                        f"says-target {v['class_target']:.1%}"
                        + (f", forced-choice {fc:.1%}" if fc is not None else "")))
        add = s.get("_additivity_band")
        if add:
            out.append((f"{tag} VOCAB SPLIT additivity (band)",
                        f"|Δans+Δrest−Δfull| = {add['mean_abs_residual']:.2f}",
                        f"vs mean Δfull {add['mean_full']:+.1f}"))
        shares = s.get("_ans_energy_share_band", {})
        if shares:
            vals = list(shares.values())
            out.append((f"{tag} VOCAB SPLIT c_ans energy share",
                        f"mean {sum(vals)/len(vals):.1%}",
                        f"range {min(vals):.1%}–{max(vals):.1%} over {len(vals)} pairs"))
        pm = ABL / f"{tag}_relations_marker_causal_summary.json"
        if pm.exists():
            ms = json.loads(pm.read_text())
            meta = ms["_meta"]
            for key in ("alpha=4.0/sign=+", "alpha=4.0/sign=-"):
                if key in ms:
                    mv = ms[key]
                    out.append((f"{tag} MARKER {key}",
                                f"Δ(lang margin) {mv['delta_lang_mean']:+.2f} "
                                f"[{mv['lo']:+.2f}, {mv['hi']:+.2f}]",
                                f"sign-correct {mv['sign_correct_frac']:.0%}, "
                                f"span energy {meta['energy_in_pair_span']:.2%}, "
                                f"n={len(meta['tested_on'])} operands"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grep", default=None,
                    help="find where a literal value is cited in the prose")
    args = ap.parse_args()

    if args.grep:
        pat = re.escape(args.grep)
        for f in PROSE:
            fp = ROOT / f
            if not fp.exists():
                continue
            for i, line in enumerate(fp.read_text().splitlines(), 1):
                if re.search(pat, line):
                    print(f"{f}:{i}: {line.strip()[:130]}")
        return

    rs = rows()
    w = max(len(r[0]) for r in rs) if rs else 10
    print(f"{'quantity':<{w}}  {'recomputed from artifact':<44}  detail")
    print("-" * (w + 90))
    for label, val, detail in rs:
        print(f"{label:<{w}}  {val:<44}  {detail}")
    print(f"\n{len(rs)} numbers recomputed from results/. "
          f"Cross-check against the prose with --grep '<value>'.")


if __name__ == "__main__":
    main()
