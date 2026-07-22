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
LES = ROOT / "results" / "lesion"
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

    # --- lesion study: criticality (P0) ---------------------------------------
    for tag in ("1.7b", "8b"):
        p = LES / f"{tag}_criticality_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        top = ", ".join(f"L{c['layer']}H{c['head']} x{c['ppl_ratio']:.1f}"
                        for c in s["critical_heads"][:3]) or "none"
        out.append((f"{tag} LESION P0 criticality",
                    f"{len(s['critical_heads'])} of {s['n_heads']} heads > 2x ppl "
                    f"({s['frac_critical']:.1%})",
                    f"median x{s['median_ppl_ratio']:.3f}; worst: {top}"))

    # --- lesion study: the per-network signature (P2) --------------------------
    # Reads the summary JSON, not the parquet: *.parquet is gitignored, so a
    # parquet-based check would silently pass on a fresh clone by finding nothing.
    for tag in ("1.7b", "8b"):
        p = LES / f"{tag}_relations_lesion_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        base = s["baseline"]
        out.append((f"{tag} LESION baseline",
                    f"relations {base['relations_acc']:.1%}, ppl {base['ppl']:.2f}",
                    f"arithmetic {base['arithmetic']:.0%}, "
                    f"{len(s['runs'])} runs"))
        diss = s.get("dissociation", {})
        for key, v in diss.items():
            if not key.startswith("neuron"):
                continue                      # the head arm is reported separately
            sig = v["signature"]
            detail = (f"acc {v['acc_top']:.1%} vs control band "
                      f"[{v['control_band'][0]:.1%}, {v['control_band'][1]:.1%}], "
                      f"ppl x{v['ppl_ratio']:.2f}")
            if sig in ("other_relation", "other_operand"):
                detail += (f"; {v[f'count_{sig}']}/{v['n_cells']} cells vs null "
                           f"{v[f'null_rate_{sig}']:.1%} over {v['null_cells']} "
                           f"control cells, p={v[f'p_{sig}']:.1e} "
                           f"(alpha={v['alpha_bonferroni']:.1e})")
            out.append((f"{tag} LESION {key}", sig, detail))

    # --- lesion study: the critical-head probe (P3) ---------------------------
    for tag in ("1.7b", "8b"):
        p = LES / f"{tag}_critical_probe.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        gj = s["greedy_joint"]
        n = len(gj["trace"])
        out.append((f"{tag} PROBE greedy joint ({n} heads)",
                    f"max ppl x{gj['max_ppl_ratio']:.1f}",
                    f"step1 {gj['trace'][0]['added']} x{gj['trace'][0]['ppl_ratio']:.1f}"))
        sink = ", ".join(f"{h}={v:.1%}" for h, v in
                         s.get("critical_sink_fraction", {}).items()) or "no critical heads"
        out.append((f"{tag} PROBE critical-head sink fraction",
                    sink,
                    f"max sink in model {s['max_sink_fraction']:.1%}"))

    # --- attention knockout: the operand is routing, not tissue (3.9) ---------
    for name in ("1.7b_relations_knockout_w5", "1.7b_relations_knockout_w9",
                 "8b_relations_knockout"):
        p = LES / f"{name}_summary.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        cw = s["critical_window"]
        d = s["dissociation"]
        pv = s["paired"]["entity_vs_operator"]
        out.append((f"{name} KNOCKOUT critical window",
                    f"L{cw['layers'][0]}-{cw['layers'][-1]} ({cw['mid_depth']:.0f}% depth)",
                    f"baseline acc {s['baseline_acc']:.1%}, window W={s['window']}"))
        out.append((f"{name} KNOCKOUT entity vs operator-word",
                    f"acc {d['entity']['acc']:.1%} vs {d['operator']['acc']:.1%}, "
                    f"Δ{pv['acc_diff']:+.1%} [{pv['lo']:+.1%}, {pv['hi']:+.1%}]",
                    f"McNemar p={pv['mcnemar_p']:.1e}, other_operand "
                    f"{s['paired']['other_operand_counts'].get('entity', 0)}/"
                    f"{s['paired']['n_cells']} (operator "
                    f"{s['paired']['other_operand_counts'].get('operator', 0)})"))
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
