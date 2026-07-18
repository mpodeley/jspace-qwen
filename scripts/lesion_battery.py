#!/usr/bin/env python
"""The lesion study: remove a functional network, measure whether the deficit is
SELECTIVE.

The neuropsychological gold standard is a DOUBLE DISSOCIATION: lesion A destroys
task X but spares Y, lesion B destroys Y but spares X. Anything less is
compatible with "we just damaged the model". This script runs that design on the
networks found by lesion_localize.py.

The readout was already written for a different purpose and happens to be exactly
right: op_audit.classify sorts a generation into
    target / source / OTHER_OPERAND (right relation, wrong entity)
                    / OTHER_RELATION (right entity, wrong relation) / degraded
which is precisely the structure of a dissociation.

Pre-registered predictions
    lesion the OPERAND network  -> other_operand rises (it says another country's
                                   capital: the relation survives, the entity is
                                   lost)
    lesion the OPERATOR network -> other_relation rises (it says something about
                                   the right country under the wrong relation)
    control lesions             -> no class shift, or unstructured degradation
    every lesion                -> WikiText perplexity spared; otherwise the
                                   deficit is damage, not localization

Outcome (1.7B and 8B, 2026-07-14/15). The operator prediction holds at both
scales. The operand prediction fails at both, at BOTH read positions: at the
query position the lesion is null (inside the control band), and at the entity
token -- where b65dbc7 predicted it would not be redundantly coded -- the deficit
is real but degrades (ppl x1.48 at 1.7B) or carries the OPERATOR signature (8B).
The dissociation is one-sided. The last prediction above is the one that settles
it, which is why scoring now GATES on it rather than merely reporting it.

Controls (CONTROLS below), in increasing order of what they rule out:
    random          k units drawn uniformly. Globally random, so NOT depth-matched
                    -- kept only to expose the depth confound (see `pick`). At
                    1.7B k=16/48 it catches the attention sink and hits ppl x38 /
                    x78: that row is the evidence FOR depth-matching, not a bug.
    layer           depth-matched: same layers, same count, non-selective.
    magnitude       depth- AND activation-magnitude-matched: "the same amount of
                    tissue, in the wrong place". The control that matters.

A null deficit is ambiguous: the IOI circuit's backup heads show that networks
compensate. So we sweep the lesion size and score each (network, kind, k) against
its OWN controls -- control-corrected class deltas and a control band on accuracy
-- never as an absolute deficit. `score_dissociation` reads the summary JSON, not
the parquet, so it re-scores from a fresh clone with no GPU:

    python scripts/lesion_battery.py 1.7b --stage anchor    # P1: induction gate
    python scripts/lesion_battery.py 1.7b --stage dissoc    # P2: the centerpiece
    python scripts/lesion_battery.py 1.7b --rescore         # re-score, no model
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import get_corpus, load_model, resolve_tag
from lesion_core import Lesion, perplexity, reference_means, unit_dims
from lesion_localize import induction_score, induction_stimuli
import op_core
import op_minimal
from op_audit import classify

ROOT = Path(__file__).resolve().parent.parent


def pick(df, kind, network, k, mode="top", seed=0):
    """The units of a network, and its matched control lesions.

    `top`        the k most selective units.
    `layer`      DEPTH-MATCHED random control: the same NUMBER of units drawn
                 from the SAME layers as the top-k, but non-selective. This is
                 the control that matters. A globally-random control is
                 confounded by depth -- a single attention-sink head (L1H5 in
                 Qwen3-1.7B) raises perplexity 17x on its own, so a random draw
                 that happens to include it "beats" any targeted lesion for
                 reasons that have nothing to do with function.
    `magnitude`  depth-matched AND activation-magnitude-matched: same layers,
                 same amount of "tissue activity", different selectivity --
                 "the same amount of tissue, in the wrong place".
    `random`     globally random (kept only to expose the depth confound above).
    """
    sub = df[df.kind == kind].reset_index(drop=True)
    top = sub.nlargest(k, network)
    if mode == "top":
        sel = top
    elif mode == "random":
        sel = sub.sample(k, random_state=seed)
    elif mode in ("layer", "magnitude"):
        want = top.groupby("layer").size()            # the depth histogram to match
        chosen, rng = [], np.random.default_rng(seed)
        for l, n in want.items():
            pool = sub[(sub.layer == l) & (~sub.unit_id.isin(top.unit_id))]
            if not len(pool):
                continue
            if mode == "layer":
                take = pool.sample(min(n, len(pool)), random_state=seed)
            else:
                # nearest-magnitude neighbours of the lesioned units in this layer
                tgt = np.sort(top[top.layer == l]["magnitude"].to_numpy())
                pool = pool.sort_values("magnitude").reset_index(drop=True)
                idx = np.searchsorted(pool["magnitude"].to_numpy(), tgt).clip(
                    0, len(pool) - 1)
                take = pool.iloc[np.unique(idx)]
                if len(take) < n:                     # top up after collisions
                    rest = pool.drop(take.index)
                    take = pd.concat([take, rest.sample(
                        min(n - len(take), len(rest)), random_state=seed)])
            chosen.append(take.head(n))
        sel = pd.concat(chosen) if chosen else top.head(0)
    else:
        raise ValueError(mode)
    return [(int(r.layer), int(r.index)) for r in sel.itertuples()], set(sel.unit_id)


def relations_readout(model, dom, tok, ops, n_pairs=None, k=8):
    """Clean-prompt generations under whatever lesion is active: for every
    (operand, relation) cell, decode k tokens and classify. The prompt asks the
    TRUE question -- we are not steering, we removed tissue and are watching what
    breaks."""
    rows = []
    for o in dom.operand_keys:
        for rel in ops:
            p = dom.render(o, rel)
            target = str(dom.answer(o, rel))
            other_op = [str(dom.answer(oo, rel)) for oo in dom.operand_keys
                        if oo != o and str(dom.answer(oo, rel)).lower()
                        != target.lower()]
            other_rel = [str(dom.answer(o, kk)) for kk in ops if kk != rel
                         and str(dom.answer(o, kk)).lower() != target.lower()]
            text = op_minimal.greedy(model, p, k=k)
            # classify with source=target: there is no "source" condition in a
            # lesion (we never switch the question), so the class collapses to
            # target / other_operand / other_relation / degraded
            cls = classify(text, target, target, other_op, other_rel)
            rows.append({"operand": o, "relation": rel, "text": text,
                         "class": cls, "correct": cls == "target"})
    return pd.DataFrame(rows)


def arithmetic_accuracy(model, k=4):
    dom = op_core.load_domain("arithmetic")
    hits = 0
    n = 0
    for o in dom.operand_keys:
        for kk in dom.op_keys:
            text = op_minimal.greedy(model, dom.render(o, kk), k=k)
            hits += op_minimal.hit(text, str(dom.answer(o, kk)))
            n += 1
    return hits / max(n, 1)


# The gate the docstring pre-registers: a lesion that costs perplexity or
# arithmetic is damage, and its class shift means nothing. PPL_GATE is the
# criticality screen's own threshold (`ppl_ratio > 2.0` below) -- one definition
# of "catastrophic" for the whole study, not a second one invented here.
PPL_GATE = 2.0
ARITH_GATE = 0.8          # fraction of baseline arithmetic that must survive
N_CELLS = 60              # 12 operands x 5 relations: class proportions quantize at 1/60


def wilson(p, n, z=1.96):
    """95% Wilson interval for a proportion of n cells.

    The classes are counts out of 60, so a "3.3% shift" is two cells. Printing the
    interval next to every proportion is what stops prose from leaning on one.
    """
    if n <= 0:
        return [0.0, 0.0]
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, centre - half), min(1.0, centre + half)]


def annotate_critical_heads(tag, domain, runs, seed=0):
    """Record which head lesions swallowed a head the criticality screen flagged.

    Needs the localizer and criticality parquets, both gitignored, so this is a
    no-op on a fresh clone -- and that is exactly why the annotation is WRITTEN
    into the tracked summary: the reader who cannot recompute it still gets it.
    """
    loc = ROOT / "results" / "lesion" / f"{tag}_{domain}_localizer.parquet"
    crit = ROOT / "results" / "lesion" / f"{tag}_criticality_summary.json"
    if not (loc.exists() and crit.exists()):
        return
    critical = {(c["layer"], c["head"])
                for c in json.loads(crit.read_text())["critical_heads"]}
    if not critical:
        return
    df = pd.read_parquet(loc)
    for r in runs.values():
        if r["kind"] != "head":
            continue
        units, _ = pick(df, "head", r["network"], r["k"], mode=r["control"],
                        seed=seed)
        hit = sorted(critical.intersection(units))
        r["contains_critical_head"] = [f"L{l}H{h}" for l, h in hit]


STRUCTURED = ("other_relation", "other_operand")   # the pre-registered signatures
CLASSES = STRUCTURED + ("degraded",)               # degraded is the sink, not a signature


def _log_comb(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def fisher_greater(k_top, n_top, k_null, n_null):
    """One-sided Fisher exact: is k_top/n_top enriched over k_null/n_null?

    Computed in log space. The naive ratio of binomials overflows float64 here --
    the pooled null is ~2160 cells and `degraded` fills ~800 of them, and
    C(2220, 800) is about 10^600.
    """
    total, hits = n_top + n_null, k_top + k_null
    hi = min(n_top, hits)
    if k_top > hi or hits == 0:
        return 1.0
    denom = _log_comb(total, hits)
    terms = [_log_comb(n_top, i) + _log_comb(n_null, hits - i) - denom
             for i in range(k_top, hi + 1)]
    m = max(terms)
    return min(1.0, math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


def pooled_null(runs, kind, n_cells=N_CELLS, ppl_gate=PPL_GATE):
    """What a NON-selective lesion of this kind does to each class: the empirical
    null, pooled over every control run in the battery, as CELL COUNTS.

    Worth the pooling. A per-cell band is 3 numbers; this is 36 control runs
    (neurons, 1.7B) = 2160 cells, of which 24 are other_relation. So the operator
    lesion's 10-of-60 is tested against a rate of 1.1%, not against a hand-waved
    "the controls do nothing".

    Counts, not the observed maximum: a ceiling of 0.0% would make ANY single cell
    look like an infinite enrichment, which is how a 1-of-60 shift got called a
    signature in the first draft of this function.

    Pooling within `kind` and not across it matters -- the 1.7B head controls
    reach 11.7% other_relation, so the head arm is held to a null seven times
    higher than the neuron arm, which is the scepticism the head rows have earned.

    Controls that are themselves damage (ppl >= gate) are excluded: they would
    poison the null with the very confound the gate exists to catch.
    """
    vals = {c: [] for c in CLASSES}
    for r in runs.values():
        if (r["kind"] != kind or r["control"] == "top"
                or r["ppl_ratio"] >= ppl_gate):
            continue
        for c in CLASSES:
            vals[c].append(r[f"class_{c}"])
    out = {}
    for c, v in vals.items():
        out[c] = {"n_runs": len(v),
                  "n_cells": len(v) * n_cells,
                  "hits": int(round(sum(v) * n_cells)),
                  "rate": float(np.mean(v)) if v else 0.0,
                  "ceiling": float(max(v)) if v else 0.0}
    return out


def score_dissociation(runs, baseline, n_cells=N_CELLS, alpha=0.05):
    """The lesion signature of every (kind, k, network).

    Supersedes the `crossed` boolean, which was retired because it (a) was true on
    a 1-cell margin at k=32 and false at k=512, the config the headline rests on,
    (b) never consulted the controls, and (c) hard-coded a two-network design, so
    the operand_entity runs it was meant to adjudicate were silently dropped.

    Two tests, each doing one job. IS there a deficit: accuracy below the band of
    this cell's own depth-matched controls. Is the deficit STRUCTURED: a one-sided
    Fisher exact of the class's cell count against the pooled control null for its
    kind, Bonferroni-corrected across every test this function runs -- 42 of them
    at 21 configurations x 2 classes, which is precisely why an uncorrected p=0.03
    must not be allowed to name a network.

    `degraded` can never be a signature. The docstring pre-registers unstructured
    degradation as a CONTROL outcome, and the controls duly sit at ~37% degraded
    (1.7B): it is the sink every failure falls into, so it wins any argmax and
    means nothing when it does.

    Reads the summary's `runs` block -- a complete copy of the parquet, which is
    gitignored -- so it re-scores from a fresh clone with no model and no GPU.
    """
    base_arith = baseline["arithmetic"]
    by = {}
    for r in runs.values():
        by.setdefault((r["kind"], r["k"], r["network"]), {})[r["control"]] = r
    nulls = {kind: pooled_null(runs, kind, n_cells) for kind in {k[0] for k in by}}

    cells = []
    for (kind, k, network), arms in sorted(by.items()):
        top = arms.get("top")
        ctrls = [arms[c] for c in ("layer", "magnitude", "random") if c in arms]
        if top is None or not ctrls:
            continue
        cells.append(((kind, k, network), top, ctrls, nulls[kind]))

    bonferroni = alpha / max(len(cells) * len(STRUCTURED), 1)

    out = {}
    for (kind, k, network), top, ctrls, null in cells:
        band = [min(c["relations_acc"] for c in ctrls),
                max(c["relations_acc"] for c in ctrls)]
        outside = top["relations_acc"] < band[0]
        interpretable = (top["ppl_ratio"] < PPL_GATE
                         and top["arithmetic"] >= ARITH_GATE * base_arith)
        counts = {c: int(round(top[f"class_{c}"] * n_cells)) for c in CLASSES}
        pvals = {c: fisher_greater(counts[c], n_cells, null[c]["hits"],
                                   null[c]["n_cells"]) for c in STRUCTURED}
        sig = {c: pvals[c] < bonferroni for c in STRUCTURED}

        if not interpretable:
            signature = "uninterpretable"        # damage: the class shift means nothing
        elif not outside:
            signature = "null"                   # no deficit vs matched controls
        else:
            hits = [c for c in STRUCTURED if sig[c]]
            signature = (min(hits, key=lambda c: pvals[c]) if hits
                         else "unstructured")    # a real deficit with no signature

        entry = {"signature": signature,
                 "interpretable": bool(interpretable),
                 "outside_control_band": bool(outside),
                 "acc_top": top["relations_acc"],
                 "control_band": band,
                 "ppl_ratio": top["ppl_ratio"],
                 "arithmetic": top["arithmetic"],
                 "n_cells": n_cells,
                 "alpha_bonferroni": bonferroni}
        for c in CLASSES:
            entry[f"class_{c}"] = top[f"class_{c}"]
            entry[f"count_{c}"] = counts[c]
            entry[f"ci_{c}"] = wilson(top[f"class_{c}"], n_cells)
            entry[f"null_rate_{c}"] = null[c]["rate"]
            entry[f"null_ceiling_{c}"] = null[c]["ceiling"]
        for c in STRUCTURED:
            entry[f"p_{c}"] = pvals[c]
            entry[f"significant_{c}"] = bool(sig[c])
        entry["null_runs"] = null["degraded"]["n_runs"]
        entry["null_cells"] = null["degraded"]["n_cells"]
        if top.get("contains_critical_head"):
            entry["contains_critical_head"] = top["contains_critical_head"]
        out[f"{kind}/k={k}/{network}"] = entry
    return out


def print_signatures(diss):
    print("\n=== lesion signatures (vs matched controls and the pooled null) ===")
    for key, v in diss.items():
        band = f"[{v['control_band'][0]:.1%}, {v['control_band'][1]:.1%}]"
        note = ""
        if v["signature"] in STRUCTURED:
            c = v["signature"]
            note = (f"  {c} {v[f'count_{c}']}/{v['n_cells']} cells "
                    f"vs null {v[f'null_rate_{c}']:.1%}  p={v[f'p_{c}']:.1e}")
        if v.get("contains_critical_head"):
            note += f"  [caught {','.join(v['contains_critical_head'])}]"
        print(f"  {key:<38} acc {v['acc_top']:>5.1%} vs ctrl {band:<16} "
              f"ppl x{v['ppl_ratio']:>6.2f} -> {v['signature']}{note}")


def rescore(tag, domain, seed=0):
    """Re-score a battery from its own summary. No model, no GPU, seconds."""
    path = (ROOT / "results" / "lesion"
            / f"{tag}_{domain}_lesion_summary.json")
    if not path.exists():
        raise SystemExit(f"nothing to rescore: {path} not found")
    summary = json.loads(path.read_text())
    annotate_critical_heads(tag, domain, summary["runs"], seed=seed)
    summary["dissociation"] = score_dissociation(summary["runs"],
                                                 summary["baseline"])
    path.write_text(json.dumps(summary, indent=2))
    print_signatures(summary["dissociation"])
    print(f"\nrescored {path} ({len(summary['runs'])} runs)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--stage", default="dissoc",
                    choices=["screen", "anchor", "dissoc", "both"])
    ap.add_argument("--networks", nargs="+", default=None,
                    help="restrict the dissoc stage to these networks")
    ap.add_argument("--rescore", action="store_true",
                    help="re-score the existing summary and exit: no model, no GPU")
    ap.add_argument("--mode", default="mean", choices=["mean", "zero", "resample"])
    ap.add_argument("--neuron-sizes", type=int, nargs="+",
                    default=[32, 128, 512, 2048])
    ap.add_argument("--head-sizes", type=int, nargs="+", default=[4, 16, 48])
    ap.add_argument("--ppl-prompts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)

    if args.rescore:                       # must precede load_model: the point is no GPU
        rescore(tag, args.domain, seed=args.seed)
        return

    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    dims = unit_dims(model)
    ops = dom.op_keys

    loc = ROOT / "results" / "lesion" / f"{tag}_{args.domain}_localizer.parquet"
    if not loc.exists():
        raise SystemExit(f"run lesion_localize.py first: {loc} not found")
    df = pd.read_parquet(loc)

    corpus = get_corpus(128)[:args.ppl_prompts]
    print(f"[{tag}] reference means over {len(corpus)} WikiText prompts "
          f"(mean-ablation substrate)...")
    means = reference_means(model, corpus, dims=dims)

    # ---- baselines (no lesion) ----
    base_ppl = perplexity(model, corpus)
    base_rel = relations_readout(model, dom, tok, ops)
    base_arith = arithmetic_accuracy(model)
    stim = induction_stimuli(model, n=24, seed=args.seed)
    base_ind = induction_score(model, stim)
    print(f"clean: ppl {base_ppl:.2f} | relations {base_rel.correct.mean():.1%} "
          f"| arithmetic {base_arith:.1%} | induction "
          f"{base_ind['induction_score']:+.2f}")

    # identity check: an empty lesion must be a no-op
    with Lesion(model, mode=args.mode, means=means, dims=dims):
        chk = perplexity(model, corpus[:4])
    ref = perplexity(model, corpus[:4])
    assert abs(chk - ref) < 1e-4, f"empty lesion changed ppl: {chk} vs {ref}"
    print(f"identity check: empty lesion is a no-op (ppl {chk:.4f})")

    rows = []
    cell_rows = []                       # the raw per-cell generations, persisted so
                                         # any classification can be re-audited without
                                         # a re-run (§3.8): the same objection §2.16
                                         # answered, kept answered at this granularity

    def run(label, network, kind, k, units, cls_mode):
        heads = units if kind == "head" else ()
        neurons = units if kind == "neuron" else ()
        with Lesion(model, heads=heads, neurons=neurons, mode=args.mode,
                    means=means, dims=dims, seed=args.seed):
            rel = relations_readout(model, dom, tok, ops)
            ppl = perplexity(model, corpus)
            arith = arithmetic_accuracy(model)
            ind = induction_score(model, stim)
        dist = {c: float((rel["class"] == c).mean())
                for c in ("target", "other_operand", "other_relation", "degraded")}
        row = {"network": network, "control": cls_mode, "kind": kind, "k": k,
               "label": label, "relations_acc": float(rel.correct.mean()),
               **{f"class_{c}": v for c, v in dist.items()},
               "ppl": ppl, "ppl_ratio": ppl / base_ppl,
               "arithmetic": arith,
               "induction_score": ind["induction_score"],
               "induction_ratio": (ind["induction_score"]
                                   / (base_ind["induction_score"] + 1e-9))}
        rows.append(row)
        for c in rel.to_dict("records"):
            cell_rows.append({"network": network, "control": cls_mode,
                              "kind": kind, "k": k, **c})
        print(f"  {label:<34} acc {row['relations_acc']:>5.1%} "
              f"o_op {dist['other_operand']:>5.1%} o_rel {dist['other_relation']:>5.1%} "
              f"deg {dist['degraded']:>5.1%} | ppl x{row['ppl_ratio']:.2f} "
              f"arith {arith:>5.1%} ind {row['induction_score']:+.2f}")

    CONTROLS = ("top", "layer", "magnitude", "random")

    # ---- P0: criticality screen (infrastructure vs functional tissue) ----
    # Ablate every head ALONE and record the damage. Brains have this structure:
    # a millimetre of brainstem is fatal, a lobe of "silent" cortex can be lost
    # quietly. If LLMs have the same, a lesion study must know where the
    # infrastructure is before it can claim any deficit is about *function*.
    if args.stage == "screen":
        print("\n=== P0 CRITICALITY SCREEN: single-head lesions, all heads ===")
        srows = []
        short = corpus[:6]
        base_short = perplexity(model, short)
        for l in range(dims.n_layers):
            for h in range(dims.n_heads):
                with Lesion(model, heads=[(l, h)], mode=args.mode, means=means,
                            dims=dims):
                    p = perplexity(model, short)
                srows.append({"kind": "head", "layer": l, "index": h,
                              "depth": 100.0 * l / (dims.n_layers - 1),
                              "ppl": p, "ppl_ratio": p / base_short})
            worst = max(srows[-dims.n_heads:], key=lambda r: r["ppl_ratio"])
            print(f"  layer {l:>2}: worst head H{worst['index']} "
                  f"x{worst['ppl_ratio']:.2f}")
        sdf = pd.DataFrame(srows)
        sout = ROOT / "results" / "lesion" / f"{tag}_criticality.parquet"
        sdf.to_parquet(sout)
        crit = sdf[sdf.ppl_ratio > 2.0].sort_values("ppl_ratio", ascending=False)
        summary = {"baseline_ppl": base_short,
                   "n_heads": int(len(sdf)),
                   "critical_heads": [
                       {"layer": int(r.layer), "head": int(r["index"]),
                        "ppl_ratio": float(r.ppl_ratio)}
                       for _, r in crit.iterrows()],
                   "frac_critical": float((sdf.ppl_ratio > 2.0).mean()),
                   "median_ppl_ratio": float(sdf.ppl_ratio.median()),
                   "_meta": {"tag": tag, "mode": args.mode,
                             "n_prompts": len(short),
                             "note": "single-head mean-ablation, WikiText ppl. "
                                     "critical = ppl_ratio > 2. The existence of "
                                     "a few catastrophic heads is why control "
                                     "lesions must be DEPTH-MATCHED."}}
        sout.with_suffix("").with_name(sout.stem + "_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(f"\ncritical heads (ppl > 2x): {len(crit)} of {len(sdf)} "
              f"({summary['frac_critical']:.1%})")
        for c in summary["critical_heads"][:10]:
            print(f"  L{c['layer']}H{c['head']}: x{c['ppl_ratio']:.1f}")
        print(f"\nsaved {sout} (+ _summary.json)")
        return

    # ---- P1: the anchor (induction heads, known answer) ----
    if args.stage in ("anchor", "both"):
        print("\n=== P1 ANCHOR: induction heads (must destroy copying, spare LM) ===")
        for k in args.head_sizes:
            for cmode in CONTROLS:
                units, _ = pick(df, "head", "induction", k, mode=cmode,
                                seed=args.seed)
                run(f"induction/{cmode} heads k={k}", "induction", "head", k,
                    units, cmode)

    # ---- P2: the double dissociation ----
    if args.stage in ("dissoc", "both"):
        print("\n=== P2 DOUBLE DISSOCIATION: operator vs operand networks ===")
        for kind, sizes in (("neuron", args.neuron_sizes), ("head", args.head_sizes)):
            for k in sizes:
                nets = args.networks or ("operator", "operand", "operand_entity")
                for network in nets:
                    for cmode in CONTROLS:
                        units, _ = pick(df, kind, network, k, mode=cmode,
                                        seed=args.seed)
                        run(f"{network}/{cmode} {kind}s k={k}", network, kind, k,
                            units, cmode)

    out = ROOT / "results" / "lesion" / f"{tag}_{args.domain}_lesion.parquet"
    res = pd.DataFrame(rows)

    # A restricted run (--stage anchor, --networks X) produces a SUBSET of the
    # battery. Writing it straight out would silently destroy every run it does
    # not repeat -- 84 rows replaced by 28. Merge on the run's identity instead,
    # this run's rows winning. Safe because the pipeline is deterministic: the
    # 1.7B entity-position re-run reproduced all 56 prior runs byte-identically.
    if out.exists():
        keys = ["network", "control", "kind", "k"]
        old = pd.read_parquet(out)
        keep = ~pd.MultiIndex.from_frame(old[keys]).isin(
            pd.MultiIndex.from_frame(res[keys]))
        if keep.any():
            print(f"merging {int(keep.sum())} preserved runs from {out.name}")
            res = pd.concat([old[keep], res], ignore_index=True)
    res.to_parquet(out)

    # the raw generations, merged on run identity like the summary parquet.
    # Gitignored like every other parquet -- re-auditable on the machine that ran
    # it, which is what §3.8 asks; the class distributions in the summary stay the
    # committed record.
    if cell_rows:
        cells_out = (ROOT / "results" / "lesion"
                     / f"{tag}_{args.domain}_lesion_cells.parquet")
        cres = pd.DataFrame(cell_rows)
        ckeys = ["network", "control", "kind", "k"]
        if cells_out.exists():
            cold = pd.read_parquet(cells_out)
            ckeep = ~pd.MultiIndex.from_frame(cold[ckeys]).isin(
                pd.MultiIndex.from_frame(cres[ckeys].drop_duplicates()))
            if ckeep.any():
                cres = pd.concat([cold[ckeep], cres], ignore_index=True)
        cres.to_parquet(cells_out)

    summary = {
        "baseline": {"ppl": base_ppl, "relations_acc": float(base_rel.correct.mean()),
                     "arithmetic": base_arith, **base_ind},
        "runs": {r["label"]: {k: v for k, v in r.items() if k != "label"}
                 for r in res.to_dict("records")},
        "_meta": {"tag": tag, "domain": args.domain, "stage": args.stage,
                  "ablation_mode": args.mode, "seed": args.seed,
                  "dims": {"n_layers": dims.n_layers, "n_heads": dims.n_heads,
                           "d_mlp": dims.d_mlp},
                  "ppl_prompts": len(corpus),
                  "note": "TRUE lesion: units are silenced at every token "
                          "position of every prompt. mean-ablation pins each unit "
                          "to its WikiText mean. class_other_operand = right "
                          "relation, wrong entity; class_other_relation = right "
                          "entity, wrong relation (no 'source' class exists in a "
                          "lesion: the prompt always asks the true question)."}}

    # the per-network lesion signature, if P2 ran
    if args.stage in ("dissoc", "both"):
        annotate_critical_heads(tag, args.domain, summary["runs"], seed=args.seed)
        summary["dissociation"] = score_dissociation(summary["runs"],
                                                     summary["baseline"])
        print_signatures(summary["dissociation"])

    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ _summary.json)")


if __name__ == "__main__":
    main()
