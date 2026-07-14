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

Controls, in increasing order of how much they hurt:
    random          k units drawn uniformly
    magnitude       k units matched on activation magnitude but NOT selective --
                    "the same amount of tissue, in the wrong place"
    permuted        top-k under permuted relation labels (the paper's null,
                    lifted to unit selection)
    cross           the OTHER network (this IS the dissociation)

A null deficit is ambiguous: the IOI circuit's backup heads show that networks
compensate. So we sweep the lesion size and report the SELECTIVITY RATIO (deficit
on the targeted function over deficit on the spared one), not the absolute
deficit.

    python scripts/lesion_battery.py 1.7b --stage anchor    # P1: induction gate
    python scripts/lesion_battery.py 1.7b --stage dissoc    # P2: the centerpiece
"""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--stage", default="dissoc",
                    choices=["screen", "anchor", "dissoc", "both"])
    ap.add_argument("--networks", nargs="+", default=None,
                    help="restrict the dissoc stage to these networks")
    ap.add_argument("--mode", default="mean", choices=["mean", "zero", "resample"])
    ap.add_argument("--neuron-sizes", type=int, nargs="+",
                    default=[32, 128, 512, 2048])
    ap.add_argument("--head-sizes", type=int, nargs="+", default=[4, 16, 48])
    ap.add_argument("--ppl-prompts", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
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
    res.to_parquet(out)

    summary = {
        "baseline": {"ppl": base_ppl, "relations_acc": float(base_rel.correct.mean()),
                     "arithmetic": base_arith, **base_ind},
        "runs": {r["label"]: {k: v for k, v in r.items() if k != "label"}
                 for r in rows},
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

    # the dissociation contrast, if P2 ran
    if args.stage in ("dissoc", "both"):
        d = res[(res.control == "top")]
        diss = {}
        for kind in d.kind.unique():
            for k in sorted(d[d.kind == kind].k.unique()):
                s = d[(d.kind == kind) & (d.k == k)]
                if set(s.network) >= {"operator", "operand"}:
                    op = s[s.network == "operator"].iloc[0]
                    od = s[s.network == "operand"].iloc[0]
                    diss[f"{kind}/k={k}"] = {
                        "operator_lesion_other_relation": op["class_other_relation"],
                        "operator_lesion_other_operand": op["class_other_operand"],
                        "operand_lesion_other_operand": od["class_other_operand"],
                        "operand_lesion_other_relation": od["class_other_relation"],
                        "crossed": bool(
                            op["class_other_relation"] > op["class_other_operand"]
                            and od["class_other_operand"] > od["class_other_relation"]),
                        "ppl_ratio_operator": op["ppl_ratio"],
                        "ppl_ratio_operand": od["ppl_ratio"]}
        summary["dissociation"] = diss
        print("\n=== dissociation check (crossed = predicted pattern holds) ===")
        for key, v in diss.items():
            print(f"  {key:<14} operator-lesion: o_rel {v['operator_lesion_other_relation']:.1%} "
                  f"vs o_op {v['operator_lesion_other_operand']:.1%} | "
                  f"operand-lesion: o_op {v['operand_lesion_other_operand']:.1%} "
                  f"vs o_rel {v['operand_lesion_other_relation']:.1%} | "
                  f"{'CROSSED ✓' if v['crossed'] else 'not crossed'}")

    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ _summary.json)")


if __name__ == "__main__":
    main()
