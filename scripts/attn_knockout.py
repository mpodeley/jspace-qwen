#!/usr/bin/env python
"""Does the operand live in the wiring, not the tissue?

Part 3's lesion study removed MLP neurons at both read positions and found the
operator has removable tissue and the operand has none. But a lesion removes
STORED computation. The entity is not stored at the query position — it is
MOVED there, by attention, from the entity token. So the natural next test is
not another lesion but an attention KNOCKOUT (Geva et al., 2304.14767): sever
the query→entity edge and ask whether *that* produces the operand deficit the
neuron lesions could not.

The hypothesis that would explain the asymmetry: the operator is stored (in MLP
tissue, removable by lesion) and the operand is routed (in attention edges,
removable only by cutting the wire). Two substrates for the two halves of a
factorization that is symmetric in the algebra.

  sweep        A sliding window of W layers, query→entity attention blocked
               across all heads, swept over depth. Geva's design: it localizes
               WHERE the entity is extracted to the query. The accuracy dip and
               the other_operand rise, as a function of the blocked band.
  dissociate   At the critical band, block query→ENTITY vs query→OPERATOR-WORD
               vs query→a filler token (matched count). If entity-knockout
               produces the operand deficit and operator-word-knockout does not
               (or produces the operator deficit), the two halves dissociate by
               substrate.

Readout is op_audit.classify, as in Part 3: other_operand = right relation,
wrong entity (the operand deficit); other_relation = right entity, wrong
relation (the operator deficit); degraded = neither. The prompt always asks the
true question — we cut a wire, we do not steer.

    python scripts/attn_knockout.py 1.7b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import load_model, resolve_tag
from jlens import ActivationRecorder
import op_core
import op_minimal
from op_audit import classify

ROOT = Path(__file__).resolve().parent.parent


# --- locating the entity and operator-word tokens -----------------------------

def token_spans(model, dom, o, k):
    """Token positions of the entity and the operator phrase in dom.render(o,k),
    via character-offset mapping (robust to multi-token entities/operators and
    to the leading space). Returns dict with 'entity', 'operator', 'filler'
    (positions that are neither, excluding the final query token), 'query'."""
    tok = model.tokenizer
    prompt = dom.render(o, k)
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=False)
    offs = enc["offset_mapping"]
    n = len(offs)
    # the model prepends BOS (force_bos=True); encode() gives [BOS, ...tokens].
    # offsets align to the non-BOS tokens, so shift by the BOS count.
    n_bos = model.encode(prompt, max_length=64).shape[1] - n
    entity_str = str(dom.items[o]["args"]["a"])
    op_phrase = dom.operators[k]
    if "{" in op_phrase:
        op_phrase = op_phrase.format(**dom.items[o]["args"])
    def span_positions(substr):
        c0 = prompt.find(substr)
        if c0 < 0:
            return []
        c1 = c0 + len(substr)
        return [i + n_bos for i, (a, b) in enumerate(offs)
                if a < c1 and b > c0 and b > a]
    ent = span_positions(entity_str)
    ope = span_positions(op_phrase)
    query = n + n_bos - 1
    used = set(ent) | set(ope) | {query}
    filler = [i + n_bos for i in range(n) if (i + n_bos) not in used]
    return {"entity": ent, "operator": ope, "filler": filler, "query": query,
            "n_prompt": n + n_bos}


# --- the knockout -------------------------------------------------------------

class Knockout:
    """Block attention edges query_pos→key_pos in the given layers, all heads.
    `block` = {layer: (q_from, [key_positions])}: every query position >= q_from
    stops attending to key_positions. Forces eager attention (the mask is only
    materialized there) and restores it on exit. An empty key list is a no-op,
    asserted bit-exact in main()."""

    def __init__(self, model, layers, q_from, keys):
        self.hf = model._hf_model
        self.layers = layers
        self.q_from = q_from
        self.keys = list(keys)
        self.h = []
        self._old_impl = None

    def __enter__(self):
        self._old_impl = self.hf.config._attn_implementation
        self.hf.config._attn_implementation = "eager"
        neg = torch.finfo(torch.bfloat16).min
        for l in self.layers:
            def pre(mod, args, kwargs, qf=self.q_from, keys=self.keys):
                m = kwargs.get("attention_mask")
                if m is None or not keys:
                    return None
                m = m.clone()
                Q = m.shape[-2]
                for q in range(qf, Q):
                    for kk in keys:
                        if kk < m.shape[-1]:
                            m[..., q, kk] = neg
                kwargs["attention_mask"] = m
                return (args, kwargs)
            self.h.append(self.hf.model.layers[l].self_attn
                          .register_forward_pre_hook(pre, with_kwargs=True))
        return self

    def __exit__(self, *a):
        for h in self.h:
            h.remove()
        self.h = []
        self.hf.config._attn_implementation = self._old_impl
        return False


@torch.no_grad()
def knockout_greedy(model, prompt, layers, q_from, keys, k=8):
    with Knockout(model, layers, q_from, keys):
        return op_minimal.greedy(model, prompt, k=k)


# --- measurement --------------------------------------------------------------

def readout(model, dom, ops, layers, target_key, k=8):
    """Block query→<target_key> across `layers` for every cell and classify the
    generation. target_key in {'entity','operator','filler'}."""
    tok = model.tokenizer
    rows = []
    for o in dom.operand_keys:
        for rel in ops:
            sp = token_spans(model, dom, o, rel)
            keys = sp[target_key]
            if not keys:
                continue
            prompt = dom.render(o, rel)
            target = str(dom.answer(o, rel))
            other_op = [str(dom.answer(oo, rel)) for oo in dom.operand_keys
                        if oo != o and str(dom.answer(oo, rel)).lower()
                        != target.lower()]
            other_rel = [str(dom.answer(o, kk)) for kk in ops if kk != rel
                         and str(dom.answer(o, kk)).lower() != target.lower()]
            text = knockout_greedy(model, prompt, layers, sp["query"], keys, k=k)
            cls = classify(text, target, target, other_op, other_rel)
            rows.append({"operand": o, "relation": rel, "target_key": target_key,
                         "text": text, "class": cls, "correct": cls == "target",
                         "n_keys": len(keys)})
    return pd.DataFrame(rows)


def dist(df):
    return {c: float((df["class"] == c).mean())
            for c in ("target", "other_operand", "other_relation", "degraded")}


def paired_dissociation(cells_df, seed=0):
    """Paired, per-cell tests on the critical-window generations. The unit is the
    (operand, relation) cell, scored under each knockout target, so the
    comparison is within-cell: a bootstrap CI of the accuracy difference and a
    McNemar exact test of which target breaks which cells. The theoretically
    loaded contrast is entity-vs-operator -- the operator WORD should be inert if
    the operator is constructed, not attention-read."""
    import numpy as np
    from math import comb
    piv = cells_df.pivot_table(index=["operand", "relation"], columns="cond",
                               values="correct")
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in (("entity", "operator"), ("entity", "filler")):
        if a not in piv or b not in piv:
            continue
        d = (piv[a] - piv[b]).dropna().to_numpy()
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean()
                       for _ in range(10000)])
        # McNemar: cells broken by a-only vs b-only
        aa, bb = piv[a].fillna(True), piv[b].fillna(True)
        a_only = int(((~aa.astype(bool)) & bb.astype(bool)).sum())
        b_only = int((aa.astype(bool) & (~bb.astype(bool))).sum())
        n = a_only + b_only
        pmc = (2 * sum(comb(n, i) for i in range(min(a_only, b_only) + 1))
               / 2 ** n) if n else 1.0
        out[f"{a}_vs_{b}"] = {
            "acc_diff": float(d.mean()),
            "lo": float(np.percentile(bs, 2.5)),
            "hi": float(np.percentile(bs, 97.5)),
            "ci_excludes_0": bool(np.percentile(bs, 97.5) < 0),
            f"{a}_only_broken": a_only, f"{b}_only_broken": b_only,
            "mcnemar_p": float(min(pmc, 1.0))}
    out["other_operand_counts"] = {
        c: int((cells_df[cells_df.cond == c]["class"] == "other_operand").sum())
        for c in cells_df.cond.unique()}
    out["n_cells"] = int(piv.shape[0])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--window", type=int, default=0,
                    help="sliding-window width for the sweep (0 = n_layers//5)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rescore", action="store_true",
                    help="recompute paired stats from the persisted parquet, no GPU")
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    if args.rescore:
        out = ROOT / "results" / "lesion" / f"{tag}_{args.domain}_knockout.parquet"
        spath = out.with_suffix("").with_name(out.stem + "_summary.json")
        summary = json.loads(spath.read_text())
        summary["paired"] = paired_dissociation(pd.read_parquet(out), seed=args.seed)
        spath.write_text(json.dumps(summary, indent=2))
        pv = summary["paired"]["entity_vs_operator"]
        print(f"rescored {spath}: entity vs operator Δacc {pv['acc_diff']:+.1%} "
              f"[{pv['lo']:+.1%}, {pv['hi']:+.1%}], McNemar p={pv['mcnemar_p']:.1e}")
        return

    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    ops = dom.op_keys
    W = args.window or max(1, model.n_layers // 5)

    # baseline + no-op identity check
    base = readout(model, dom, ops, [], "entity")   # empty layers = no knockout
    p = dom.render("Italy", "currency")
    sp = token_spans(model, dom, "Italy", "currency")
    from jlens import ActivationRecorder as AR
    def final_logit(prompt, layers, keys, qf):
        ids = model.encode(prompt, max_length=64)
        with Knockout(model, layers, qf, keys):
            with AR(model.layers, at=[model.n_layers - 1]) as rec:
                model.forward(ids)
                h = rec.activations[model.n_layers - 1][0, -1]
        return model.unembed(h[None].float())[0]
    L0 = final_logit(p, [], [], sp["query"])
    Lnoop = final_logit(p, list(range(model.n_layers)), [], sp["query"])
    assert float((L0 - Lnoop).abs().max()) < 1e-3, "empty-key knockout not a no-op"
    print(f"[{tag}] baseline relations acc {base.correct.mean():.1%} "
          f"(n={len(base)}); window W={W}; entity token located in "
          f"{(base.n_keys > 0).all() and 'every' or 'some'} cell")

    # ---- sweep: where is the entity extracted? ----
    print("\n=== SWEEP: query->entity blocked in a sliding window of layers ===")
    sweep = []
    starts = list(range(0, model.n_layers - W + 1, max(1, W // 2)))
    for s in starts:
        layers = list(range(s, min(s + W, model.n_layers)))
        df = readout(model, dom, ops, layers, "entity", k=args.k)
        d = dist(df)
        mid_depth = 100.0 * (s + W / 2) / (model.n_layers - 1)
        sweep.append({"start": s, "layers": layers, "mid_depth": mid_depth,
                      "acc": float(df.correct.mean()), **d})
        print(f"  L{s:>2}-{layers[-1]:<2} ({mid_depth:>3.0f}%): acc {df.correct.mean():>5.1%} "
              f"o_op {d['other_operand']:>5.1%} o_rel {d['other_relation']:>5.1%} "
              f"deg {d['degraded']:>5.1%}")

    # the critical window = the one that most destroys accuracy
    crit = min(sweep, key=lambda r: r["acc"])
    crit_layers = crit["layers"]
    print(f"\ncritical window: L{crit_layers[0]}-{crit_layers[-1]} "
          f"({crit['mid_depth']:.0f}% depth), accuracy {crit['acc']:.1%}")

    # ---- dissociation at the critical window ----
    print(f"\n=== DISSOCIATION at L{crit_layers[0]}-{crit_layers[-1]}: "
          f"block query-> entity vs operator-word vs filler ===")
    diss = {}
    diss_dfs = []
    for tgt in ("entity", "operator", "filler"):
        df = readout(model, dom, ops, crit_layers, tgt, k=args.k)
        diss_dfs.append(df.assign(cond=tgt))
        d = dist(df)
        diss[tgt] = {"acc": float(df.correct.mean()), "n": int(len(df)), **d}
        print(f"  block->{tgt:<9} acc {df.correct.mean():>5.1%} "
              f"o_op {d['other_operand']:>5.1%} o_rel {d['other_relation']:>5.1%} "
              f"deg {d['degraded']:>5.1%}  (n={len(df)})")

    out = ROOT / "results" / "lesion" / f"{tag}_{args.domain}_knockout.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    # the critical-window generations, persisted for audit (reused from the
    # dissociation loop above -- no extra forwards)
    cells_df = pd.concat(diss_dfs, ignore_index=True)
    cells_df.to_parquet(out)
    paired = paired_dissociation(cells_df, seed=args.seed)
    print(f"\n  entity vs operator: Δacc {paired['entity_vs_operator']['acc_diff']:+.1%} "
          f"[{paired['entity_vs_operator']['lo']:+.1%}, "
          f"{paired['entity_vs_operator']['hi']:+.1%}], McNemar p="
          f"{paired['entity_vs_operator']['mcnemar_p']:.1e}")
    summary = {"baseline_acc": float(base.correct.mean()),
               "window": W, "sweep": sweep,
               "critical_window": {"layers": crit_layers, "mid_depth": crit["mid_depth"]},
               "dissociation": diss, "paired": paired,
               "_meta": {"tag": tag, "domain": args.domain, "k": args.k,
                         "n_cells": int(len(base)),
                         "note": "query->KEY attention blocked across a layer "
                                 "window, all heads, every query position >= the "
                                 "prompt's last token. entity/operator token "
                                 "positions from char-offset mapping. class via "
                                 "op_audit.classify (source=target, no steer)."}}
    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ _summary.json)")


if __name__ == "__main__":
    main()
