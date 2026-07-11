#!/usr/bin/env python
"""Audit the generation-level claims: what does the model ACTUALLY say?

A reader raised the right objection to the generation metrics: a two-token logit
margin and a 3-token containment check can both miss semantically correct answers
that surface differently ("the euro", "euros", a clause that reaches the answer at
token 5), and no one has classified the raw generations. This script re-runs the
key intervention conditions and scores each cell three ways:

  1. GREEDY, LONGER LEASH: decode k=8 tokens (not 3) and save the RAW TEXT, so
     every number below can be re-derived by a human reading the outputs.
  2. CLASSIFICATION (the reader's taxonomy): each generation is labeled
        target          contains the gold target answer (case-insensitive substring,
                        so "the euro"/"euros"/"Euro" all count)
        source          contains the source relation's answer instead
        other_operand   contains another operand's answer for the TARGET relation
                        (right relation, wrong entity)
        other_relation  contains the same operand's answer for a DIFFERENT relation
        degraded        none of the above
     Priority order target > source > other_operand > other_relation; the raw text
     is kept for manual reclassification.
  3. FORCED CHOICE (constrained generation): teacher-forced mean log-prob of each
     of the operand's gold answers as a full SEQUENCE (all tokens, not the first),
     length-normalized; report which answer wins. This is immune to "the"
     interception, multi-token answers, and surface-variant mass splitting across
     first tokens.

Conditions audited (relations): clean; composed patch (mu+operand+operator, band);
full donor patch (band); additive band/all at alpha=4 (the old "0.5%" claim);
additive band/all at alpha=0.1 (the calibrated dose); additive single-layer/query
at alpha=1. For animals: clean + composed + donor (the "6% floor" check).

    python scripts/op_audit.py 1.7b --domain relations
    python scripts/op_audit.py 1.7b --domain animals
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core
import op_minimal

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def seq_logprob(model, prompt, answer, leading_space=True):
    """Mean log-prob of the answer's tokens, teacher-forced after the prompt."""
    pre = " " if leading_space else ""
    ids_p = model.encode(prompt, max_length=64)
    ids_a = model.tokenizer.encode(pre + str(answer).strip(), add_special_tokens=False)
    ids = torch.cat([ids_p, torch.tensor([ids_a], device=ids_p.device)], dim=1)
    from jlens import ActivationRecorder
    final = model.n_layers - 1
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0]
    logits = model.unembed(h.float())
    lp = F.log_softmax(logits, -1)
    n_p = ids_p.shape[1]
    tot = 0.0
    for j, t in enumerate(ids_a):
        tot += float(lp[n_p - 1 + j, t])
    return tot / len(ids_a)


def classify(text, target, source, other_operand_answers, other_relation_answers):
    t = text.strip().lower()
    if str(target).strip().lower() in t:
        return "target"
    if str(source).strip().lower() in t:
        return "source"
    if any(str(a).strip().lower() in t for a in other_operand_answers):
        return "other_operand"
    if any(str(a).strip().lower() in t for a in other_relation_answers):
        return "other_relation"
    return "degraded"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--k", type=int, default=8, help="greedy tokens (longer leash)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(layers, model.n_layers)["workspace"]
    mid = ws[len(ws) // 2]
    ops = dom.op_keys
    pairs = [(a, b) for a in ops for b in ops if a != b]

    print(f"[{tag} / {args.domain}] generation audit, k={args.k}")

    v = op_core.op_dirs(model, ws, dom)
    comp = op_core.factorize_components(model, ws, dom, pos=-1)

    # {condition: builder(frm, to, o) -> greedy kwargs}
    def conds(frm, to, o):
        dv4 = {l: 4.0 * (v[to][l] - v[frm][l]) for l in ws}
        dv01 = {l: 0.1 * (v[to][l] - v[frm][l]) for l in ws}
        dv1 = {mid: 1.0 * (v[to][mid] - v[frm][mid])}
        composed = {l: comp["mu"][l] + comp["stem"][l][o] + comp["case"][l][to]
                    for l in ws}
        donor = {l: comp["mu"][l] + comp["stem"][l][o] + comp["case"][l][to]
                 + comp["inter"][l][(o, to)] for l in ws}
        c = {
            "clean": {},
            "patch composed": {"patch": composed},
            "patch donor": {"patch": donor},
            "add band a=4": {"add": dv4},
            "add band a=0.1": {"add": dv01},
            "add 1L query a=1": {"add": dv1, "add_positions": [-1]},
        }
        return c

    cond_names = ["clean", "patch composed", "patch donor", "add band a=4",
                  "add band a=0.1", "add 1L query a=1"]
    if args.domain != "relations":
        cond_names = ["clean", "patch composed", "patch donor"]

    rows = []
    for frm, to in pairs:
        for o in dom.operand_keys:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            p = dom.render(o, frm)
            target, source = str(dom.answer(o, to)), str(dom.answer(o, frm))
            other_op = [str(dom.answer(oo, to)) for oo in dom.operand_keys
                        if oo != o and str(dom.answer(oo, to)).lower()
                        not in (target.lower(), source.lower())]
            other_rel = [str(dom.answer(o, k)) for k in ops if k not in (frm, to)
                         and str(dom.answer(o, k)).lower()
                         not in (target.lower(), source.lower())]
            cc = conds(frm, to, o)
            for name in cond_names:
                kw = cc[name]
                text = op_minimal.greedy(model, p, k=args.k, **kw)
                cls = classify(text, target, source, other_op, other_rel)
                # forced choice among the operand's gold answers, full-sequence
                # log-prob under the SAME intervention (hooked forward)
                cands = {k: str(dom.answer(o, k)) for k in ops}
                n_prompt = model.encode(p, max_length=64).shape[1]
                # dedup surface-identical candidates (syncretism) keeping labels
                scores = {}
                for k, ans in cands.items():
                    key = ans.lower()
                    if key not in scores:
                        with _hooked(model, kw, n_prompt):
                            scores[key] = (seq_logprob(
                                model, p, ans, dom.answer_leading_space), k, ans)
                best = max(scores.values(), key=lambda x: x[0])
                fc_target = best[2].lower() == target.lower()
                rows.append({"from": frm, "to": to, "operand": o,
                             "condition": name, "text": text, "class": cls,
                             "forced_choice_target": fc_target,
                             "forced_choice_says": best[2]})

    df = pd.DataFrame(rows)
    print(f"\n{'condition':<18} " + " ".join(f"{c:>13}" for c in
          ["target", "source", "other_operand", "other_relation", "degraded"])
          + f" {'forced-choice':>13}")
    summary = {}
    for name in cond_names:
        s = df[df["condition"] == name]
        dist = {c: float((s["class"] == c).mean())
                for c in ["target", "source", "other_operand", "other_relation",
                          "degraded"]}
        fc = float(s["forced_choice_target"].mean())
        summary[name] = {**dist, "forced_choice_target": fc, "n": int(len(s))}
        print(f"{name:<18} " + " ".join(f"{dist[c]:>12.1%}" for c in dist)
              + f" {fc:>12.1%}")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_audit.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    summary["_meta"] = {"tag": tag, "domain": args.domain, "k": args.k,
                        "ws": [int(l) for l in ws], "mid_layer": int(mid),
                        "note": "class = substring taxonomy on k-token greedy text "
                                "(raw texts persisted); forced choice = "
                                "length-normalized full-sequence log-prob among the "
                                "operand's gold answers, under the intervention"}
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ .json) — raw generations included for manual review")


class _hooked:
    """Context manager: register the same add/patch hooks greedy() uses, so the
    forced-choice sequence scoring sees the identical intervention. `n_prompt`
    anchors position-dependent hooks at the prompt's query token even though the
    teacher-forced forward extends the sequence with answer tokens."""
    def __init__(self, model, kw, n_prompt):
        self.model, self.kw = model, kw
        self.n_prompt = n_prompt
        self.handles = []

    def __enter__(self):
        add = self.kw.get("add")
        patch = self.kw.get("patch")
        addpos = self.kw.get("add_positions")
        m = self.model
        if add:
            # resolve negative positions against the PROMPT length, not the
            # teacher-forced sequence (else [-1] would hit the last answer token)
            pos = None if addpos is None else [p_ % self.n_prompt for p_ in addpos]
            for l, vec in add.items():
                def mk(vc):
                    def h(_m, _i, o):
                        a = o[0] if isinstance(o, tuple) else o
                        if pos is None:
                            a = a + vc.to(a.dtype)
                        else:
                            a = a.clone()
                            for p_ in pos:
                                a[:, p_, :] = a[:, p_, :] + vc.to(a.dtype)
                        return (a, *o[1:]) if isinstance(o, tuple) else a
                    return h
                self.handles.append(m.layers[l].register_forward_hook(mk(vec)))
        if patch:
            # NOTE: patch anchors at the prompt's last token; for teacher-forced
            # scoring the prompt is extended by the answer, so anchor by absolute
            # index captured here is wrong -- instead re-anchor per forward via
            # negative indexing is unavailable; we approximate by patching the
            # position where the query token sits: len(prompt tokens) - 1. The
            # caller passes prompts of fixed template, so we compute it lazily on
            # first forward from the stored n_prompt.
            for l, vec in patch.items():
                def mkp(vc):
                    def h(_m, _i, o):
                        a = o[0] if isinstance(o, tuple) else o
                        a = a.clone()
                        a[:, self.n_prompt - 1, :] = vc.to(a.dtype)
                        return (a, *o[1:]) if isinstance(o, tuple) else a
                    return h
                self.handles.append(m.layers[l].register_forward_hook(mkp(vec)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


if __name__ == "__main__":
    main()
