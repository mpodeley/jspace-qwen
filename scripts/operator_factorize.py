#!/usr/bin/env python
"""Factorize the workspace representation of a relational query into operator
(case) and operand (stem), and isolate a pure desinence.

Companion to operator_paradigm.py. Where that script shows the cases are
causally-manipulable directions, this one asks how the representation is
STRUCTURED:

  (A) two-way factorization  H[country, case] = mu + stem(c) + case(k) + interaction
      Variance shares say how much is the operand, the operator, and the FUSION
      (the interaction, where stem and ending do not cleanly concatenate -- as in
      the phonological fusion of a Latin declension). Principal angles between the
      stem- and case-subspaces say whether operator and operand live in separate
      subspaces. Reported in the raw residual and in the J-space readout.

  (B) pure desinence  v = mean_country[ h(language) - h(demonym) ], built from the
      relations that emit the SAME word (Italian = language = demonym), so the
      exponent cancels and the direction is a case marker stripped of its form.
      Its causal efficacy (added to a currency prompt) shows the desinence is
      functional, not just a surface artifact.

  (C) reading-position control: repeat (A) at the country token (--pos -2) instead
      of the query token, to check the case-dominance is not mere template echo.
      The stronger control is that (B) and operator_paradigm's swaps are causal.

    python scripts/operator_factorize.py 1.7b
    python scripts/operator_factorize.py 8b
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import torch

from _common import band_layers, first_token, load_model, resolve_tag
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent


def _load():
    d = json.loads((ROOT / "data" / "relations.json").read_text())
    return d["meta"]["template"], d["meta"]["cases"], d["answers"]


@torch.no_grad()
def _resid_at(model, prompt, layer, pos, *, lens=None, use_j=False):
    ids = model.encode(prompt, max_length=64)
    with ActivationRecorder(model.layers, at=[layer]) as rec:
        model.forward(ids)
    h = rec.activations[layer][0, pos].float()
    return lens.transport(h[None], layer)[0] if use_j else h


@torch.no_grad()
def _final_logits(model, prompt, hook_vecs=None):
    ids = model.encode(prompt, max_length=64)
    final = model.n_layers - 1
    handles = []
    if hook_vecs:
        for l, v in hook_vecs.items():
            def mk(vec):
                def h(m, i, o):
                    a = o[0] if isinstance(o, tuple) else o
                    a = a + vec.to(a.dtype)
                    return (a, *o[1:]) if isinstance(o, tuple) else a
                return h
            handles.append(model.layers[l].register_forward_hook(mk(v)))
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0, -1]
    for hd in handles:
        hd.remove()
    return model.unembed(h[None])[0]


def factorize(model, lens, layer, pos, tmpl, cases, answers, use_j, name):
    countries, keys = list(answers), list(cases)
    H = {(c, k): _resid_at(model, tmpl.format(rel=cases[k], country=c), layer, pos,
                           lens=lens, use_j=use_j)
         for c in countries for k in keys}
    mu = torch.stack([H[(c, k)] for c in countries for k in keys]).mean(0)
    stem = {c: torch.stack([H[(c, k)] for k in keys]).mean(0) - mu for c in countries}
    case = {k: torch.stack([H[(c, k)] for c in countries]).mean(0) - mu for k in keys}
    tot = sum(float((H[(c, k)] - mu).pow(2).sum()) for c in countries for k in keys)
    v_stem = len(keys) * sum(float(stem[c].pow(2).sum()) for c in countries)
    v_case = len(countries) * sum(float(case[k].pow(2).sum()) for k in keys)
    inter = sum(float((H[(c, k)] - mu - stem[c] - case[k]).pow(2).sum())
                for c in countries for k in keys)
    S = torch.stack([stem[c] for c in countries]).t()
    T = torch.stack([case[k] for k in keys]).t()
    Qs, _ = torch.linalg.qr(S)
    Qt, _ = torch.linalg.qr(T)
    sv = torch.linalg.svdvals(Qs.t() @ Qt).clamp(-1, 1)
    angles = [round(float(a), 1) for a in torch.rad2deg(torch.arccos(sv))]
    print(f"  {name:<22} stem={v_stem/tot:5.1%}  case={v_case/tot:5.1%}  "
          f"interaction={inter/tot:5.1%}  angles={angles}")
    return {"stem": v_stem/tot, "case": v_case/tot, "interaction": inter/tot,
            "angles": angles}


def pure_desinence(model, lens, ws, tmpl, cases, answers):
    countries = list(answers)
    dev = model._lm_head.weight.device
    same = [c for c in countries if answers[c]["language"] == answers[c]["demonym"]]
    v = {l: torch.zeros(model.d_model, device=dev) for l in ws}
    for c in same:
        rl = {l: _resid_at(model, tmpl.format(rel=cases["language"], country=c), l, -1) for l in ws}
        rd = {l: _resid_at(model, tmpl.format(rel=cases["demonym"], country=c), l, -1) for l in ws}
        for l in ws:
            v[l] += (rl[l] - rd[l]) / len(same)
    tok = model.tokenizer
    shifts = []
    for c in countries[:8]:
        p = tmpl.format(rel=cases["currency"], country=c)
        L0 = _final_logits(model, p)
        Lv = _final_logits(model, p, {l: 6.0 * v[l] for l in ws})
        lan, cur = first_token(tok, answers[c]["language"]), first_token(tok, answers[c]["currency"])
        shifts.append((float(L0[lan] - L0[cur]), float(Lv[lan] - Lv[cur])))
    return same, shifts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    tmpl, cases, answers = _load()
    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    lens = JacobianLens.from_pretrained(str(lens_path))
    ws = band_layers(lens.source_layers, model.n_layers)["workspace"]
    L = ws[len(ws) // 2]

    print(f"\n(A) two-way factorization  [{tag}, layer {L}, query position]")
    factorize(model, lens, L, -1, tmpl, cases, answers, False, "raw residual")
    factorize(model, lens, L, -1, tmpl, cases, answers, True, "J-space")

    print(f"\n(C) reading-position control  [country token, position -2]")
    factorize(model, lens, L, -2, tmpl, cases, answers, False, "raw @ country tok")

    print(f"\n(B) pure desinence (language-demonym, exponent fixed) -> currency prompt")
    same, shifts = pure_desinence(model, lens, ws, tmpl, cases, answers)
    clean = st.mean(s[0] for s in shifts)
    plus = st.mean(s[1] for s in shifts)
    print(f"  exponent-shared countries: {len(same)}")
    print(f"  logit(language)-logit(currency):  clean={clean:+.2f}  +desinence={plus:+.2f}")
    print(f"  (positive shift = the exponent-free case marker installs the relation)")


if __name__ == "__main__":
    main()
