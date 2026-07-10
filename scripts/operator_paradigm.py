#!/usr/bin/env python
"""The model's relational 'declension paradigm'.

Treat relations (currency-of, capital-of, language-of, demonym-of, continent-of)
as CASES applied to a country STEM. The workspace holds not the bare concept but
the concept marked by its pending relational role -- Italy-in-the-currency-case vs
Italy-in-the-capital-case -- and the case is a manipulable direction (a function
vector; cf. Todd et al. 2024, Hernandez et al. 2024 relational linearity).

Two measurements:

  (A) CAUSAL case-swap (raw residual). Build v(case) as the mean, over stems, of
      the deviation of that case from the stem's average-over-cases. Add
      v(to)-v(from) to a `from` prompt and read the answer-token logit shift, vs a
      matched-norm random control. A working swap flips a strongly-negative clean
      score (from-answer wins) to strongly positive (to-answer wins).

  (B) PARADIGM GEOMETRY, J-space vs logit-space. Build the case directions in the
      J-lens readout (unembed(J h)) and in the logit-lens readout (unembed(h)) --
      the J-space vs logit-space of the flow figure -- and compare how separated
      the cases are, and how they cluster.

Findings on 1.7B (see docs/findings.md): every case-swap works and is specific
(random control ~0). The cases are distinct in the causal representation --
including language vs demonym, which produce the SAME output word (Italian) yet
are separate case-directions: the operation is distinct from its realization. In
J-space the cases are LESS separable and reorganize by output form -- language and
demonym are pulled together -- so the J-space is where the surface-form SYNCRETISM
shows, not where the cases are cleaner. (Syncretism is a form-level phenomenon, as
in Latin.)

    python scripts/operator_paradigm.py 1.7b
    python scripts/operator_paradigm.py 8b --int8
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics as st
from pathlib import Path

import pandas as pd
import torch

from _common import band_layers, first_token, load_model, resolve_tag
from jlens import ActivationRecorder, JacobianLens

ROOT = Path(__file__).resolve().parent.parent


def _load_relations():
    d = json.loads((ROOT / "data" / "relations.json").read_text())
    tmpl = d["meta"]["template"]
    cases = d["meta"]["cases"]          # case -> phrase in the template
    answers = d["answers"]              # country -> {case: answer}
    return tmpl, cases, answers


@torch.no_grad()
def _resid(model, ws, prompt):
    ids = model.encode(prompt, max_length=64)
    with ActivationRecorder(model.layers, at=ws) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0, -1].float() for l in ws}


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


@torch.no_grad()
def _readout(model, lens, ws, prompt, use_j):
    """mean over workspace layers of the lens logits at the last position."""
    ll, _, _ = lens.apply(model, prompt, layers=lens.source_layers,
                          positions=[-1], use_jacobian=use_j)
    return torch.stack([ll[l][0] for l in ws]).mean(0).float()


def _case_dirs_resid(model, ws, tmpl, cases, answers):
    countries, keys = list(answers), list(cases)
    R = {(c, k): _resid(model, ws, tmpl.format(rel=cases[k], country=c))
         for c in countries for k in keys}
    v = {k: {l: torch.zeros(model.d_model, device=next(iter(R.values()))[ws[0]].device)
             for l in ws} for k in keys}
    for c in countries:
        for l in ws:
            mean_l = sum(R[(c, k)][l] for k in keys) / len(keys)
            for k in keys:
                v[k][l] += (R[(c, k)][l] - mean_l) / len(countries)
    return v


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


def measure_swaps(model, ws, tmpl, cases, answers, tok, seed=0):
    v = _case_dirs_resid(model, ws, tmpl, cases, answers)
    countries, keys = list(answers), list(cases)
    g = torch.Generator().manual_seed(seed)
    dev = v[keys[0]][ws[0]].device
    rows = []
    swap_pairs = [(a, b) for a in keys for b in keys if a != b]
    for frm, to in swap_pairs:
        dv = {l: 4.0 * (v[to][l] - v[frm][l]) for l in ws}
        norm = torch.cat([dv[l] for l in ws]).norm()
        rv = {l: torch.randn(model.d_model, generator=g).to(dev) for l in ws}
        rn = torch.cat([rv[l] for l in ws]).norm()
        rv = {l: rv[l] / rn * norm for l in ws}
        clean, swap, rand = [], [], []
        for c in countries:
            p = tmpl.format(rel=cases[frm], country=c)
            L0, Ls, Lr = (_final_logits(model, p), _final_logits(model, p, dv),
                          _final_logits(model, p, rv))
            af = first_token(tok, answers[c][frm])
            at = first_token(tok, answers[c][to])
            clean.append(float(L0[at] - L0[af]))
            swap.append(float(Ls[at] - Ls[af]))
            rand.append(float(Lr[at] - Lr[af]))
        rows.append({"from": frm, "to": to, "clean": st.mean(clean),
                     "swap": st.mean(swap), "random": st.mean(rand)})
    return pd.DataFrame(rows)


def measure_geometry(model, lens, ws, tmpl, cases, answers, use_j):
    countries, keys = list(answers), list(cases)
    R = {(c, k): _readout(model, lens, ws, tmpl.format(rel=cases[k], country=c), use_j)
         for c in countries for k in keys}
    v = {k: 0 for k in keys}
    for c in countries:
        mean_c = sum(R[(c, k)] for k in keys) / len(keys)
        for k in keys:
            v[k] = v[k] + (R[(c, k)] - mean_c) / len(countries)
    offs = [(a, b, _cos(v[a], v[b])) for a, b in itertools.combinations(keys, 2)]
    return v, offs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    tmpl, cases, answers = _load_relations()

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    ws = band_layers(lens.source_layers, model.n_layers)["workspace"]

    print(f"\n(A) causal case-swap  [{tag}]  clean<0 (from wins); working swap>0; random~0")
    df = measure_swaps(model, ws, tmpl, cases, answers, tok, seed=args.seed)
    print(df.to_string(index=False, float_format=lambda x: f"{x:+.2f}"))
    print(f"  swaps that flipped sign: {(df['swap'] > 0).sum()}/{len(df)}; "
          f"random that flipped: {(df['random'] > 0).sum()}/{len(df)}")

    print("\n(B) paradigm geometry: mean |off-diagonal cosine| (lower = cases more distinct)")
    for name, uj in (("J-space (unembed(J h))", True), ("logit-space (unembed(h))", False)):
        _, offs = measure_geometry(model, lens, ws, tmpl, cases, answers, uj)
        mabs = st.mean(abs(o[2]) for o in offs)
        syn = max(offs, key=lambda o: o[2])
        print(f"  {name:<26} mean|off|={mabs:.3f}   most-syncretic: "
              f"{syn[0]}~{syn[1]} cos={syn[2]:+.2f}")

    out = ROOT / "results" / "ablation" / f"{tag}_operator_swap.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
