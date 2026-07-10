#!/usr/bin/env python
"""Domain-general core for the operator/operand experiments.

An OPERATION applied to OPERANDS is rendered into a prompt, and we ask whether the
OPERATOR is a manipulable direction separable from the operand. This module holds
the shared machinery so the same code runs on relational facts, arithmetic, and
logic; `operator_paradigm.py` and `operator_factorize.py` are thin CLIs over it.

Unified dataset schema  data/<domain>.json:

    {
      "meta": {
        "domain": "relations",
        "template": "The {op} of {a} is",          # {op} + operand slots
        "operators": {"currency": "currency", ...}, # operator key -> template phrase
        "operand_slots": ["a"],                     # template fields filled per operand
        "desinence_pair": ["language", "demonym"]   # optional (relations only)
      },
      "items": {                                    # operand key -> args + answers
        "Italy": {"args": {"a": "Italy"},
                  "answers": {"currency": "euro", "capital": "Rome", ...}},
        ...
      }
    }

`render(operand, operator)` fills the template; the efficacy metric is a logit
difference of the two answers' first tokens, so multi-token article prefixes
("the euro" vs "Rome") do not contaminate. A first-token-distinguishability guard
runs on load, because in Qwen3 BPE two answers can collide at token 0 (e.g. some
two-digit numbers) and silently zero the metric.
"""

from __future__ import annotations

import itertools
import json
import statistics as st
from dataclasses import dataclass
from pathlib import Path

import torch

from _common import first_token, first_token_distinguishable, single_leading_space_token
from jlens import ActivationRecorder

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Domain:
    name: str
    template: str
    operators: dict           # op_key -> template phrase
    operand_slots: list       # template fields filled per operand
    items: dict               # operand_key -> {"args": {...}, "answers": {op: ans}}
    desinence_pair: list | None
    answer_leading_space: bool = True  # False for digits (Qwen3 splits " 3" -> [space,3])

    @property
    def op_keys(self):
        return list(self.operators)

    @property
    def operand_keys(self):
        return list(self.items)

    def render(self, operand_key, op_key):
        return self.template.format(op=self.operators[op_key],
                                    **self.items[operand_key]["args"])

    def answer(self, operand_key, op_key):
        return self.items[operand_key]["answers"][op_key]

    def answer_tok(self, tok, operand_key, op_key) -> int:
        """Token id used to score the answer. For digits we score the bare digit
        token, since Qwen3 tokenizes ' 3' as [space, 3] and the leading space
        collides across all digits."""
        ans = str(self.answer(operand_key, op_key))
        prefix = " " if self.answer_leading_space else ""
        return tok.encode(prefix + ans, add_special_tokens=False)[0]


def load_domain(name: str) -> Domain:
    d = json.loads((ROOT / "data" / f"{name}.json").read_text())
    m = d["meta"]
    return Domain(name=m.get("domain", name), template=m["template"],
                  operators=m["operators"], operand_slots=m.get("operand_slots", ["a"]),
                  items=d["items"], desinence_pair=m.get("desinence_pair"),
                  answer_leading_space=m.get("answer_leading_space", True))


def guard_tokenization(dom: Domain, tok) -> dict:
    """Report, per operator pair, on how many operands have first-token-distinct
    answers (only those carry swap signal for that pair), plus single-token coverage.

    Reports -- does NOT drop -- because some domains (comparison logic) have inherent
    answer collisions for specific operator pairs on specific operands; measure_swaps
    handles this per (from, to) pair. The desinence_pair (language/demonym) is a
    deliberate syncretism, flagged separately."""
    single = tot = 0
    pair_signal = {}
    for ka, kb in itertools.combinations(dom.op_keys, 2):
        n = sum(1 for o in dom.operand_keys
                if dom.answer_tok(tok, o, ka) != dom.answer_tok(tok, o, kb))
        pair_signal[(ka, kb)] = n
    for o in dom.operand_keys:
        for k in dom.op_keys:
            tot += 1
            if single_leading_space_token(tok, dom.answer(o, k)) is not None:
                single += 1
    return {"operands_total": len(dom.operand_keys),
            "single_token_frac": single / max(tot, 1),
            "pair_signal": pair_signal,
            "min_pair_signal": min(pair_signal.values()) if pair_signal else 0}


# --- residual / logit helpers (identical math to the original scripts) -------

@torch.no_grad()
def resid(model, layers, prompt, pos=-1):
    ids = model.encode(prompt, max_length=64)
    with ActivationRecorder(model.layers, at=layers) as rec:
        model.forward(ids)
    return {l: rec.activations[l][0, pos].float() for l in layers}


@torch.no_grad()
def final_logits(model, prompt, hook_vecs=None):
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
def readout(model, lens, ws, prompt, use_j):
    ll, _, _ = lens.apply(model, prompt, layers=lens.source_layers,
                          positions=[-1], use_jacobian=use_j)
    return torch.stack([ll[l][0] for l in ws]).mean(0).float()


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


# --- operator directions -----------------------------------------------------

def op_dirs(model, ws, dom: Domain, operands=None):
    """v(op)[layer] = mean over operands of (op's residual - operand's mean over ops).
    The function-vector construction, generalized. `operands` restricts the build
    set (for held-out generalization)."""
    ops = dom.op_keys
    operands = operands or dom.operand_keys
    R = {(o, k): resid(model, ws, dom.render(o, k)) for o in operands for k in ops}
    dev = next(iter(R.values()))[ws[0]].device
    v = {k: {l: torch.zeros(model.d_model, device=dev) for l in ws} for k in ops}
    for o in operands:
        for l in ws:
            mean_l = sum(R[(o, k)][l] for k in ops) / len(ops)
            for k in ops:
                v[k][l] += (R[(o, k)][l] - mean_l) / len(operands)
    return v


def measure_swaps(model, ws, dom: Domain, tok, seed=0, alpha=4.0, operands=None,
                  build_operands=None):
    """All-pairs operator swap: add alpha*(v[to]-v[from]) to a `from` prompt, read
    the logit shift toward the `to` answer, vs a matched-norm random control. If
    build_operands is set, v(op) is built from those and tested on `operands`
    (held-out generalization)."""
    import pandas as pd
    v = op_dirs(model, ws, dom, build_operands)
    ops = dom.op_keys
    test = operands or dom.operand_keys
    g = torch.Generator().manual_seed(seed)
    dev = v[ops[0]][ws[0]].device
    rows = []
    for frm, to in [(a, b) for a in ops for b in ops if a != b]:
        dv = {l: alpha * (v[to][l] - v[frm][l]) for l in ws}
        norm = torch.cat([dv[l] for l in ws]).norm()
        rv = {l: torch.randn(model.d_model, generator=g).to(dev) for l in ws}
        rn = torch.cat([rv[l] for l in ws]).norm()
        rv = {l: rv[l] / rn * norm for l in ws}
        clean, swap, rand = [], [], []
        for o in test:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue  # this operand can't distinguish from/to (e.g. AND=OR on TT)
            p = dom.render(o, frm)
            L0, Ls, Lr = (final_logits(model, p), final_logits(model, p, dv),
                          final_logits(model, p, rv))
            clean.append(float(L0[at] - L0[af]))
            swap.append(float(Ls[at] - Ls[af]))
            rand.append(float(Lr[at] - Lr[af]))
        n = len(clean)
        rows.append({"from": frm, "to": to, "n": n,
                     "clean": st.mean(clean) if n else float("nan"),
                     "swap": st.mean(swap) if n else float("nan"),
                     "random": st.mean(rand) if n else float("nan")})
    return pd.DataFrame(rows)


def measure_geometry(model, lens, ws, dom: Domain, use_j):
    ops = dom.op_keys
    R = {(o, k): readout(model, lens, ws, dom.render(o, k), use_j)
         for o in dom.operand_keys for k in ops}
    v = {k: 0 for k in ops}
    for o in dom.operand_keys:
        mean_o = sum(R[(o, k)] for k in ops) / len(ops)
        for k in ops:
            v[k] = v[k] + (R[(o, k)] - mean_o) / len(dom.operand_keys)
    offs = [(a, b, _cos(v[a], v[b])) for a, b in itertools.combinations(ops, 2)]
    return v, offs


def factorize(model, lens, layer, pos, dom: Domain, use_j):
    """H[operand, operator] = mu + operand + operator + interaction. Variance shares
    and principal angles between the operand- and operator-subspaces."""
    ops, operands = dom.op_keys, dom.operand_keys
    H = {(o, k): resid(model, [layer], dom.render(o, k), pos)[layer]
         if not use_j else lens.transport(
             resid(model, [layer], dom.render(o, k), pos)[layer][None], layer)[0]
         for o in operands for k in ops}
    mu = torch.stack([H[(o, k)] for o in operands for k in ops]).mean(0)
    stem = {o: torch.stack([H[(o, k)] for k in ops]).mean(0) - mu for o in operands}
    case = {k: torch.stack([H[(o, k)] for o in operands]).mean(0) - mu for k in ops}
    tot = sum(float((H[(o, k)] - mu).pow(2).sum()) for o in operands for k in ops)
    v_stem = len(ops) * sum(float(stem[o].pow(2).sum()) for o in operands)
    v_case = len(operands) * sum(float(case[k].pow(2).sum()) for k in ops)
    inter = sum(float((H[(o, k)] - mu - stem[o] - case[k]).pow(2).sum())
                for o in operands for k in ops)
    S = torch.stack([stem[o] for o in operands]).t()
    T = torch.stack([case[k] for k in ops]).t()
    Qs, _ = torch.linalg.qr(S)
    Qt, _ = torch.linalg.qr(T)
    sv = torch.linalg.svdvals(Qs.t() @ Qt).clamp(-1, 1)
    angles = [round(float(a), 1) for a in torch.rad2deg(torch.arccos(sv))]
    return {"stem": v_stem / tot, "case": v_case / tot,
            "interaction": inter / tot, "angles": angles}


def held_out_generalization(model, ws, dom: Domain, tok, seed=0, alpha=4.0):
    """Build v(op) from half the operands, test the swap on the held-out half. If it
    generalizes, the operator is a real ending, not interpolation among examples."""
    ops = dom.operand_keys
    half = len(ops) // 2
    build, test = ops[:half], ops[half:]
    df = measure_swaps(model, ws, dom, tok, seed=seed, alpha=alpha,
                       operands=test, build_operands=build)
    return build, test, df


def pure_desinence(model, ws, dom: Domain, tok, alpha=6.0):
    """Isolate an exponent-free case marker from a syncretic pair (two operators
    that emit the same word), then test its causal efficacy. Relations only."""
    if not dom.desinence_pair:
        return None
    ka, kb = dom.desinence_pair
    dev = model._lm_head.weight.device
    same = [o for o in dom.operand_keys if dom.answer(o, ka) == dom.answer(o, kb)]
    if not same:
        return None
    v = {l: torch.zeros(model.d_model, device=dev) for l in ws}
    for o in same:
        ra = resid(model, ws, dom.render(o, ka))
        rb = resid(model, ws, dom.render(o, kb))
        for l in ws:
            v[l] += (ra[l] - rb[l]) / len(same)
    # efficacy against a third operator (the first one that isn't the pair)
    other = next(k for k in dom.op_keys if k not in (ka, kb))
    shifts = []
    for o in dom.operand_keys[:8]:
        p = dom.render(o, other)
        L0 = final_logits(model, p)
        Lv = final_logits(model, p, {l: alpha * v[l] for l in ws})
        a_pair, a_other = dom.answer_tok(tok, o, ka), dom.answer_tok(tok, o, other)
        shifts.append((float(L0[a_pair] - L0[a_other]), float(Lv[a_pair] - Lv[a_other])))
    return {"pair": (ka, kb), "other": other, "n_same": len(same),
            "clean": st.mean(s[0] for s in shifts), "desinence": st.mean(s[1] for s in shifts)}
