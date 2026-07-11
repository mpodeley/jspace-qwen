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
    templates: list | None = None  # paraphrase frames; templates[0] == template (canonical)

    def __post_init__(self):
        if not self.templates:
            self.templates = [self.template]

    @property
    def op_keys(self):
        return list(self.operators)

    @property
    def operand_keys(self):
        return list(self.items)

    def render(self, operand_key, op_key, tpl: int = 0):
        args = self.items[operand_key]["args"]
        phrase = self.operators[op_key]
        if "{" in phrase:  # operand-bearing phrase, e.g. "money used in {a}"
            phrase = phrase.format(**args)
        return self.templates[tpl].format(op=phrase, **args)

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
                  answer_leading_space=m.get("answer_leading_space", True),
                  templates=m.get("templates"))


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

def op_dirs(model, ws, dom: Domain, operands=None, templates=None):
    """v(op)[layer] = mean over (operand, template) cells of (op's residual - the
    cell's mean over ops). The function-vector construction, generalized. `operands`
    restricts the build set (held-out generalization); `templates` (list of indices
    into dom.templates, default [0]) restricts the paraphrase frames used to build."""
    ops = dom.op_keys
    operands = operands or dom.operand_keys
    tpls = templates or [0]
    R = {(o, k, t): resid(model, ws, dom.render(o, k, t))
         for o in operands for k in ops for t in tpls}
    dev = next(iter(R.values()))[ws[0]].device
    v = {k: {l: torch.zeros(model.d_model, device=dev) for l in ws} for k in ops}
    cells = len(operands) * len(tpls)
    for o in operands:
        for t in tpls:
            for l in ws:
                mean_l = sum(R[(o, k, t)][l] for k in ops) / len(ops)
                for k in ops:
                    v[k][l] += (R[(o, k, t)][l] - mean_l) / cells
    return v


def measure_swaps_long(model, ws, dom: Domain, tok, seed=0, alpha=4.0, operands=None,
                       build_operands=None, templates=None, build_templates=None,
                       dirs=None):
    """All-pairs operator swap, one row PER (pair, operand, template): add
    alpha*(v[to]-v[frm]) to a `frm` prompt and read the logit shift toward the `to`
    answer, vs a matched-norm random control. Long form is the substrate for
    distributions and cluster-bootstrap CIs; `measure_swaps` aggregates it to
    per-pair means. If build_operands is set, v(op) is built from those and tested
    on `operands` (held-out-operand generalization). If build_templates differs from
    `templates`, v(op) is built on one paraphrase frame and tested on another
    (cross-template transfer). Pass precomputed `dirs` (from op_dirs) to reuse the
    same operator directions across several test conditions."""
    import pandas as pd
    tpls = templates or [0]
    v = dirs if dirs is not None else op_dirs(
        model, ws, dom, build_operands, build_templates or tpls)
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
        for o in test:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue  # this operand can't distinguish from/to (e.g. AND=OR on TT)
            for t in tpls:
                p = dom.render(o, frm, t)
                L0, Ls, Lr = (final_logits(model, p), final_logits(model, p, dv),
                              final_logits(model, p, rv))
                rows.append({"from": frm, "to": to, "operand": o, "template": t,
                             "clean": float(L0[at] - L0[af]),
                             "swap": float(Ls[at] - Ls[af]),
                             "random": float(Lr[at] - Lr[af])})
    return pd.DataFrame(rows, columns=["from", "to", "operand", "template",
                                       "clean", "swap", "random"])


def measure_swaps(model, ws, dom: Domain, tok, seed=0, alpha=4.0, operands=None,
                  build_operands=None, long_df=None):
    """Per-pair means of `measure_swaps_long` in the legacy wide schema
    (from, to, n, clean, swap, random). Pass `long_df` to aggregate an existing
    long-form frame without re-running the model."""
    import pandas as pd
    ldf = long_df if long_df is not None else measure_swaps_long(
        model, ws, dom, tok, seed=seed, alpha=alpha, operands=operands,
        build_operands=build_operands)
    rows = []
    for frm, to in [(a, b) for a in dom.op_keys for b in dom.op_keys if a != b]:
        sub = ldf[(ldf["from"] == frm) & (ldf["to"] == to)]
        n = len(sub)
        rows.append({"from": frm, "to": to, "n": n,
                     "clean": sub["clean"].mean() if n else float("nan"),
                     "swap": sub["swap"].mean() if n else float("nan"),
                     "random": sub["random"].mean() if n else float("nan")})
    return pd.DataFrame(rows)


def bootstrap_pair_ci(long_df, n_boot=10_000, seed=0, ci=95):
    """Percentile cluster-bootstrap CI per ordered pair, resampling OPERANDS with
    replacement within the pair. The operand is the independent unit at this level;
    the 20 ordered pairs share 5 operator directions and are NOT independent of one
    another -- paradigm-level claims must use `bootstrap_family_ci`."""
    import numpy as np
    import pandas as pd
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    rows = []
    for (frm, to), sub in long_df.groupby(["from", "to"], sort=False):
        swap = sub["swap"].to_numpy()
        contrast = (sub["swap"] - sub["random"]).to_numpy()
        n = len(sub)
        idx = rng.integers(0, n, size=(n_boot, n))
        bs, bc = swap[idx].mean(axis=1), contrast[idx].mean(axis=1)
        rows.append({"from": frm, "to": to, "n": n,
                     "swap_mean": float(swap.mean()),
                     "swap_lo": float(np.percentile(bs, lo_q)),
                     "swap_hi": float(np.percentile(bs, hi_q)),
                     "contrast_mean": float(contrast.mean()),
                     "contrast_lo": float(np.percentile(bc, lo_q)),
                     "contrast_hi": float(np.percentile(bc, hi_q))})
    return pd.DataFrame(rows)


def bootstrap_family_ci(long_df, n_boot=10_000, seed=0, ci=95):
    """Paradigm-level cluster bootstrap honoring the dependence structure: the
    ordered pairs reuse the same operator directions, so OPERATORS (not pairs) are
    the top-level unit, operands the nested unit. Each replicate resamples the
    operator set with replacement (a pair (a,b) enters with multiplicity
    count(a)*count(b) -- the dyadic node bootstrap), then resamples operands within
    each surviving pair. Returns percentile CIs for the weighted mean swap-random
    contrast and for the weighted fraction of pairs whose mean swap is positive."""
    import numpy as np
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    ops = list(dict.fromkeys(long_df["from"]))
    groups = {(f, t): (s["swap"].to_numpy(), (s["swap"] - s["random"]).to_numpy())
              for (f, t), s in long_df.groupby(["from", "to"], sort=False)}
    contrasts, flips = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        cnt = {k: 0 for k in ops}
        for k in rng.choice(ops, size=len(ops), replace=True):
            cnt[k] += 1
        tot_w = tot_c = tot_f = 0.0
        for (f, t), (swap, contrast) in groups.items():
            w = cnt[f] * cnt[t]
            if w == 0:
                continue
            idx = rng.integers(0, len(swap), size=len(swap))
            tot_w += w
            tot_c += w * contrast[idx].mean()
            tot_f += w * (swap[idx].mean() > 0)
        contrasts[b] = tot_c / tot_w if tot_w else np.nan
        flips[b] = tot_f / tot_w if tot_w else np.nan
    obs_c = float((long_df["swap"] - long_df["random"]).mean())
    wide = long_df.groupby(["from", "to"], sort=False)["swap"].mean()
    return {"n_operators": len(ops), "n_pairs": len(groups),
            "contrast_mean": obs_c,
            "contrast_lo": float(np.nanpercentile(contrasts, lo_q)),
            "contrast_hi": float(np.nanpercentile(contrasts, hi_q)),
            "flip_frac": float((wide > 0).mean()),
            "flip_lo": float(np.nanpercentile(flips, lo_q)),
            "flip_hi": float(np.nanpercentile(flips, hi_q))}


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
    ldf = measure_swaps_long(model, ws, dom, tok, seed=seed, alpha=alpha,
                             operands=test, build_operands=build)
    df = measure_swaps(model, ws, dom, tok, long_df=ldf)
    return build, test, df, ldf


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
