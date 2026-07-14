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
def hooked_logits(model, prompt, add=None, add_positions=None, patch=None):
    """Final-position logits under a generalized intervention. `add` ({layer: vec})
    is added at every position when `add_positions` is None (identical to
    `final_logits`), else only at the given prompt indices (negatives resolved
    against the prompt length). `patch` ({layer: vec}) REPLACES the residual at the
    last prompt position (the query token), the op_minimal patch convention."""
    ids = model.encode(prompt, max_length=64)
    n_prompt = ids.shape[1]
    pos = None if add_positions is None else [p % n_prompt for p in add_positions]
    final = model.n_layers - 1
    handles = []
    for l, v in (add or {}).items():
        def mk_add(vec):
            def h(m, i, o):
                a = o[0] if isinstance(o, tuple) else o
                if pos is None:
                    a = a + vec.to(a.dtype)
                else:
                    a = a.clone()
                    for p_ in pos:
                        a[:, p_, :] = a[:, p_, :] + vec.to(a.dtype)
                return (a, *o[1:]) if isinstance(o, tuple) else a
            return h
        handles.append(model.layers[l].register_forward_hook(mk_add(v)))
    for l, v in (patch or {}).items():
        def mk_patch(vec):
            def h(m, i, o):
                a = o[0] if isinstance(o, tuple) else o
                a = a.clone()
                a[:, n_prompt - 1, :] = vec.to(a.dtype)
                return (a, *o[1:]) if isinstance(o, tuple) else a
            return h
        handles.append(model.layers[l].register_forward_hook(mk_patch(v)))
    with ActivationRecorder(model.layers, at=[final]) as rec:
        model.forward(ids)
        h = rec.activations[final][0, -1]
    for hd in handles:
        hd.remove()
    return model.unembed(h[None])[0]


@torch.no_grad()
def metric_bundle(model, dom: Domain, tok, o, frm, to, add=None, add_positions=None,
                  patch=None, tpl=0, clean_logits=None, k=3):
    """The full per-cell metric set for one (pair, operand, condition): margin
    (logit(to)-logit(from), the paper's convention), target rank before/after
    (0 = top-1), top-1 hit, normalized margin shift (dimensionless, comparable
    across models with different logit scales), on-task KL(clean || intervened) at
    the query position, and greedy k-token exact match of the target answer.
    One hooked forward + one clean forward (if not precomputed) + one greedy."""
    import torch.nn.functional as F
    import op_minimal  # lazy: op_minimal imports op_core
    p = dom.render(o, frm, tpl)
    af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
    L0 = clean_logits if clean_logits is not None else final_logits(model, p)
    L = hooked_logits(model, p, add=add, add_positions=add_positions, patch=patch)
    m0, m1 = float(L0[at] - L0[af]), float(L[at] - L[af])
    lp0, lp1 = F.log_softmax(L0.float(), -1), F.log_softmax(L.float(), -1)
    text = op_minimal.greedy(model, p, k=k, add=add, patch=patch,
                             add_positions=add_positions)
    return {"margin_clean": m0, "margin": m1, "delta_margin": m1 - m0,
            "norm_margin": (m1 - m0) / (abs(m0) + 1e-9),
            "rank_to_clean": int((L0 > L0[at]).sum()),
            "rank_to": int((L > L[at]).sum()),
            "top1": bool(int(L.argmax()) == at),
            "kl_ontask": float((lp0.exp() * (lp0 - lp1)).sum()),
            "exact_match": op_minimal.hit(text, str(dom.answer(o, to))),
            "text": text}  # the decoded continuation, so callers can score it
                           # against other answers without a second greedy pass


@torch.no_grad()
def readout(model, lens, ws, prompt, use_j):
    ll, _, _ = lens.apply(model, prompt, layers=lens.source_layers,
                          positions=[-1], use_jacobian=use_j)
    return torch.stack([ll[l][0] for l in ws]).mean(0).float()


def _cos(a, b):
    return float((a @ b) / (a.norm() * b.norm() + 1e-9))


# --- operator directions -----------------------------------------------------

def op_resids(model, ws, dom: Domain, operands=None, templates=None):
    """The residual grid R[(operand, op, template)] -> {layer: vec} that `op_dirs`
    averages. Exposed so callers that redraw label assignments many times (the
    permuted-relations null) reuse one grid instead of re-running the model."""
    ops = dom.op_keys
    operands = operands or dom.operand_keys
    tpls = templates or [0]
    return {(o, k, t): resid(model, ws, dom.render(o, k, t))
            for o in operands for k in ops for t in tpls}


def dirs_from_resids(R, ws, ops, operands, tpls, d_model, relabel=None):
    """The averaging step of `op_dirs` over a precomputed grid. `relabel`, if set,
    maps (operand, template) -> {op_label: op_whose_residual_to_use} and implements
    the label-permutation nulls; the cell mean is permutation-invariant, so only
    the numerator assignment changes."""
    dev = next(iter(R.values()))[ws[0]].device
    v = {k: {l: torch.zeros(d_model, device=dev) for l in ws} for k in ops}
    cells = len(operands) * len(tpls)
    for o in operands:
        for t in tpls:
            lab = relabel[(o, t)] if relabel else None
            for l in ws:
                mean_l = sum(R[(o, k, t)][l] for k in ops) / len(ops)
                for k in ops:
                    src = lab[k] if lab else k
                    v[k][l] += (R[(o, src, t)][l] - mean_l) / cells
    return v


def op_dirs(model, ws, dom: Domain, operands=None, templates=None):
    """v(op)[layer] = mean over (operand, template) cells of (op's residual - the
    cell's mean over ops). The function-vector construction, generalized. `operands`
    restricts the build set (held-out generalization); `templates` (list of indices
    into dom.templates, default [0]) restricts the paraphrase frames used to build."""
    operands = operands or dom.operand_keys
    tpls = templates or [0]
    R = op_resids(model, ws, dom, operands, tpls)
    return dirs_from_resids(R, ws, dom.op_keys, operands, tpls, model.d_model)


def permuted_op_dirs(model, ws, dom: Domain, seed=0, per_operand=True, R=None,
                     templates=None):
    """Directions built with PERMUTED operator labels (the decisive null: keeps
    every statistical property of the extraction -- same residuals, same averaging,
    same norms -- while destroying the intended semantics). `per_operand=True`
    draws an independent permutation per (operand, template) cell, which destroys
    label coherence entirely; False applies one global permutation, which keeps the
    directions coherent but mis-assigned (v'(k) = v(pi(k))). Pass a precomputed
    `R` from `op_resids` to redraw cheaply across seeds."""
    import random as _random
    ops = dom.op_keys
    operands = dom.operand_keys
    tpls = templates or [0]
    if R is None:
        R = op_resids(model, ws, dom, operands, tpls)
    rng = _random.Random(seed)
    def draw():
        perm = ops[:]
        rng.shuffle(perm)
        return dict(zip(ops, perm))
    if per_operand:
        relabel = {(o, t): draw() for o in operands for t in tpls}
    else:
        one = draw()
        relabel = {(o, t): one for o in operands for t in tpls}
    return dirs_from_resids(R, ws, ops, operands, tpls, model.d_model, relabel)


def operator_subspace_basis(dirs, ws):
    """Per-layer orthonormal basis Q[l] (d_model x r) of the span of the operator
    directions (already mean-centered across ops by construction; r <= n_ops-1).
    Substrate for the random-within-operator-subspace null."""
    ops = list(dirs)
    Q = {}
    for l in ws:
        M = torch.stack([dirs[k][l] for k in ops], dim=1)  # [d, n_ops]
        M = M - M.mean(1, keepdim=True)
        Qf, Rf = torch.linalg.qr(M)
        d = Rf.diagonal().abs()
        Q[l] = Qf[:, d > 1e-6 * max(float(d.max()), 1e-30)]
    return Q


def random_in_subspace(Q, norms, gen):
    """{layer: random vector inside span(Q[l])}, per-layer norm matched to
    `norms[l]` (typically the injected dv's norms) -- a far more competitive null
    than the full-space Gaussian, since it lives in the operator subspace."""
    out = {}
    for l, Ql in Q.items():
        c = torch.randn(Ql.shape[1], generator=gen).to(Ql.device, Ql.dtype)
        u = Ql @ c
        out[l] = u / (u.norm() + 1e-9) * norms[l]
    return out


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


@torch.no_grad()
def factorize_components(model, ws, dom: Domain, pos=-1, H=None):
    """Per-layer component VECTORS of the two-way factorization at `pos` -- the
    same math as `factorize` (which reports variance shares) but retaining the
    parts, so partial reconstructions can be patched back into the model
    (op_patch_decomp). For every layer l in ws:

        H[(o,k)][l] == mu[l] + stem[l][o] + case[l][k] + inter[l][(o,k)]   exactly,

    because inter is defined as the cell residual. One forward per (operand, op)
    cell; all ws layers are recorded in that single pass. Pass a precomputed
    grid `H` ({(operand, op): {layer: vec}}, e.g. derived from `op_resids`) to
    skip the forwards and share one grid with `dirs_from_resids` callers."""
    ops, operands = dom.op_keys, dom.operand_keys
    if H is None:
        H = {(o, k): resid(model, ws, dom.render(o, k), pos)
             for o in operands for k in ops}
    mu, stem, case, inter = {}, {}, {}, {}
    for l in ws:
        mu[l] = torch.stack([H[(o, k)][l] for o in operands for k in ops]).mean(0)
        stem[l] = {o: torch.stack([H[(o, k)][l] for k in ops]).mean(0) - mu[l]
                   for o in operands}
        case[l] = {k: torch.stack([H[(o, k)][l] for o in operands]).mean(0) - mu[l]
                   for k in ops}
        inter[l] = {(o, k): H[(o, k)][l] - mu[l] - stem[l][o] - case[l][k]
                    for o in operands for k in ops}
    return {"mu": mu, "stem": stem, "case": case, "inter": inter}


def held_out_generalization(model, ws, dom: Domain, tok, seed=0, alpha=4.0,
                            split_seed=None, build=None, test=None):
    """Build v(op) from half the operands, test the swap on the held-out half. If it
    generalizes, the operator is a real ending, not interpolation among examples.

    The default split is the paper's original (insertion order). `split_seed` draws
    a shuffled half/half partition instead (multi-partition robustness); `build`/
    `test` override both (e.g. leave-one-operand-out)."""
    ops = dom.operand_keys
    if build is None or test is None:
        pool = ops[:]
        if split_seed is not None:
            import random as _random
            _random.Random(split_seed).shuffle(pool)
        half = len(pool) // 2
        build, test = pool[:half], pool[half:]
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
