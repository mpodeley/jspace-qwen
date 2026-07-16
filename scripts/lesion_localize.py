#!/usr/bin/env python
"""Functional localizers: which units belong to which network?

Neuroscience defines a region by a LOCALIZER -- a contrast between conditions
that isolates a function -- and then asks whether the units it selects are
spatially clustered (an "area") or scattered (a "network"). We borrow the method,
not the stimuli: every contrast below is built from this repo's own data, and
every claim is about the model.

Three localizers:

  factorial   The paper's balanced 12x5 grid IS a localizer. For every scalar
              unit (each MLP neuron, each attention head) decompose its query-
              position activation exactly as the paper decomposes the residual:
                  a_u(o,k) = mu_u + stem_u(o) + case_u(k) + inter_u(o,k)
              The operand/operator variance of a unit is the direct analog of an
              fMRI voxel's contrast effect size. Ranking by (operator variance -
              operand variance) gives the OPERATOR network; the reverse gives the
              OPERAND network. Null: the same statistic under permuted relation
              labels (identical distributional properties, destroyed semantics).

  induction   The field's most universal "organ" -- our positive control with a
              known answer. Repeated random-token sequences (A B C ... A B C) vs
              non-repeating ones; units selective for the repeated condition at
              the second-half positions should be the induction heads, and
              lesioning them must destroy in-context copying while sparing
              ordinary language modelling. If this fails, the pipeline is broken.

  text_symbolic  A Fedorenko-flavored contrast with OUR OWN stimuli: natural
              text (WikiText) vs symbolic tasks (arithmetic + logic prompts),
              length-matched and read at the same position, so the contrast is
              not a length artifact.

Outputs per-unit selectivity plus the region geometry (depth profile,
concentration vs a random-unit null, segregation between networks).

    python scripts/lesion_localize.py 1.7b
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy
import pandas as pd
import torch
import torch.nn.functional as F

from _common import BANDS, depth_percent, get_corpus, load_model, resolve_tag
from jlens import ActivationRecorder
from lesion_core import UnitRecorder, unit_dims
import op_core

ROOT = Path(__file__).resolve().parent.parent


# --- stimuli ------------------------------------------------------------------

def induction_stimuli(model, n=24, seq_len=32, seed=0):
    """Pairs of token sequences: `rep` = X X (a random sequence, repeated) and
    `ctl` = X Y (two independent random sequences). Induction can predict the
    second half of `rep` and cannot predict the second half of `ctl`; everything
    else about the two is matched (length, token distribution, position)."""
    g = torch.Generator().manual_seed(seed)
    V = model.tokenizer.vocab_size
    # sample from a mid-frequency band of the vocab: avoids specials and the
    # byte-fallback tail, both of which have idiosyncratic activations
    lo, hi = 1000, min(V, 30000)
    out = []
    for _ in range(n):
        x = torch.randint(lo, hi, (seq_len,), generator=g)
        y = torch.randint(lo, hi, (seq_len,), generator=g)
        out.append({"rep": torch.cat([x, x])[None].to(model.input_device),
                    "ctl": torch.cat([x, y])[None].to(model.input_device),
                    "half": seq_len})
    return out


@torch.no_grad()
def induction_score(model, stim) -> dict:
    """Mean log-prob of the tokens in the SECOND half. On `rep` this is the
    in-context copying ability (high only if induction works); on `ctl` it is the
    floor (the tokens are unpredictable). The gap is the induction score."""
    final = model.n_layers - 1
    got = {"rep": [], "ctl": []}
    for s in stim:
        for kind in ("rep", "ctl"):
            ids = s[kind]
            with ActivationRecorder(model.layers, at=[final]) as rec:
                model.forward(ids)
                h = rec.activations[final][0]
            lp = F.log_softmax(model.unembed(h.float()), -1)
            half = s["half"]
            tgt = ids[0, half:]                      # predicted from position half-1 on
            pred = lp[half - 1:-1]
            got[kind].append(float(pred.gather(-1, tgt[:, None]).mean()))
    rep = sum(got["rep"]) / len(got["rep"])
    ctl = sum(got["ctl"]) / len(got["ctl"])
    return {"logprob_repeat": rep, "logprob_control": ctl, "induction_score": rep - ctl}


def text_symbolic_stimuli(model, n=48, seed=0):
    """Length-matched natural text vs symbolic-task prompts. Symbolic prompts are
    short; WikiText chunks are long, so we truncate each to the symbolic median
    token length and read both at the final position -- otherwise the 'contrast'
    would mostly track sequence length."""
    sym = []
    for domain in ("arithmetic", "logic"):
        dom = op_core.load_domain(domain)
        for o in dom.operand_keys:
            for k in dom.op_keys:
                sym.append(dom.render(o, k))
    lens = sorted(model.encode(p, max_length=64).shape[1] for p in sym)
    L = lens[len(lens) // 2]
    txt = []
    for p in get_corpus(n * 2)[:n * 2]:
        ids = model.encode(p, max_length=512)
        if ids.shape[1] < L:
            continue
        txt.append(model.tokenizer.decode(ids[0, :L]))
        if len(txt) >= n:
            break
    return txt, sym[:n] if len(sym) > n else sym, L


# --- localizers ---------------------------------------------------------------

@torch.no_grad()
def grid_activations(model, dom, dims, pos=-1):
    """Per-unit activations over the balanced grid: {(operand, op): (heads, neurons)}
    with heads [n_layers, n_heads] and neurons [n_layers, d_mlp]."""
    out = {}
    for o in dom.operand_keys:
        for k in dom.op_keys:
            ids = model.encode(dom.render(o, k), max_length=64)
            with UnitRecorder(model, dims=dims) as rec:
                model.forward(ids)
            H = torch.zeros(dims.n_layers, dims.n_heads)
            N = torch.zeros(dims.n_layers, dims.d_mlp)
            for l in range(dims.n_layers):
                H[l] = rec.heads[l][pos]
                N[l] = rec.neurons[l][pos]
            out[(o, k)] = (H, N)
    return out


def factorial_selectivity(grid, operands, ops, relabel=None):
    """Two-way decomposition per scalar unit -- the same math as
    op_core.factorize, applied to each unit instead of the residual vector. Returns
    dict of [n_layers, width] tensors: ss_operator, ss_operand, ss_total,
    plus the shares. `relabel` maps operand -> {op_label: op_to_read} and
    implements the permuted-label null."""
    def cell(o, k, which):
        src = relabel[o][k] if relabel else k
        return grid[(o, src)][which]

    res = {}
    for which, name in ((0, "head"), (1, "neuron")):
        A = torch.stack([torch.stack([cell(o, k, which) for k in ops])
                         for o in operands])          # [n_op, n_k, L, W]
        mu = A.mean((0, 1))
        stem = A.mean(1) - mu                          # [n_op, L, W]
        case = A.mean(0) - mu                          # [n_k, L, W]
        ss_tot = (A - mu).pow(2).sum((0, 1))
        ss_stem = len(ops) * stem.pow(2).sum(0)
        ss_case = len(operands) * case.pow(2).sum(0)
        res[name] = {
            "ss_operand": ss_stem, "ss_operator": ss_case, "ss_total": ss_tot,
            "share_operand": ss_stem / (ss_tot + 1e-12),
            "share_operator": ss_case / (ss_tot + 1e-12),
            "magnitude": A.abs().mean((0, 1)),
        }
    return res


@torch.no_grad()
def contrast_selectivity(model, ids_A, ids_B, dims, pos_A=None, pos_B=None):
    """t-like contrast statistic per unit: (mean_A - mean_B) / pooled SD across
    prompts. `pos_*` is None (mean over positions), an int, or a slice."""
    def collect(idss, pos):
        Hs, Ns = [], []
        for ids in idss:
            with UnitRecorder(model, dims=dims) as rec:
                model.forward(ids)
            H = torch.zeros(dims.n_layers, dims.n_heads)
            N = torch.zeros(dims.n_layers, dims.d_mlp)
            for l in range(dims.n_layers):
                h, n = rec.heads[l], rec.neurons[l]
                if pos is None:
                    H[l], N[l] = h.mean(0), n.mean(0)
                elif isinstance(pos, slice):
                    H[l], N[l] = h[pos].mean(0), n[pos].mean(0)
                else:
                    H[l], N[l] = h[pos], n[pos]
            Hs.append(H)
            Ns.append(N)
        return torch.stack(Hs), torch.stack(Ns)

    HA, NA = collect(ids_A, pos_A)
    HB, NB = collect(ids_B, pos_B)
    out = {}
    for name, A, B in (("head", HA, HB), ("neuron", NA, NB)):
        sd = torch.sqrt((A.var(0) + B.var(0)) / 2) + 1e-6
        out[name] = {"contrast": (A.mean(0) - B.mean(0)) / sd,
                     "mean_A": A.mean(0), "mean_B": B.mean(0),
                     "magnitude": torch.cat([A, B]).abs().mean(0)}
    return out


# --- region geometry ----------------------------------------------------------

def region_stats(df, network, k, n_layers, seed=0):
    """Depth profile, concentration (vs a random-unit null) and band membership
    of the top-k units of `network`. Concentration = normalized entropy of the
    layer distribution: 1.0 = spread evenly across layers (a distributed
    NETWORK), 0.0 = all in one layer (an AREA)."""
    def norm_entropy(counts):
        p = counts.to_numpy() / max(counts.sum(), 1)
        nz = p[p > 0]
        return float(-(nz * numpy.log(nz)).sum() / math.log(n_layers))

    top = df.nlargest(k, network)
    prof = top.groupby("layer").size().reindex(range(n_layers), fill_value=0)
    ent = norm_entropy(prof)
    rnd = df.sample(k, random_state=seed)
    rprof = rnd.groupby("layer").size().reindex(range(n_layers), fill_value=0)
    rent = norm_entropy(rprof)
    bands = {b: float(((top["depth"] >= lo) & (top["depth"] <= hi)).mean())
             for b, (lo, hi) in BANDS.items()}
    return {
        "k": int(k),
        "layer_entropy": ent,                    # 1 = dispersed, 0 = one layer
        "random_layer_entropy": rent,            # the null (spread by construction)
        "top3_layer_share": float(prof.nlargest(3).sum() / max(prof.sum(), 1)),
        "center_of_mass_depth": float((top["depth"]).mean()),
        "bands": bands,
        "layer_profile": {int(l): int(c) for l, c in prof.items() if c},
    }


def jaccard(df, a, b, k):
    A = set(df.nlargest(k, a)["unit_id"])
    B = set(df.nlargest(k, b)["unit_id"])
    return len(A & B) / max(len(A | B), 1)


# --- main ---------------------------------------------------------------------

# operator_entity was exported as a column but missing from this list, so it had
# no region stats and no Jaccards: localized, never described. Note that
# `operand := -operator` and `operand_entity := -operator_entity` BY CONSTRUCTION
# (see the rows built above), so the within-position Jaccards of those two pairs
# are 0.0 as a TAUTOLOGY, not a finding -- no "the networks are segregated" claim
# can rest on them. The informative Jaccards are the cross-position ones, e.g.
# operand|operand_entity.
NETS = ["operator", "operand", "operator_entity", "operand_entity",
        "induction", "text_symbolic", "symbolic_text"]


def geometry_summary(df, base_ind, n_layers, k_neurons, seed=0):
    """Region geometry: areas or networks? Pure pandas over the localizer frame,
    so `--rescore` recomputes it without a model."""
    summary = {"induction_baseline": base_ind, "regions": {}, "segregation": {}}
    for kind in ("head", "neuron"):
        sub = df[df.kind == kind].reset_index(drop=True)
        k = min(k_neurons, len(sub) // 4) if kind == "neuron" else min(16, len(sub) // 4)
        summary["regions"][kind] = {
            net: region_stats(sub, net, k, n_layers, seed=seed) for net in NETS}
        summary["segregation"][kind] = {
            f"{a}|{b}": jaccard(sub, a, b, k)
            for ia, a in enumerate(NETS) for b in NETS[ia + 1:]}

    print(f"\n{'network':<16} {'kind':<7} {'entropy':>8} {'null':>6} "
          f"{'top3':>6} {'CoM depth':>10} {'workspace':>10}")
    for kind in ("head", "neuron"):
        for net in NETS:
            r = summary["regions"][kind][net]
            print(f"{net:<16} {kind:<7} {r['layer_entropy']:>8.2f} "
                  f"{r['random_layer_entropy']:>6.2f} {r['top3_layer_share']:>6.0%} "
                  f"{r['center_of_mass_depth']:>9.0f}% "
                  f"{r['bands']['workspace']:>10.0%}")
    print("\nsegregation (Jaccard of top-k):")
    for kind in ("head", "neuron"):
        pairs = summary["segregation"][kind]
        print(f"  {kind}: " + "  ".join(f"{k_}={v:.2f}" for k_, v in pairs.items()
                                        if v > 0.02) or f"  {kind}: all < 0.02")
    return summary


def rescore(tag, domain, k_neurons, seed=0):
    """Recompute the geometry summary from the existing localizer frame. No model.

    The parquet is gitignored, so unlike the battery's rescore this one only runs
    where the artifact was produced -- but it is what lets a new network be
    described without paying for the forward passes again.
    """
    out = ROOT / "results" / "lesion" / f"{tag}_{domain}_localizer.parquet"
    js = out.with_suffix("").with_name(out.stem + "_summary.json")
    if not (out.exists() and js.exists()):
        raise SystemExit(f"nothing to rescore: need {out} and {js}")
    df = pd.read_parquet(out)
    old = json.loads(js.read_text())
    n_layers = int(df.layer.max()) + 1
    summary = geometry_summary(df, old["induction_baseline"], n_layers, k_neurons,
                               seed=seed)
    summary["_meta"] = old["_meta"]
    js.write_text(json.dumps(summary, indent=2))
    print(f"\nrescored {js}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute the geometry summary from the existing frame")
    ap.add_argument("--k", type=int, default=512, help="network size for the stats")
    ap.add_argument("--null-seeds", type=int, default=5)
    ap.add_argument("--n-induction", type=int, default=24)
    ap.add_argument("--n-text", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)

    if args.rescore:                       # must precede load_model: the point is no GPU
        rescore(tag, args.domain, args.k, seed=args.seed)
        return

    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    dims = unit_dims(model)
    print(f"[{tag}] units: {dims.n_layers} layers x ({dims.n_heads} heads, "
          f"{dims.d_mlp} neurons) = {dims.n_head_units} heads, "
          f"{dims.n_neuron_units} neurons")

    # ---- factorial localizer (operator vs operand) ----
    # TWO read positions, to test a HYPOTHESIS (b65dbc7) that has since FAILED.
    # The hypothesis: the query token is where the operator is constructed, while
    # the operand only visits it -- attention re-imports the entity from the
    # entity token on every forward pass, so the operand is redundantly coded at
    # the query, which would explain why the query-position operand network is
    # unlesionable. It predicted the operand would be lesionable at the entity
    # token, where it is not redundant.
    # The prediction fails at both scales (02a19fb). Lesioning the entity-position
    # operand network degrades (1.7B: ppl x1.48, degraded 81.7%, other_operand
    # 1.7%) or produces the OPERATOR signature (8B: other_relation 10.0%). The
    # second read position is still worth localizing -- it is how we know the
    # failure is not a localizer artifact -- but "the entity token is where the
    # operand lives" is not something this repo has shown.
    print("factorial localizer (12x5 grid, query position)...")
    grid = grid_activations(model, dom, dims, pos=-1)
    fac = factorial_selectivity(grid, dom.operand_keys, dom.op_keys)
    print("factorial localizer (12x5 grid, ENTITY token)...")
    grid_e = grid_activations(model, dom, dims, pos=-2)
    fac_e = factorial_selectivity(grid_e, dom.operand_keys, dom.op_keys)

    import random as _random
    nulls = {"head": [], "neuron": []}
    for s in range(args.null_seeds):
        rng = _random.Random(s)
        relabel = {}
        for o in dom.operand_keys:
            perm = dom.op_keys[:]
            rng.shuffle(perm)
            relabel[o] = dict(zip(dom.op_keys, perm))
        nf = factorial_selectivity(grid, dom.operand_keys, dom.op_keys, relabel)
        for kind in ("head", "neuron"):
            nulls[kind].append(nf[kind]["ss_operator"] - nf[kind]["ss_operand"])

    # ---- induction localizer (the positive control) ----
    print("induction localizer (repeated vs non-repeating random sequences)...")
    stim = induction_stimuli(model, n=args.n_induction, seed=args.seed)
    half = stim[0]["half"]
    ind = contrast_selectivity(model, [s["rep"] for s in stim],
                               [s["ctl"] for s in stim], dims,
                               pos_A=slice(half, None), pos_B=slice(half, None))
    base_ind = induction_score(model, stim)
    print(f"  clean induction score: {base_ind['induction_score']:+.3f} "
          f"(repeat {base_ind['logprob_repeat']:+.2f} vs control "
          f"{base_ind['logprob_control']:+.2f})")

    # ---- text vs symbolic (own stimuli, length-matched) ----
    print("text-vs-symbolic localizer (length-matched, final position)...")
    txt, sym, L = text_symbolic_stimuli(model, n=args.n_text, seed=args.seed)
    ts = contrast_selectivity(model,
                              [model.encode(p, max_length=64) for p in txt],
                              [model.encode(p, max_length=64) for p in sym],
                              dims, pos_A=-1, pos_B=-1)
    print(f"  {len(txt)} text vs {len(sym)} symbolic prompts, {L} tokens each")

    # ---- assemble the per-unit table ----
    rows = []
    for kind, width in (("head", dims.n_heads), ("neuron", dims.d_mlp)):
        f, i, t = fac[kind], ind[kind], ts[kind]
        fe = fac_e[kind]
        e_op_minus_od = fe["ss_operator"] - fe["ss_operand"]
        null = torch.stack(nulls[kind])                       # [seeds, L, W]
        op_minus_od = f["ss_operator"] - f["ss_operand"]
        for l in range(dims.n_layers):
            for u in range(width):
                rows.append({
                    "unit_id": f"{kind}:{l}:{u}", "kind": kind, "layer": l,
                    "index": u, "depth": depth_percent(l, dims.n_layers),
                    "operator": float(op_minus_od[l, u]),
                    "operand": float(-op_minus_od[l, u]),
                    "operand_entity": float(-e_op_minus_od[l, u]),
                    "operator_entity": float(e_op_minus_od[l, u]),
                    "ss_operand_entity": float(fe["ss_operand"][l, u]),
                    "ss_total": float(f["ss_total"][l, u]),
                    "ss_operator": float(f["ss_operator"][l, u]),
                    "ss_operand": float(f["ss_operand"][l, u]),
                    "share_operator": float(f["share_operator"][l, u]),
                    "share_operand": float(f["share_operand"][l, u]),
                    "operator_null_max": float(null[:, l, u].max()),
                    "induction": float(i["contrast"][l, u]),
                    "text_symbolic": float(t["contrast"][l, u]),
                    "symbolic_text": float(-t["contrast"][l, u]),
                    "magnitude": float(f["magnitude"][l, u]),
                })
    df = pd.DataFrame(rows)
    out = ROOT / "results" / "lesion" / f"{tag}_{args.domain}_localizer.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    summary = geometry_summary(df, base_ind, dims.n_layers, args.k, seed=args.seed)
    summary["_meta"] = {
        "tag": tag, "domain": args.domain, "k_neurons": args.k,
        "dims": {"n_layers": dims.n_layers, "n_heads": dims.n_heads,
                 "head_dim": dims.head_dim, "d_mlp": dims.d_mlp},
        "n_induction": args.n_induction, "n_text": len(txt), "n_symbolic": len(sym),
        "text_symbolic_tokens": int(L), "null_seeds": args.null_seeds,
        "note": "operator/operand = per-unit two-way variance contrast on the "
                "balanced grid at the query position (null: permuted relation "
                "labels). induction = repeated-vs-nonrepeating random sequences, "
                "second half. text_symbolic = WikiText vs arithmetic/logic, "
                "length-matched at the final position. layer_entropy: 1 = "
                "distributed network, 0 = single-layer area."}
    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))
    print(f"\nsaved {out} (+ _summary.json)")


if __name__ == "__main__":
    main()
