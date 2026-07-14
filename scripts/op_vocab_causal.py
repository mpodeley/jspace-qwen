#!/usr/bin/env python
"""P2 of the vocab-semantics battery: WHICH PART of the case vector is causal?

Todd et al. decoded function vectors to answer exemplars; Nadaf finds FV
projections incoherent. Both are correlational. Here we split the very vector
that flips margins and composes into generating states,

    case = c_ans (+) c_rest,

where c_ans is the exact orthogonal projection onto the span of the
gamma-weighted unembedding rows of the relation's answer tokens (the only part
a logit lens could ever attribute to "answer directions"), and c_rest is the
complement. Then we inject each part alone, at the paper's published
conditions, and ask which part carries (a) the margin flip, (b) the
generation.

  E4  SWAPS: dv = alpha*(v[to]-v[frm]) split on span{r_t : t in A_frm u A_to};
      band alpha=4 and single-mid-layer alpha=1 (the published conditions).
      Nulls: random-in-pair-span (norm-matched to dv_ans) and wrong-span
      (dv projected onto the OTHER relations' answer span, norm-matched).
  E4p PATCH: composed = mu + stem + {case | c_ans | c_rest} at the query
      position (band); k=8 greedy text + class + forced choice -- the
      generation-level readout.
  E5  SYNCRETIC MARKER, causal: m = mean over the 8 syncretic operands of
      (H[o,language] - H[o,demonym]) -- answer-free by construction -- tested
      on the 4 disambiguating operands (Brazil/Egypt/India/Mexico): +-alpha*m
      on demonym and language prompts must move the language-vs-demonym answer
      margin in the predicted directions. n=4 operands: per-operand values
      reported, operand bootstrap CI, stated as small-n.

Pre-registered hypotheses: (A) dissociation -- c_ans dominates the late margin
flip, c_rest is required for generation; (B) all-in-c_rest (pure Nadaf world).
Additivity of delta_margin(ans)+delta_margin(rest) vs full is EMPIRICAL (the
downstream computation is nonlinear); its residual is itself a result.

    python scripts/op_vocab_causal.py 1.7b --domain relations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
import op_core
import op_minimal
from op_audit import _hooked, classify, seq_logprob
from op_vocab_portrait import Readout, answer_spans

ROOT = Path(__file__).resolve().parent.parent


def span_basis(ro: Readout, token_ids):
    """Orthonormal basis [d, r] of span{gamma * W_U[t] : t in token_ids}."""
    R = ro.read_dirs(token_ids)                     # [n, d] float32
    Q, Rf = torch.linalg.qr(R.T)
    d = Rf.diagonal().abs()
    return Q[:, d > 1e-6 * max(float(d.max()), 1e-30)]


def project(Q, x):
    return Q @ (Q.T @ x.float())


def rand_in_span(Q, norm, gen):
    c = torch.randn(Q.shape[1], generator=gen).to(Q.device)
    u = Q @ c
    return u / (u.norm() + 1e-9) * norm


def family_ci(df, col, n_boot=10_000, seed=0, ci=95):
    """Dyadic operator-level bootstrap (operators top level, operands nested)
    of the mean of `col` -- the same dependence structure as
    op_core.bootstrap_family_ci, generalized to any value column."""
    rng = np.random.default_rng(seed)
    lo_q, hi_q = (100 - ci) / 2, 100 - (100 - ci) / 2
    ops = list(dict.fromkeys(df["from"]))
    groups = {(f, t): s[col].to_numpy()
              for (f, t), s in df.groupby(["from", "to"], sort=False)}
    means = np.empty(n_boot)
    for b in range(n_boot):
        cnt = {k: 0 for k in ops}
        for k in rng.choice(ops, size=len(ops), replace=True):
            cnt[k] += 1
        tw = tv = 0.0
        for (f, t), vals in groups.items():
            w = cnt[f] * cnt[t]
            if w == 0:
                continue
            idx = rng.integers(0, len(vals), size=len(vals))
            tw += w
            tv += w * vals[idx].mean()
        means[b] = tv / tw if tw else np.nan
    return {"mean": float(df[col].mean()),
            "lo": float(np.nanpercentile(means, lo_q)),
            "hi": float(np.nanpercentile(means, hi_q))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alpha", type=float, default=4.0, help="band swap dose")
    ap.add_argument("--span-variant", default="answers",
                    choices=["answers", "answers+word"])
    ap.add_argument("--rand-seeds", type=int, default=2)
    ap.add_argument("--k", type=int, default=8, help="greedy tokens")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--marker-only", action="store_true",
                    help="skip the E4 swap/patch loop (reuse the persisted "
                         "parquet + summary) and run only the E5 marker test")
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(layers, model.n_layers)["workspace"]
    mid = ws[len(ws) // 2]
    ops = dom.op_keys
    operands = dom.operand_keys
    ro = Readout(model)

    print(f"[{tag} / {args.domain}] vocab causal split, ws={ws}, mid={mid}, "
          f"alpha={args.alpha}, span={args.span_variant}")

    R = op_core.op_resids(model, ws, dom)
    H = {(o, k): R[(o, k, 0)] for o in operands for k in ops}
    comp = op_core.factorize_components(model, ws, dom, H=H)
    case, mu, stem = comp["case"], comp["mu"], comp["stem"]
    v = op_core.dirs_from_resids(R, ws, ops, operands, [0], model.d_model)

    spans, rel_words, exclusions = answer_spans(dom, tok)
    if args.span_variant == "answers+word":
        for k, w in rel_words.items():
            if w not in spans[k]:
                spans[k] = spans[k] + [w]

    # per-relation and per-pair span bases (layer-independent: the readout rows
    # live in the final basis, and the split is defined against that readout)
    Q_rel = {k: span_basis(ro, spans[k]) for k in ops}
    Q_pair = {}
    for frm in ops:
        for to in ops:
            if frm != to:
                union = sorted(set(spans[frm]) | set(spans[to]))
                Q_pair[(frm, to)] = span_basis(ro, union)
    Q_other = {}
    for frm in ops:
        for to in ops:
            if frm != to:
                rest_tokens = sorted({t for k in ops if k not in (frm, to)
                                      for t in spans[k]})
                Q_other[(frm, to)] = span_basis(ro, rest_tokens)

    g = torch.Generator().manual_seed(args.seed)
    pairs = [] if args.marker_only else [(a, b) for a in ops for b in ops if a != b]
    if args.marker_only:
        print("marker-only: skipping E4 (parquet already persisted)")
    rows = []
    fc_conds = set()

    for frm, to in pairs:
        Qp, Qo = Q_pair[(frm, to)], Q_other[(frm, to)]
        dv = {l: args.alpha * (v[to][l] - v[frm][l]) for l in ws}
        dv_ans = {l: project(Qp, dv[l]) for l in ws}
        dv_rest = {l: dv[l] - dv_ans[l] for l in ws}
        share = (sum(float(dv_ans[l].pow(2).sum()) for l in ws)
                 / (sum(float(dv[l].pow(2).sum()) for l in ws) + 1e-12))
        rand_span = [{l: rand_in_span(Qp, dv_ans[l].norm(), g) for l in ws}
                     for _ in range(args.rand_seeds)]
        wrong = {l: (lambda p: p / (p.norm() + 1e-9) * dv_ans[l].norm())(
            project(Qo, dv[l])) for l in ws}
        dv1 = {mid: 1.0 * (v[to][mid] - v[frm][mid])}
        dv1_ans = {mid: project(Qp, dv1[mid])}
        dv1_rest = {mid: dv1[mid] - dv1_ans[mid]}

        for o in operands:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            p = dom.render(o, frm)
            L0 = op_core.final_logits(model, p)
            target, source = str(dom.answer(o, to)), str(dom.answer(o, frm))
            other_op = [str(dom.answer(oo, to)) for oo in operands
                        if oo != o and str(dom.answer(oo, to)).lower()
                        not in (target.lower(), source.lower())]
            other_rel = [str(dom.answer(o, k)) for k in ops if k not in (frm, to)
                         and str(dom.answer(o, k)).lower()
                         not in (target.lower(), source.lower())]

            conds = {
                ("band", "full"): {"add": dv},
                ("band", "ans"): {"add": dv_ans},
                ("band", "rest"): {"add": dv_rest},
                ("band", "wrong_span"): {"add": wrong},
                ("1L-mid", "full"): {"add": dv1, "add_positions": [-1]},
                ("1L-mid", "ans"): {"add": dv1_ans, "add_positions": [-1]},
                ("1L-mid", "rest"): {"add": dv1_rest, "add_positions": [-1]},
            }
            for s, rv in enumerate(rand_span):
                conds[("band", f"rand_span{s}")] = {"add": rv}
            # composed patch: generation-level readout, split on A_to
            Qt = Q_rel[to]
            c_full = {l: case[l][to] for l in ws}
            c_ans = {l: project(Qt, case[l][to]) for l in ws}
            for name, cc in [("full", c_full), ("ans", c_ans),
                             ("rest", {l: c_full[l] - c_ans[l] for l in ws})]:
                conds[("patch", name)] = {"patch": {
                    l: mu[l] + stem[l][o] + cc[l] for l in ws}}
                fc_conds.add(("patch", name))

            for (scope, component), kw in conds.items():
                mb = op_core.metric_bundle(model, dom, tok, o, frm, to,
                                           clean_logits=L0, k=args.k, **kw)
                text = mb.pop("text")
                cls = classify(text, target, source, other_op, other_rel)
                row = {"from": frm, "to": to, "operand": o, "scope": scope,
                       "component": component, "ans_share_band": share,
                       "class": cls, "text": text, **mb}
                if (scope, component) in fc_conds:
                    cands = {k: str(dom.answer(o, k)) for k in ops}
                    n_prompt = model.encode(p, max_length=64).shape[1]
                    scores = {}
                    for k, ans in cands.items():
                        key = ans.lower()
                        if key not in scores:
                            with _hooked(model, kw, n_prompt):
                                scores[key] = (seq_logprob(
                                    model, p, ans, dom.answer_leading_space), k, ans)
                    best = max(scores.values(), key=lambda x: x[0])
                    row["forced_choice_target"] = best[2].lower() == target.lower()
                rows.append(row)
        print(f"  {frm}->{to} done ({len(rows)} rows)")

    if not args.marker_only:
        df = pd.DataFrame(rows)
        out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_vocab_causal.parquet"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)

        summary = {"_meta": {
            "tag": tag, "domain": args.domain, "ws": [int(x) for x in ws],
            "mid_layer": int(mid), "alpha_band": args.alpha,
            "span_variant": args.span_variant, "k": args.k,
            "exclusions": exclusions,
            "note": "swap split on span{gamma*W_U[t]: t in A_frm u A_to}; patch "
                    "split on A_to; wrong_span = dv projected on the other "
                    "relations' span, norm-matched to dv_ans; rand_span = random "
                    "in pair span, norm-matched to dv_ans."}}
        for (scope, component), sub in df.groupby(["scope", "component"]):
            key = f"{scope}/{component}"
            stats = {"n": int(len(sub)),
                     "delta_margin": family_ci(sub, "delta_margin", seed=args.seed),
                     "flip_frac": float((sub.groupby(["from", "to"])["margin"]
                                         .mean() > 0).mean()),
                     "top1": float(sub["top1"].mean()),
                     "exact_match": float(sub["exact_match"].mean()),
                     "kl_ontask_median": float(sub["kl_ontask"].median()),
                     "class_target": float((sub["class"] == "target").mean())}
            if "forced_choice_target" in sub and sub["forced_choice_target"].notna().any():
                stats["forced_choice_target"] = float(
                    sub["forced_choice_target"].mean())
            summary[key] = stats
        # empirical additivity of the split (band swaps)
        piv = df[df.scope == "band"].pivot_table(
            index=["from", "to", "operand"], columns="component",
            values="delta_margin")
        if {"full", "ans", "rest"} <= set(piv.columns):
            resid_add = (piv["ans"] + piv["rest"] - piv["full"])
            summary["_additivity_band"] = {
                "mean_abs_residual": float(resid_add.abs().mean()),
                "mean_full": float(piv["full"].mean())}
        summary["_ans_energy_share_band"] = {
            f"{f}->{t}": float(s["ans_share_band"].iloc[0])
            for (f, t), s in df[df.scope == "band"].groupby(["from", "to"])}
        out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
            json.dumps(summary, indent=2))

        print(f"\n{'condition':<22} {'d-margin':>9} {'CI':>19} {'flips':>6} "
              f"{'top1':>6} {'exact':>6} {'fc':>6}")
        for key in sorted(k for k in summary if "/" in k):
            s = summary[key]
            dm = s["delta_margin"]
            fc = s.get("forced_choice_target")
            fc_str = "—" if fc is None else format(fc, ".1%")
            print(f"{key:<22} {dm['mean']:>+9.2f} "
                  f"[{dm['lo']:>+7.2f},{dm['hi']:>+7.2f}] "
                  f"{s['flip_frac']:>6.0%} {s['top1']:>6.1%} {s['exact_match']:>6.1%} "
                  f"{fc_str:>6}")

    # ---- E5 causal: the syncretic marker on disambiguating operands --------
    ka, kb = dom.desinence_pair                       # (language, demonym)
    syn = [o for o in operands if dom.answer(o, ka) == dom.answer(o, kb)]
    dis = [o for o in operands if dom.answer(o, ka) != dom.answer(o, kb)]
    m = {l: torch.stack([H[(o, ka)][l] - H[(o, kb)][l] for o in syn]).mean(0)
         for l in ws}
    union = sorted(set(spans[ka]) | set(spans[kb]))
    Qm = span_basis(ro, union)
    m_energy = (sum(float(project(Qm, m[l]).pow(2).sum()) for l in ws)
                / sum(float(m[l].pow(2).sum()) for l in ws))
    mrows = []
    for o in dis:
        a_lang, a_dem = dom.answer_tok(tok, o, ka), dom.answer_tok(tok, o, kb)
        for base, sgn_expect in [(kb, +1), (ka, -1)]:
            # on a demonym prompt, +m should push toward the LANGUAGE answer;
            # on a language prompt, -m should push toward the DEMONYM answer
            p = dom.render(o, base)
            L0 = op_core.final_logits(model, p)
            clean = float(L0[a_lang] - L0[a_dem])
            for alpha in (1.0, 2.0, 4.0):
                for sgn in (+1, -1):
                    L = op_core.final_logits(
                        model, p, {l: sgn * alpha * m[l] for l in ws})
                    mrows.append({
                        "operand": o, "base_prompt": base, "alpha": alpha,
                        "sign": sgn, "clean_margin_lang": clean,
                        "margin_lang": float(L[a_lang] - L[a_dem]),
                        "delta_lang": float(L[a_lang] - L[a_dem]) - clean,
                        "predicted_direction": sgn})
    mdf = pd.DataFrame(mrows)
    mout = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_marker_causal.parquet"
    mdf.to_parquet(mout)
    rng = np.random.default_rng(args.seed)
    msum = {"_meta": {"built_on": syn, "tested_on": dis,
                      "energy_in_pair_span": m_energy,
                      "note": "m = mean_syncretic(H[lang]-H[dem]); +m on a "
                              "demonym prompt must raise the language-answer "
                              "margin, -m lower it (and mirrored on language "
                              "prompts). n=4 operands: small-n, operand "
                              "bootstrap."}}
    for (alpha, sgn), sub in mdf.groupby(["alpha", "sign"]):
        per_op = sub.groupby("operand")["delta_lang"].mean()
        boots = [per_op.sample(len(per_op), replace=True,
                               random_state=int(rng.integers(1 << 31))).mean()
                 for _ in range(4000)]
        msum[f"alpha={alpha}/sign={'+' if sgn > 0 else '-'}"] = {
            "delta_lang_mean": float(sub["delta_lang"].mean()),
            "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5)),
            "per_operand": {o: float(x) for o, x in per_op.items()},
            "sign_correct_frac": float(((sub["delta_lang"] * sub["sign"]) > 0).mean())}
    mout.with_suffix("").with_name(mout.stem + "_summary.json").write_text(
        json.dumps(msum, indent=2))
    print(f"\nmarker: energy in answer span {m_energy:.2%} (built on {len(syn)}, "
          f"tested on {len(dis)})")
    for key in sorted(k for k in msum if k.startswith("alpha")):
        s = msum[key]
        print(f"  {key:<18} d(lang margin) {s['delta_lang_mean']:>+7.2f} "
              f"[{s['lo']:>+6.2f},{s['hi']:>+6.2f}]  sign-correct "
              f"{s['sign_correct_frac']:.0%}")
    saved = f"{mout} (+ _summary.json)"
    if not args.marker_only:
        saved = f"{ROOT / 'results' / 'ablation' / f'{tag}_{args.domain}_vocab_causal.parquet'} (+ _summary.json), " + saved
    print(f"\nsaved {saved}")


if __name__ == "__main__":
    main()
