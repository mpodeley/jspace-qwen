#!/usr/bin/env python
"""P1 of the vocab-semantics battery: does the case vector READ as anything?

The case vector v_l(k) (the operator main effect, == the steering vector) flips
answer margins and composes into generating states -- but does it MEAN anything
in vocabulary space? Todd et al. (ICLR'24, Tab. 5) decoded function vectors to
answer-space exemplars; Nadaf (2026) finds FV projections "universally
incoherent" despite >0.9 steering accuracy. Both are correlational projections.
This script produces the descriptive layer of our causal version:

  E1  PORTRAITS: top-k tokens of the case vector per layer, under two readouts
      -- raw:      unembed(v)            (scale-free; comparable with Nadaf)
      -- marginal: mean_o[unembed(H(o,k)) - unembed(H(o,k) - v)]
                   (the case's exact marginal logit contribution at the real
                   operating point; RMSNorm handled without linearization)
      and three lenses: vanilla logit lens, the fitted JacobianLens transport,
      and the random-projection control lens.
  E2  CATEGORY SCORES: mean marginal boost of relation j's answer-token set
      under relation k's case vector -> 5x5 matrix per layer, with null bands
      from permuted-label case vectors and random-in-operator-subspace vectors.
  E3  ALIGNMENT: cosine of v with the gamma-weighted unembedding row of the
      relation WORD (" capital") vs the centroid of its ANSWER tokens vs random
      rows (z-scores; raw cosines are incomparable across token frequencies).
  E5g SYNCRETIC MARKER, geometry: m = mean over syncretic operands of
      (H[o,language] - H[o,demonym]) -- zero answer-token information by
      construction. Its energy inside the answer-token span + its portrait.
  E6  OPERAND SIDE: cos(stem_o - stem_o', E_o - E_o') per layer (the
      king-queen analog on operands; descriptive only).

Verification baked in: asserts case[l][k] == op_dirs()[k][l] (template 0), the
same identity the paper verifies for v == b. Everything else is persisted to
results/ablation/{tag}_{domain}_vocab_portrait.parquet + _summary.json.

    python scripts/op_vocab_portrait.py 1.7b --domain relations
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, first_token, load_model, resolve_tag
from jlens import JacobianLens
import op_core

ROOT = Path(__file__).resolve().parent.parent


# --- vocabulary geometry helpers ---------------------------------------------

def answer_spans(dom, tok):
    """Per relation: unique first-token ids of its answers, minus tokens that
    collide with an operand's own first token (capital(Mexico)='Mexico' would
    smuggle operand identity into the 'answer' span -- pre-registered
    exclusion). Returns (spans, relation_word_tokens, exclusions)."""
    operand_toks = {first_token(tok, str(dom.items[o]["args"]["a"])): o
                    for o in dom.operand_keys}
    spans, excl = {}, []
    for k in dom.op_keys:
        ids = []
        for o in dom.operand_keys:
            t = dom.answer_tok(tok, o, k)
            if t in operand_toks:
                excl.append({"relation": k, "operand": o,
                             "answer": str(dom.answer(o, k)),
                             "collides_with": operand_toks[t]})
                continue
            if t not in ids:
                ids.append(t)
        spans[k] = ids
    # relation WORDS: only single-word operator phrases have an atomic token
    rel_words = {k: first_token(tok, phrase) for k, phrase in dom.operators.items()
                 if " " not in phrase.strip()}
    return spans, rel_words, excl


def classify_token(tid, k, spans, rel_words):
    if tid in spans[k]:
        return "answer_exemplar"
    if k in rel_words and tid == rel_words[k]:
        return "relation_word"
    if any(tid == w for kk, w in rel_words.items() if kk != k):
        return "other_relation_word"
    if any(tid in spans[j] for j in spans if j != k):
        return "other_relation_answer"
    return "other"


class Readout:
    """Batched vocab readouts through one lens. `transport=None` is the vanilla
    logit lens; else a JacobianLens (fitted or the randproj control)."""

    def __init__(self, model, lens=None):
        self.m = model
        self.lens = lens
        # effective RMSNorm scale: Gemma stores (1 + w), Qwen stores w directly
        w = model._final_norm.weight.detach().float()
        mt = getattr(model._hf_model.config, "model_type", "")
        self.gamma = (w + 1.0) if mt.startswith("gemma") else w
        self.W = model._lm_head.weight.detach()  # [vocab, d], bf16

    def _z(self, X, layer):
        """final_norm(transported X): [n, d] in lm_head dtype."""
        X = X.float()
        if self.lens is not None:
            X = self.lens.transport(X, layer)
        return self.m._final_norm(X.to(self.W.dtype))

    @torch.no_grad()
    def full(self, X, layer):
        """Full-vocab logits [n, vocab] (float32)."""
        return (self._z(X, layer) @ self.W.T).float()

    @torch.no_grad()
    def subset(self, X, layer, token_ids):
        """Logits restricted to `token_ids`: [n, len(ids)] (float32). Avoids
        materializing the vocab when only answer spans are scored."""
        return (self._z(X, layer) @ self.W[token_ids].T).float()

    def read_dirs(self, token_ids):
        """Gamma-weighted unembedding rows r_t = gamma * W_U[t] (float32) --
        the pre-norm direction whose amplification raises token t's logit."""
        return self.W[token_ids].float() * self.gamma


def cos_rows(v, R):
    """cos(v, each row of R): [n]."""
    v = v / (v.norm() + 1e-9)
    R = R / (R.norm(dim=1, keepdim=True) + 1e-9)
    return (R @ v).float()


# --- main ---------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--null-seeds", type=int, default=10)
    ap.add_argument("--rand-tokens", type=int, default=2000,
                    help="random vocab rows for the E3 z-score reference")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(layers, model.n_layers)["workspace"]
    ops = dom.op_keys
    operands = dom.operand_keys
    tied = bool(getattr(model._hf_model.config, "tie_word_embeddings", False))

    lenses = {"vanilla": None}
    for name, fn in [("jlens", ROOT / "out" / "lenses" / f"{tag}.pt"),
                     ("randproj", ROOT / "out" / "lenses" / f"{tag}-randproj.pt")]:
        if fn.exists():
            L = JacobianLens.from_pretrained(fn)
            if all(l in L.source_layers for l in ws):
                lenses[name] = L
            else:
                print(f"[warn] {name} lens misses ws layers; skipped")
        else:
            print(f"[warn] {fn.name} not found; {name} lens skipped")

    print(f"[{tag} / {args.domain}] vocab portrait, ws={ws}, "
          f"lenses={list(lenses)}, tied_embeddings={tied}")

    # one residual grid; everything (components, dirs, permuted nulls) derives
    # from it, so the v == case identity check is exact, not approximate
    R = op_core.op_resids(model, ws, dom)
    H = {(o, k): R[(o, k, 0)] for o in operands for k in ops}
    comp = op_core.factorize_components(model, ws, dom, H=H)
    case, mu, stem = comp["case"], comp["mu"], comp["stem"]
    v = op_core.dirs_from_resids(R, ws, ops, operands, [0], model.d_model)
    worst = max(float((case[l][k] - v[k][l]).abs().max())
                for l in ws for k in ops)
    assert worst < 1e-4, f"case != op_dirs (max abs diff {worst})"
    print(f"identity check: case[l][k] == op_dirs()[k][l], max|diff| = {worst:.2e}")

    spans, rel_words, exclusions = answer_spans(dom, tok)
    span_union = sorted({t for ids in spans.values() for t in ids})
    dec = {t: tok.decode([t]) for t in span_union}
    print("answer spans:", {k: [dec[t] for t in ids] for k, ids in spans.items()})
    if exclusions:
        print("excluded (operand collision):", exclusions)

    # null vectors for E2: permuted-label case (destroys semantics, keeps all
    # statistics) and random-in-operator-subspace (norm-matched per layer)
    Q = op_core.operator_subspace_basis(v, ws)
    g = torch.Generator().manual_seed(args.seed)
    nulls = {"perm": [], "rand_sub": []}
    for s in range(args.null_seeds):
        pv = op_core.permuted_op_dirs(model, ws, dom, seed=s, per_operand=True, R=R)
        nulls["perm"].append({k: {l: pv[k][l] for l in ws} for k in ops})
        rs = {}
        for k in ops:
            rv = op_core.random_in_subspace(
                Q, {l: case[l][k].norm() for l in ws}, g)
            rs[k] = rv
        nulls["rand_sub"].append(rs)

    rows = []          # portrait parquet rows
    e2 = {}            # {lens: {layer: {"obs": 5x5, "perm": [...], "rand_sub": [...]}}}
    e3 = []            # alignment rows
    e5, e6 = {}, {}

    # E5g marker: built ONLY on syncretic operands (identical answers)
    ka, kb = dom.desinence_pair
    syn = [o for o in operands if dom.answer(o, ka) == dom.answer(o, kb)]
    m_syn = {l: torch.stack([H[(o, ka)][l] - H[(o, kb)][l] for o in syn]).mean(0)
             for l in ws}
    span_pair = sorted(set(spans[ka]) | set(spans[kb]))

    # E6 reference: embedding differences of operand first tokens
    E = model._embed_tokens.weight.detach()
    op_tok = {o: first_token(tok, str(dom.items[o]["args"]["a"])) for o in operands}
    pairs_oo = [(a, b) for i, a in enumerate(operands) for b in operands[i + 1:]]

    rand_ids = torch.randperm(model._lm_head.weight.shape[0],
                              generator=torch.Generator().manual_seed(args.seed)
                              )[:args.rand_tokens].tolist()

    for lens_name, L in lenses.items():
        ro = Readout(model, L)
        e2[lens_name] = {}
        for l in ws:
            # ---- batched pieces for this (lens, layer) ----
            Hmat = torch.stack([H[(o, k)][l] for o in operands for k in ops])
            U_H = ro.subset(Hmat, l, span_union)              # [60, |union|]
            idx = {(o, k): i for i, (o, k) in enumerate(
                [(o, k) for o in operands for k in ops])}
            col = {t: j for j, t in enumerate(span_union)}

            def marginal_scores(cvecs):
                """cvecs: {k: vec}. Returns {k: {j: mean marginal boost of
                relation j's span under k's vector}} via the subset trick."""
                Hm = torch.stack([H[(o, k)][l] - cvecs[k] for o in operands
                                  for k in ops])
                U_Hm = ro.subset(Hm, l, span_union)
                delta = U_H - U_Hm                            # [60, |union|]
                out = {}
                for k in ops:
                    rws = [idx[(o, k)] for o in operands]
                    d = delta[rws]                            # [12, |union|]
                    out[k] = {j: float(d[:, [col[t] for t in spans[j]]].mean())
                              for j in ops}
                return out

            obs = marginal_scores({k: case[l][k] for k in ops})
            e2[lens_name][l] = {"obs": obs, "perm": [], "rand_sub": []}
            for kind in ("perm", "rand_sub"):
                for s in range(args.null_seeds):
                    cv = {k: nulls[kind][s][k][l] for k in ops}
                    e2[lens_name][l][kind].append(marginal_scores(cv))

            # ---- E1 portraits (full vocab; small batch) ----
            for k in ops:
                c = case[l][k]
                raw_p = ro.full(torch.stack([c, -c]), l)      # [2, vocab]
                Hk = torch.stack([H[(o, k)][l] for o in operands])
                marg = (ro.full(Hk, l) - ro.full(Hk - c, l)).mean(0)  # [vocab]
                for readout, sign, scores in [
                        ("raw", "+", raw_p[0]), ("raw", "-", raw_p[1]),
                        ("marginal", "+", marg), ("marginal", "-", -marg)]:
                    top = torch.topk(scores, args.topk)
                    for r, (tid, sc) in enumerate(zip(top.indices.tolist(),
                                                      top.values.tolist())):
                        rows.append({
                            "layer": l, "relation": k, "lens": lens_name,
                            "readout": readout, "sign": sign, "rank": r,
                            "token_id": tid, "token": tok.decode([tid]),
                            "score": sc,
                            "class": classify_token(tid, k, spans, rel_words)})

            # ---- E3 alignment (vanilla lens only; cosines live pre-norm) ----
            if lens_name == "vanilla":
                R_rand = ro.read_dirs(rand_ids)
                for k in ops:
                    c = case[l][k].float()
                    rnd = cos_rows(c, R_rand)
                    mu_r, sd_r = float(rnd.mean()), float(rnd.std())
                    cen = ro.read_dirs(spans[k]).mean(0)
                    cos_cen = float(torch.nn.functional.cosine_similarity(
                        c, cen, dim=0))
                    row = {"layer": l, "relation": k,
                           "cos_answer_centroid": cos_cen,
                           "z_answer_centroid": (cos_cen - mu_r) / (sd_r + 1e-9),
                           "rand_mean": mu_r, "rand_sd": sd_r}
                    if k in rel_words:
                        cw = float(cos_rows(c, ro.read_dirs([rel_words[k]]))[0])
                        row["cos_relation_word"] = cw
                        row["z_relation_word"] = (cw - mu_r) / (sd_r + 1e-9)
                    e3.append(row)

                # ---- E5g marker energy + portrait ----
                Rp = ro.read_dirs(span_pair)
                Qp, _ = torch.linalg.qr(Rp.T)                  # [d, r]
                mm = m_syn[l].float()
                energy = float((Qp.T @ mm).pow(2).sum() / (mm.pow(2).sum() + 1e-9))
                g2 = torch.Generator().manual_seed(args.seed + l)
                rnd_e = []
                for _ in range(50):
                    x = torch.randn(model.d_model, generator=g2).to(Qp.device)
                    rnd_e.append(float((Qp.T @ x).pow(2).sum() / x.pow(2).sum()))
                mtop = torch.topk(ro.full(mm[None], l)[0], args.topk)
                e5[l] = {"energy_in_pair_span": energy,
                         "rand_energy_mean": sum(rnd_e) / len(rnd_e),
                         "n_syncretic_build": len(syn),
                         "marker_top": [
                             {"token": tok.decode([t]), "score": s,
                              "class": classify_token(t, ka, spans, rel_words)}
                             for t, s in zip(mtop.indices.tolist(),
                                             mtop.values.tolist())][:10]}

                # ---- E6 operand side ----
                cs = []
                for a, b in pairs_oo:
                    dstem = (stem[l][a] - stem[l][b]).float()
                    dE = (E[op_tok[a]] - E[op_tok[b]]).float()
                    cs.append(float(torch.nn.functional.cosine_similarity(
                        dstem, dE, dim=0)))
                e6[l] = {"cos_mean": sum(cs) / len(cs),
                         "cos_min": min(cs), "cos_max": max(cs)}

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_vocab_portrait.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)

    # ---- condensed summary: diagonal contrast with null bands ----
    def diag_contrast(mat):
        d = sum(mat[k][k] for k in ops) / len(ops)
        off = sum(mat[k][j] for k in ops for j in ops if j != k) / (len(ops) * (len(ops) - 1))
        return d - off

    e2_summary = {}
    for lens_name in e2:
        e2_summary[lens_name] = {}
        for l in ws:
            obs = diag_contrast(e2[lens_name][l]["obs"])
            bands = {kind: [diag_contrast(m) for m in e2[lens_name][l][kind]]
                     for kind in ("perm", "rand_sub")}
            e2_summary[lens_name][l] = {
                "diag_contrast": obs,
                **{f"{kind}_mean": sum(b) / len(b) for kind, b in bands.items()},
                **{f"{kind}_max": max(b) for kind, b in bands.items()}}

    summary = {
        "e2_diag_contrast": {ln: {str(l): d for l, d in ls.items()}
                             for ln, ls in e2_summary.items()},
        "e2_matrices": {ln: {str(l): e2[ln][l]["obs"] for l in ws} for ln in e2},
        "e3_alignment": e3,
        "e5_marker": {str(l): e5[l] for l in e5},
        "e6_stem_vs_embedding": {str(l): e6[l] for l in e6},
        "_meta": {"tag": tag, "domain": args.domain, "ws": [int(x) for x in ws],
                  "lenses": list(lenses), "tied_embeddings": tied,
                  "topk": args.topk, "null_seeds": args.null_seeds,
                  "identity_max_abs_diff": worst,
                  "answer_spans": {k: [dec[t] for t in ids]
                                   for k, ids in spans.items()},
                  "relation_word_tokens": {k: tok.decode([t])
                                           for k, t in rel_words.items()},
                  "exclusions": exclusions,
                  "note": "raw = unembed(v) (scale-free top-k); marginal = "
                          "mean_o[unembed(H)-unembed(H-v)] at the real operating "
                          "point. e2 scores are marginal boosts on answer spans; "
                          "nulls: permuted-label case + random-in-op-subspace, "
                          "norm-matched."}}
    out.with_suffix("").with_name(out.stem + "_summary.json").write_text(
        json.dumps(summary, indent=2))

    # ---- human gate: qualitative portrait at three depths ----
    show = [ws[0], ws[len(ws) // 2], ws[-1]]
    for l in show:
        print(f"\n=== layer {l} ({100 * l / (model.n_layers - 1):.0f}% depth), "
              f"vanilla lens ===")
        for k in ops:
            for readout in ("raw", "marginal"):
                sub = df[(df.layer == l) & (df.relation == k) & (df.lens == "vanilla")
                         & (df.readout == readout) & (df.sign == "+")].head(8)
                toks = " ".join(f"{t!r}{'*' if c == 'answer_exemplar' else '†' if c == 'relation_word' else ''}"
                                for t, c in zip(sub.token, sub["class"]))
                print(f"  {k:<10} {readout:<9} {toks}")
        c = e2_summary["vanilla"][l]
        print(f"  E2 diag contrast {c['diag_contrast']:+.3f}  "
              f"(perm max {c['perm_max']:+.3f}, rand_sub max {c['rand_sub_max']:+.3f})")
    print(f"\nsaved {out} (+ _summary.json)")


if __name__ == "__main__":
    main()
