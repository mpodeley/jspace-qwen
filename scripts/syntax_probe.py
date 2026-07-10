#!/usr/bin/env python
"""Readout-rescue experiment: does the J-lens READOUT carry grammatical number
better than the logit-lens, measured by a linear probe over the lens logits --
not the lossy two-token difference that the form eval used?

Motivation. lens_eval's form metric reads a single logit difference (are vs is)
and found no J-lens advantage. But that is a 1-D readout; the number signal may
live in a richer set of logit dimensions that a probe can see. This script trains
an L2 logistic probe to predict number from the lens logits restricted to a
number-diagnostic axis set, per depth band, for the J-lens, the logit-lens, and
the randproj null, and reports Hewitt & Liang (2019) selectivity = AUC(real) -
AUC(shuffled-labels). The claim under test: selectivity(J) > selectivity(logit)
in some band. If it does not hold, the readout claim is a null and we say so.

CRITICAL (data-processing inequality). transport() is linear and low-rank, so a
linear probe on J_l @ h can only *lose* information relative to raw h -- probing
the transported residual would be a guaranteed null by construction. We therefore
probe the LENS LOGITS, i.e. unembed(J_l @ h): unembed applies a NON-linear RMSNorm
before the (non-square) lm_head, so the J and logit readouts can genuinely differ.
The features come from lens.apply(...) (logits), never from lens.transport().

    python scripts/syntax_probe.py 1.7b
    python scripts/syntax_probe.py 1.7b --lens out/lenses/1.7b-randproj.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from _common import BANDS, band_layers, load_model, resolve_tag, single_leading_space_token
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
MORPH = ROOT / "data" / "morphosyntax"

VERB_AXES = ["is", "are", "was", "were", "has", "have", "do", "does"]


def number_axis(tok, items) -> list[int]:
    """Number-diagnostic token ids: agreeing verbs + each noun's sing/plur forms."""
    ids = set()
    for w in VERB_AXES:
        t = single_leading_space_token(tok, w)
        if t is not None:
            ids.add(t)
    for it in items:
        for w in (it["noun"], it["noun"] + "s"):
            t = single_leading_space_token(tok, w)
            if t is not None:
                ids.add(t)
    return sorted(ids)


@torch.no_grad()
def band_features(model, lens, prompt, axis, bands, *, use_jacobian):
    """{band: feature vector over `axis`} = mean over layers-in-band of the lens
    logits at the last position, restricted to the number-diagnostic axis."""
    ll, _, _ = lens.apply(model, prompt, layers=lens.source_layers,
                          positions=[-1], use_jacobian=use_jacobian)
    axis_t = torch.tensor(axis)
    feats = {}
    for b, layers in bands.items():
        if not layers:
            feats[b] = None
            continue
        stack = torch.stack([ll[l][0][axis_t] for l in layers])  # [n_layer, |axis|]
        feats[b] = stack.mean(0).float().numpy()
    return feats


def _fit_logreg(X, y, *, l2=1.0, iters=300, lr=0.1):
    """Tiny L2 logistic regression (full-batch gradient descent, torch)."""
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    w = torch.zeros(Xt.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], lr=lr, max_iter=iters)

    def closure():
        opt.zero_grad()
        logits = Xt @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, yt)
        loss = loss + l2 * (w @ w) / len(yt)
        loss.backward()
        return loss

    opt.step(closure)
    return w.detach().numpy(), float(b.detach())


def _auc(scores, labels) -> float:
    """ROC AUC via the Mann-Whitney U statistic."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def grouped_auc(X, y, groups, *, shuffle=False, seed=0, k=5):
    """Grouped k-fold AUC. Folds split by `groups` (noun) so no noun's examples
    straddle train/test. With shuffle=True, labels are permuted within train to
    get the Hewitt-Liang control ceiling."""
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups)))
    rng.shuffle(uniq)
    folds = np.array_split(uniq, min(k, len(uniq)))
    scores = np.full(len(y), np.nan)
    for test_groups in folds:
        te = np.isin(groups, test_groups)
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        ytr = y[tr].copy()
        if shuffle:
            rng.shuffle(ytr)
        w, b = _fit_logreg(Xtr, ytr)
        scores[te] = Xte @ w + b
    ok = ~np.isnan(scores)
    return _auc(scores[ok], y[ok])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    items = json.loads((MORPH / "number-probe.json").read_text())["items"]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(args.model)
    else:
        model = load_model(args.model)
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    bands = band_layers(lens.source_layers, model.n_layers)
    axis = number_axis(tok, items)

    y = np.array([1 if it["number"] == "plur" else 0 for it in items])
    groups = np.array([it["noun"] for it in items])

    # Collect features once per lens-variant; the J and logit variants share the
    # same forward pass (apply toggles use_jacobian), so this is 2 passes/item.
    results = {}
    for which, use_j in (("jlens", True), ("logit", False)):
        feats = {b: [] for b in BANDS}
        for it in items:
            bf = band_features(model, lens, it["prompt"], axis, bands, use_jacobian=use_j)
            for b in BANDS:
                feats[b].append(bf[b] if bf[b] is not None else np.zeros(len(axis)))
        results[which] = {}
        for b in BANDS:
            if not bands[b]:
                continue
            X = np.array(feats[b])
            auc = grouped_auc(X, y, groups, shuffle=False, seed=args.seed)
            auc0 = grouped_auc(X, y, groups, shuffle=True, seed=args.seed)
            results[which][b] = {"auc": auc, "auc_shuffled": auc0,
                                 "selectivity": auc - auc0}

    out = {"model": tag, "lens": lens_path.stem, "n_items": len(items),
           "n_axis": len(axis), "bands": results}
    print(json.dumps(out, indent=2))
    print("\n  selectivity = AUC(real) - AUC(shuffled). claim: jlens > logit somewhere")
    print(f"  {'band':<10} {'J sel':>8} {'logit sel':>10} {'J auc':>8} {'logit auc':>10}")
    for b in BANDS:
        j = results["jlens"].get(b)
        l = results["logit"].get(b)
        if j and l:
            print(f"  {b:<10} {j['selectivity']:>8.3f} {l['selectivity']:>10.3f} "
                  f"{j['auc']:>8.3f} {l['auc']:>10.3f}")

    dest = ROOT / "results" / "metrics" / f"{tag}_syntaxprobe_{lens_path.stem}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
