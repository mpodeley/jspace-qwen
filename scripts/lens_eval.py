#!/usr/bin/env python
"""Lens-quality eval, in two halves that together make the project's argument.

SEMANTIC IDENTITY (the shipped multihop set). For each two-hop prompt we read out
at the token before `target` across all fitted layers and ask whether each
`intermediate` (the unstated bridge entity, e.g. "Brazil") appears at lens rank
<= k for its best layer. pass@k, J-lens vs logit-lens. On Qwen3 the J-lens only
*ties* the logit-lens here -- and that is the point: pass@k scores next-token
*identity*, a lexical-semantic axis, so a tie is exactly what the pointer-table
hypothesis predicts (the workspace names the lemma; both lenses can read a lemma).

    python scripts/lens_eval.py 1.7b                      # multihop identity

SURFACE FORM (the new lemma-form set). Conditioning on the model getting the
clean case right, does the lens rank the correct surface form (mice, are, Paris)
above a minimal wrong form (mouses, is, paris)? Scored by a normalized logit
difference (Zhang & Nanda 2024: a difference, not a probability, so a suppressed
form is visible), taken at the layer that resolves it most sharply, and profiled
per band to locate *where* the form is decided. This is where we expect the
J-lens to beat the logit-lens.

    python scripts/lens_eval.py 1.7b --set lemma-form     # surface form

The dataset schema selects which eval runs (an `intermediates` key -> identity,
a `form_correct` key -> form).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import BANDS, MODELS, band_layers, first_token, load_model, resolve_tag
from causal_swap import greedy_next
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "jacobian-lens" / "data" / "evaluations"


@torch.no_grad()
def ranks_at_last_pos(model, lens, prompt, *, use_jacobian):
    """min-over-layers rank of every vocab token at the last prompt position."""
    lens_logits, _, _ = lens.apply(
        model, prompt, layers=lens.source_layers, positions=[-1],
        use_jacobian=use_jacobian,
    )
    # stack layers -> [n_layers, vocab]; rank = # tokens strictly greater
    stack = torch.stack([lens_logits[l][0] for l in lens.source_layers])  # [L, V]
    # per-layer rank of each token, then min over layers
    order = stack.argsort(dim=-1, descending=True)  # [L, V]
    ranks = torch.empty_like(order)
    ar = torch.arange(stack.shape[-1])
    for i in range(stack.shape[0]):
        ranks[i, order[i]] = ar
    return ranks.min(dim=0).values  # [V] best (lowest) rank across layers


def evaluate(model, lens, items, ks=(1, 5, 10)):
    tok = model.tokenizer
    out = {f"jlens_pass@{k}": [] for k in ks}
    out.update({f"logit_pass@{k}": [] for k in ks})
    for it in items:
        for use_j, pre in ((True, "jlens"), (False, "logit")):
            best = ranks_at_last_pos(model, lens, it["prompt"], use_jacobian=use_j)
            fracs = {k: [] for k in ks}
            for w in it["intermediates"]:
                r = int(best[first_token(tok, w)])
                for k in ks:
                    fracs[k].append(1.0 if r < k else 0.0)
            for k in ks:
                out[f"{pre}_pass@{k}"].append(sum(fracs[k]) / max(len(fracs[k]), 1))
    return {m: sum(v) / max(len(v), 1) for m, v in out.items()}


@torch.no_grad()
def _per_layer_logits(model, lens, prompt, *, use_jacobian):
    """{layer: [vocab]} lens logits at the last position."""
    lens_logits, _, _ = lens.apply(
        model, prompt, layers=lens.source_layers, positions=[-1],
        use_jacobian=use_jacobian,
    )
    return {l: lens_logits[l][0] for l in lens.source_layers}


def _norm_logit_diff(logits, id_correct: int, id_wrong: int) -> float:
    """(L_correct - L_wrong) / (|L_correct| + |L_wrong|), in [-1, 1]. A difference,
    not a probability, so a *suppressed* correct form still reads as positive."""
    lc = float(logits[id_correct])
    lw = float(logits[id_wrong])
    return (lc - lw) / (abs(lc) + abs(lw) + 1e-9)


@torch.no_grad()
def evaluate_form(model, lens, items):
    """Does the lens rank the correct surface form above a minimal wrong form,
    given the model itself gets the clean case right? J-lens vs logit-lens.

    Reported: form accuracy (sign of the best-over-layers logit diff), the mean
    diff, and a per-band diff profile to locate where the form is resolved. Only
    items the full model answers correctly are scored (n_clean / n_items)."""
    tok = model.tokenizer
    layers = lens.source_layers
    bands = band_layers(layers, model.n_layers)

    per_lens = {}
    for use_j, pre in ((True, "jlens"), (False, "logit")):
        by_kind: dict[str, list[float]] = {}
        band_diffs: dict[str, list[float]] = {b: [] for b in BANDS}
        best_diffs, n_clean = [], 0
        for it in items:
            ic = first_token(tok, it["form_correct"])
            iw = first_token(tok, it["form_wrong"])
            if ic == iw:
                continue  # not first-token distinguishable; skip (report coverage)
            # behavioural screen on the FULL model, once (lens-independent)
            if use_j:
                it["_clean"] = greedy_next(model, it["prompt"]) == ic
            if not it.get("_clean"):
                continue
            n_clean += 1
            logits = _per_layer_logits(model, lens, it["prompt"], use_jacobian=use_j)
            diffs = {l: _norm_logit_diff(logits[l], ic, iw) for l in layers}
            best = max(diffs.values())
            best_diffs.append(best)
            by_kind.setdefault(it["kind"], []).append(1.0 if best > 0 else 0.0)
            for b, ls in bands.items():
                if ls:
                    band_diffs[b].append(max(diffs[l] for l in ls))
        per_lens[pre] = {
            "form_acc": sum(1.0 for d in best_diffs if d > 0) / max(len(best_diffs), 1),
            "form_logitdiff_mean": sum(best_diffs) / max(len(best_diffs), 1),
            "n_clean": n_clean,
            "by_kind_acc": {k: sum(v) / len(v) for k, v in by_kind.items()},
            "band_logitdiff_mean": {
                b: (sum(v) / len(v) if v else None) for b, v in band_diffs.items()
            },
        }
    return per_lens


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--set", default="lens-eval-multihop")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    # vendored eval sets live under jacobian-lens/data/evaluations; our authored
    # morphosyntax sets live under this repo's data/morphosyntax.
    for base in (DATA, ROOT / "data" / "morphosyntax"):
        cand = base / f"{args.set}.json"
        if cand.exists():
            items = json.loads(cand.read_text())["items"]
            break
    else:
        raise SystemExit(f"dataset {args.set}.json not found")
    if args.limit:
        items = items[: args.limit]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(key)
    else:
        model = load_model(key)
    lens = JacobianLens.from_pretrained(str(lens_path))

    is_form = "form_correct" in items[0]
    if is_form:
        res = {"eval": "surface-form", "model": tag, "n_items": len(items),
               "lens": lens_path.stem, **evaluate_form(model, lens, items)}
        suffix = "formeval"
    else:
        res = {"eval": "semantic-identity", **evaluate(model, lens, items),
               "model": tag, "n_items": len(items), "lens": lens_path.stem}
        suffix = "lenseval"

    print(json.dumps(res, indent=2))
    out = ROOT / "results" / "metrics" / f"{tag}_{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
