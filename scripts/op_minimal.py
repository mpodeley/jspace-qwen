#!/usr/bin/env python
"""Answers the "you injected over half the network" objection, three ways.

(A) Band-width sweep: the same operator-difference injection applied over the
    full workspace band, its middle half, and a SINGLE mid-workspace layer --
    reporting margin flips, greedy exact-match answers, and off-task KL per
    condition. If a single layer suffices at reasonable cost, the effect is not
    an artifact of blanketing the network.
(B) Real-activation patching: instead of adding a mean-difference vector,
    REPLACE the query-position residual with the activation from a real donor
    prompt (same operand, target operator) at the same layers. No averaged
    direction at all -- classic activation patching.
(C) Greedy exact match: for every condition, greedy-decode 3 tokens and check
    the target answer string appears -- "the answer actually changes", not just
    the pairwise logit margin.

    python scripts/op_minimal.py 1.7b
    python scripts/op_minimal.py 1.7b --no-kl        # skip the KL pass
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from _common import band_layers, evenly_spaced_layers, load_model, resolve_tag
from jlens import ActivationRecorder
import op_core

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def greedy(model, prompt, k=3, add=None, patch=None, add_positions=None):
    """Greedy-decode k tokens. `add`: {layer: vec} added at every position, or --
    when `add_positions` is given -- only at those prompt indices (negatives
    resolved against the prompt length once, so they stay pinned to the same
    tokens as the sequence grows). `patch`: {layer: vec} REPLACING the residual
    at the last prompt position."""
    ids = model.encode(prompt, max_length=64)
    n_prompt = ids.shape[1]
    pos = None if add_positions is None else [p % n_prompt for p in add_positions]
    out = []
    for _ in range(k):
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
                    a[:, n_prompt - 1, :] = vec.to(a.dtype)  # the query position
                    return (a, *o[1:]) if isinstance(o, tuple) else a
                return h
            handles.append(model.layers[l].register_forward_hook(mk_patch(v)))
        final = model.n_layers - 1
        with ActivationRecorder(model.layers, at=[final]) as rec:
            model.forward(ids)
            h = rec.activations[final][0, -1]
        for hd in handles:
            hd.remove()
        nxt = int(model.unembed(h[None].float())[0].argmax())
        out.append(nxt)
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return model.tokenizer.decode(out)


def hit(text: str, answer: str) -> bool:
    return answer.strip().lower() in text.strip().lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--no-kl", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]
    q = len(ws) // 4
    bands = {"full band": ws, "half band": ws[q:q + max(1, len(ws) // 2)],
             "single layer": [ws[len(ws) // 2]]}
    print(f"[{tag}] bands: " + "; ".join(f"{k}={v[0]}..{v[-1]} ({len(v)} layers)"
                                          for k, v in bands.items()))

    v = op_core.op_dirs(model, ws, dom)
    ops = dom.op_keys
    pairs = [(a, b) for a in ops for b in ops if a != b]
    g = torch.Generator().manual_seed(args.seed)
    dev = v[ops[0]][ws[0]].device

    rows = []
    for frm, to in pairs:
        for o in dom.operand_keys:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            p = dom.render(o, frm)
            ansA, ansB = str(dom.answer(o, frm)), str(dom.answer(o, to))
            rec = {"from": frm, "to": to, "operand": o,
                   "clean": hit(greedy(model, p), ansA)}
            for bname, bl in bands.items():
                dv = {l: args.alpha * (v[to][l] - v[frm][l]) for l in bl}
                rec[bname] = hit(greedy(model, p, add=dv), ansB)
            # matched-norm random control on the full band
            dv_full = {l: args.alpha * (v[to][l] - v[frm][l]) for l in ws}
            norm = torch.cat([dv_full[l] for l in ws]).norm()
            rv = {l: torch.randn(model.d_model, generator=g).to(dev) for l in ws}
            rn = torch.cat([rv[l] for l in ws]).norm()
            rv = {l: rv[l] / rn * norm for l in ws}
            rec["random"] = hit(greedy(model, p, add=rv), ansB)
            # real-activation patch: donor = same operand, target operator
            donor = op_core.resid(model, ws, dom.render(o, to), -1)
            rec["activation patch"] = hit(greedy(model, p, patch=donor), ansB)
            rows.append(rec)

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\ngreedy exact-match rates over {n} (pair, operand) items, k=3 tokens:")
    print(f"  clean says answer_A:        {df['clean'].mean():.2f}")
    for c in ("full band", "half band", "single layer", "activation patch", "random"):
        print(f"  {c:<26} says answer_B: {df[c].mean():.2f}")

    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_minimal.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"saved {out}")

    print("\nmargin flips per band (operator-level cluster-bootstrap CIs):")
    margins = {}
    for bname, bl in bands.items():
        dirs = {k: {l: v[k][l] for l in bl} for k in ops}
        ldf = op_core.measure_swaps_long(model, bl, dom, tok, seed=args.seed, dirs=dirs)
        wide = op_core.measure_swaps(model, bl, dom, tok, long_df=ldf)
        fam = op_core.bootstrap_family_ci(ldf, seed=args.seed)
        valid = wide.dropna(subset=["swap"])
        margins[bname] = {"layers": len(bl),
                          "flips": f"{int((valid['swap'] > 0).sum())}/{len(valid)}",
                          "contrast": round(fam["contrast_mean"], 2),
                          "lo": round(fam["contrast_lo"], 2),
                          "hi": round(fam["contrast_hi"], 2)}
        m = margins[bname]
        print(f"  {bname:<14} {m['flips']}  contrast {m['contrast']:+.2f} "
              f"[{m['lo']:+.2f}, {m['hi']:+.2f}]")

    if not args.no_kl:
        from op_dose import collateral
        from _common import get_corpus
        corpus = [c[:600] for c in get_corpus(8)]
        print("\noff-task KL (nats/token, mean over 2 pairs, 8 WikiText prompts):")
        for bname, bl in bands.items():
            kl = 0.0
            for frm, to in pairs[:2]:
                dv = {l: args.alpha * (v[to][l] - v[frm][l]) for l in bl}
                k, _ = collateral(model, corpus, dv)
                kl += k / 2
            margins[bname]["kl_offtask"] = round(kl, 3)
            print(f"  {bname:<14} KL {kl:.3f}")

    import json
    mj = out.with_name(out.name.replace("_minimal.parquet", "_minimal_margins.json"))
    mj.write_text(json.dumps({
        "model": tag, "domain": args.domain, "alpha": args.alpha, "bands": margins,
        "activation_patch": {"positions": 1,
                             "greedy_answerB": round(float(df["activation patch"].mean()), 3),
                             "clean_greedy_answerA": round(float(df["clean"].mean()), 3)},
        "random_control": {"greedy_answerB": round(float(df["random"].mean()), 3)},
        "greedy_by_band": {c: round(float(df[c].mean()), 3)
                           for c in ("full band", "half band", "single layer")},
    }, indent=2))
    print(f"saved {mj}")


if __name__ == "__main__":
    main()
