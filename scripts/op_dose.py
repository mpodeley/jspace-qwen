#!/usr/bin/env python
"""Dose-response and collateral damage of the operator intervention.

(A) Efficacy vs dose: sweep the intervention strength alpha and read the mean
    swap-random contrast over all ordered pairs x operands. Shows the effect is
    graded and where the default (alpha=4) sits on the curve.
(B) Collateral damage vs dose: with the SAME hook active, run unrelated WikiText
    prompts and measure the per-token KL(clean || intervened) of the full
    next-token distribution and the change in NLL of the actual continuation.
    A surgical intervention moves the target relation at low collateral cost;
    a blunt one degrades everything.

    python scripts/op_dose.py 1.7b
    python scripts/op_dose.py 1.7b --alphas 1 2 4 8 --n-corpus 12
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from _common import band_layers, evenly_spaced_layers, get_corpus, load_model, resolve_tag
from jlens import ActivationRecorder
import op_core

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def all_pos_logits(model, ids, hook_vecs=None):
    """Full-sequence final-layer logits [seq, vocab], optionally with the
    intervention hook active (same hook construction as op_core.final_logits)."""
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
        h = rec.activations[final][0]  # [seq, d_model]
    for hd in handles:
        hd.remove()
    return model.unembed(h.float())


def collateral(model, prompts, dv, max_len=64):
    """Mean per-token KL(clean || hooked) in nats and mean delta-NLL of the actual
    continuation, over unrelated corpus prompts."""
    kls, dnlls = [], []
    for p in prompts:
        ids = model.encode(p, max_length=max_len)
        Lc = all_pos_logits(model, ids)
        Lh = all_pos_logits(model, ids, dv)
        pc, ph = F.log_softmax(Lc, -1), F.log_softmax(Lh, -1)
        kls.append(float((pc.exp() * (pc - ph)).sum(-1).mean()))
        tgt = ids[0, 1:]
        dnlls.append(float((-ph[:-1].gather(1, tgt[:, None]).mean())
                           - (-pc[:-1].gather(1, tgt[:, None]).mean())))
    return sum(kls) / len(kls), sum(dnlls) / len(dnlls)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--domain", default="relations")
    ap.add_argument("--alphas", nargs="*", type=float,
                    default=[0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0])
    ap.add_argument("--n-corpus", type=int, default=12,
                    help="unrelated WikiText prompts for the collateral measure")
    ap.add_argument("--pairs", nargs="*", default=None,
                    help="from:to pairs for collateral (default: 3 representative)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    dom = op_core.load_domain(args.domain)
    model = load_model(args.model)
    tok = model.tokenizer
    source_layers = evenly_spaced_layers(model.n_layers)
    ws = band_layers(source_layers, model.n_layers)["workspace"]

    # operator directions once; alpha only scales the injected vector
    v = op_core.op_dirs(model, ws, dom)
    ops = dom.op_keys
    pairs = ([tuple(p.split(":")) for p in args.pairs] if args.pairs
             else [(a, b) for a in ops for b in ops if a != b])
    coll_pairs = pairs[:3] if len(pairs) > 3 else pairs
    corpus = [p[:600] for p in get_corpus(args.n_corpus)]

    # clean baseline once per (pair, operand)
    print(f"[{tag} / {args.domain}] dose-response over alphas {args.alphas}")
    clean = {}
    for frm, to in pairs:
        for o in dom.operand_keys:
            af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
            if af == at:
                continue
            L0 = op_core.final_logits(model, dom.render(o, frm))
            clean[(frm, to, o)] = float(L0[at] - L0[af])

    g = torch.Generator().manual_seed(args.seed)
    dev = v[ops[0]][ws[0]].device
    rows = []
    for alpha in args.alphas:
        swaps, rands = [], []
        gg = torch.Generator().manual_seed(args.seed)  # same rv per alpha
        for frm, to in pairs:
            dv = {l: alpha * (v[to][l] - v[frm][l]) for l in ws}
            norm = torch.cat([dv[l] for l in ws]).norm()
            rv = {l: torch.randn(model.d_model, generator=gg).to(dev) for l in ws}
            rn = torch.cat([rv[l] for l in ws]).norm()
            rv = {l: rv[l] / rn * norm for l in ws}
            for o in dom.operand_keys:
                if (frm, to, o) not in clean:
                    continue
                af, at = dom.answer_tok(tok, o, frm), dom.answer_tok(tok, o, to)
                p = dom.render(o, frm)
                Ls = op_core.final_logits(model, p, dv)
                Lr = op_core.final_logits(model, p, rv)
                swaps.append(float(Ls[at] - Ls[af]) - clean[(frm, to, o)])
                rands.append(float(Lr[at] - Lr[af]) - clean[(frm, to, o)])
        # collateral on unrelated text, averaged over a few representative pairs
        kl, dnll = 0.0, 0.0
        for frm, to in coll_pairs:
            dv = {l: alpha * (v[to][l] - v[frm][l]) for l in ws}
            k, d = collateral(model, corpus, dv)
            kl += k / len(coll_pairs)
            dnll += d / len(coll_pairs)
        rows.append({"alpha": alpha,
                     "swap_shift": sum(swaps) / len(swaps),
                     "random_shift": sum(rands) / len(rands),
                     "kl_nats": kl, "dnll_nats": dnll})
        r = rows[-1]
        print(f"  alpha={alpha:5.1f}  swap shift {r['swap_shift']:+7.2f}  "
              f"random {r['random_shift']:+6.2f}  KL {r['kl_nats']:.4f}  "
              f"dNLL {r['dnll_nats']:+.4f}")

    df = pd.DataFrame(rows)
    out = ROOT / "results" / "ablation" / f"{tag}_{args.domain}_dose.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
