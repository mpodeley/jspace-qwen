#!/usr/bin/env python
"""Why does Qwen3-1.7B keep a head it cannot lose, and Qwen3-8B keep none?

§3.2's screen found two load-bearing heads in the small model (L1H5 x19.7, L0H3
x4.4) and zero in the large one. §3.8 asks why, and notes the screen cannot
answer: it ablates one head at a time, and §3.6 shows joint damage is
superadditive. This probe adds the two measurements the screen is missing.

  characterize   What IS the critical head? Its attention pattern over real text:
                 an attention sink attends almost everything to position 0 (the
                 BOS / first token), a documented load-bearing pattern (Xiao et
                 al., attention sinks). We measure the fraction of attention mass
                 each head sends to position 0, and whether the critical heads are
                 the extreme sinks -- and whether the large model has heads that
                 are just as sink-like but individually dispensable.

  joint          Is the large model's robustness real, or does it just spread the
                 same single point of failure across a SET? A greedy search: ablate
                 the head whose removal (on top of those already removed) costs the
                 most perplexity, repeat. If 8B has no k-head set whose joint
                 ablation is catastrophic, the function L1H5 performs is not merely
                 distributed -- it is absent or unnecessary at scale.

    python scripts/lesion_critical_probe.py 1.7b
    python scripts/lesion_critical_probe.py 8b --joint-k 12
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from _common import get_corpus, load_model, resolve_tag
from lesion_core import Lesion, perplexity, reference_means, unit_dims

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def sink_fractions(model, prompts, dims):
    """Per-head fraction of attention mass on position 0, averaged over tokens and
    prompts. [n_layers, n_heads].

    The model is loaded with SDPA ("flash"), which never materializes the
    attention matrix -- output_attentions is silently ignored and returns None,
    which an earlier version summed as zeros for every head. We force the eager
    path for this measurement (config._attn_implementation is read per forward),
    restore it after, and GUARD loudly if the weights still fail to appear."""
    hf = model._hf_model
    old = hf.config._attn_implementation
    hf.config._attn_implementation = "eager"
    try:
        frac = torch.zeros(dims.n_layers, dims.n_heads)
        ntok = 0
        for p in prompts:
            ids = model.encode(p, max_length=256)
            out = hf(input_ids=ids, use_cache=False, output_attentions=True)
            if out.attentions is None or out.attentions[0] is None:
                raise RuntimeError(
                    "no attention weights returned even under eager -- the sink "
                    "measurement cannot run on this attention implementation")
            # attentions: tuple[n_layers] of [batch, n_heads, q, k]
            for l, a in enumerate(out.attentions):
                frac[l] += a[0, :, :, 0].sum(-1).float().cpu()   # mass on key pos 0
            ntok += ids.shape[1]
    finally:
        hf.config._attn_implementation = old
    return frac / ntok


def greedy_joint(model, prompts, means, dims, k, seed=0):
    """Greedily grow a head set by marginal perplexity damage. Returns the ordered
    picks and the perplexity after each. If even the greedy set stays cheap, no
    small set is catastrophic."""
    base = perplexity(model, prompts)
    chosen, trace = [], []
    remaining = [(l, h) for l in range(dims.n_layers) for h in range(dims.n_heads)]
    for step in range(k):
        best, best_ppl = None, -1.0
        for lh in remaining:
            with Lesion(model, heads=chosen + [lh], mode="mean", means=means,
                        dims=dims):
                p = perplexity(model, prompts)
            if p > best_ppl:
                best, best_ppl = lh, p
        chosen.append(best)
        remaining.remove(best)
        trace.append({"step": step + 1, "added": f"L{best[0]}H{best[1]}",
                      "ppl": best_ppl, "ppl_ratio": best_ppl / base})
        print(f"  +L{best[0]}H{best[1]}: ppl x{best_ppl / base:.2f}")
    return base, chosen, trace


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--sink-prompts", type=int, default=8)
    ap.add_argument("--joint-k", type=int, default=8)
    ap.add_argument("--joint-prompts", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tag = resolve_tag(args.model)
    model = load_model(args.model)
    dims = unit_dims(model)
    corpus = get_corpus(128)
    means = reference_means(model, corpus[:24], dims=dims)

    crit_path = ROOT / "results" / "lesion" / f"{tag}_criticality_summary.json"
    critical = []
    if crit_path.exists():
        critical = [(c["layer"], c["head"], c["ppl_ratio"])
                    for c in json.loads(crit_path.read_text())["critical_heads"]]
    print(f"[{tag}] critical heads (from the screen): "
          f"{[f'L{l}H{h}(x{r:.1f})' for l, h, r in critical] or 'none'}")

    # ---- characterize: attention-sink fractions ----
    print(f"\nattention-sink fractions over {args.sink_prompts} prompts...")
    frac = sink_fractions(model, corpus[:args.sink_prompts], dims)
    flat = [(float(frac[l, h]), l, h)
            for l in range(dims.n_layers) for h in range(dims.n_heads)]
    flat.sort(reverse=True)
    print("  top-8 sink heads (fraction of attention on position 0):")
    crit_set = {(l, h) for l, h, _ in critical}
    for v, l, h in flat[:8]:
        mark = "  <-- CRITICAL" if (l, h) in crit_set else ""
        print(f"    L{l}H{h}: {v:.1%}{mark}")
    for l, h, r in critical:
        rank = next(i for i, (_, ll, hh) in enumerate(flat) if (ll, hh) == (l, h))
        print(f"  critical L{l}H{h}: sink fraction {float(frac[l, h]):.1%}, "
              f"rank {rank + 1} of {len(flat)} heads")

    # ---- joint: greedy set-of-heads search ----
    print(f"\ngreedy joint-ablation search, up to {args.joint_k} heads, "
          f"{args.joint_prompts} prompts...")
    base, chosen, trace = greedy_joint(model, corpus[:args.joint_prompts], means,
                                       dims, args.joint_k, seed=args.seed)

    out = {"tag": tag,
           "critical_heads": [{"layer": l, "head": h, "ppl_ratio": r}
                              for l, h, r in critical],
           "sink_top8": [{"layer": l, "head": h, "sink_fraction": v,
                          "is_critical": (l, h) in crit_set} for v, l, h in flat[:8]],
           "critical_sink_fraction": {f"L{l}H{h}": float(frac[l, h])
                                      for l, h, _ in critical},
           "max_sink_fraction": flat[0][0],
           "greedy_joint": {"base_ppl": base, "trace": trace,
                            "max_ppl_ratio": max(t["ppl_ratio"] for t in trace)},
           "_meta": {"note": "sink fraction = mean attention mass on key position "
                             "0 per head. greedy_joint grows a head set by marginal "
                             "ppl damage; if it stays cheap, no small set is "
                             "catastrophic and the small model's single point of "
                             "failure has no large-model analogue."}}
    p = ROOT / "results" / "lesion" / f"{tag}_critical_probe.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\ngreedy set of {args.joint_k} heads reaches ppl "
          f"x{out['greedy_joint']['max_ppl_ratio']:.2f}")
    print(f"saved {p}")


if __name__ == "__main__":
    main()
