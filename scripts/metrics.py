#!/usr/bin/env python
"""Per-layer J-space metrics for one model+lens, for the scale study.

Computes, per fitted source layer (reindexed to 0-100 depth):

  jlens_top1 / jlens_top5 / logitlens_top1  next-token accuracy on held-out text
      -> where the model becomes "poised to say" the next token; J-lens vs the
         vanilla logit-lens baseline.
  kurtosis        excess kurtosis of the lens logit vector (heavy tail = a few
                  concepts dominate; the "workspace" signature).
  eff_dim         participation ratio of J_l's singular values (effective linear
                  dimensionality of the readout).
  read_var_frac   fraction of residual variance lying in J_l's top-k=25 read
                  directions (the paper's "<10% of variance" J-space claim).
  top1_autocorr   fraction of adjacent positions sharing the J-lens top-1 token
                  (persistence of abstract content across positions).

Writes results/metrics/<tag>.parquet (one row per layer).

    python scripts/metrics.py 1.7b
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from _common import MODELS, depth_percent, load_model, resolve_tag
from jlens import ActivationRecorder, JacobianLens
from jlens.fitting import valid_position_mask

ROOT = Path(__file__).resolve().parent.parent
JSPACE_K = 25  # paper's sparsity: # of J-lens vectors meaningfully active at once


def excess_kurtosis(x: torch.Tensor) -> float:
    x = x.float()
    m = x.mean()
    d = x - m
    var = (d * d).mean()
    if var <= 0:
        return 0.0
    return float((d.pow(4).mean() / var.pow(2)) - 3.0)


@torch.no_grad()
def compute(model, lens: JacobianLens, prompts: list[str], *, max_pos_var: int = 24):
    layers = lens.source_layers
    final = model.n_layers - 1
    tok = model.tokenizer

    # scalar accumulators per layer
    acc = {l: dict(j1=0, j5=0, l1=0, n=0, kurt=0.0, kn=0, ac_hit=0, ac_n=0)
           for l in layers}
    act_samples: dict[int, list[torch.Tensor]] = {l: [] for l in layers}

    for prompt in prompts:
        input_ids = model.encode(prompt, max_length=128)
        seq_len = input_ids.shape[1]
        if seq_len < 4:
            continue
        mask = valid_position_mask(seq_len)
        vpos = mask.nonzero(as_tuple=True)[0]
        targets = input_ids[0, 1:]  # next-token labels

        with ActivationRecorder(model.layers, at=[*layers, final]) as rec:
            model.forward(input_ids)
            acts = {l: rec.activations[l][0].detach() for l in [*layers, final]}

        for l in layers:
            h = acts[l].float()  # [seq, d]
            trans = lens.transport(h, l)  # J_l @ h  -> [seq, d]
            logits = model.unembed(trans).float().cpu()  # [seq, vocab]
            ll = model.unembed(h).float().cpu()  # logit-lens baseline

            top5 = logits[vpos].topk(5, dim=-1).indices  # [nv,5]
            tgt = targets[vpos].cpu()  # aligned: position p predicts token p+1
            a = acc[l]
            a["j1"] += int((top5[:, 0] == tgt).sum())
            a["j5"] += int((top5 == tgt[:, None]).any(dim=1).sum())
            a["l1"] += int((ll[vpos].argmax(-1) == tgt).sum())
            a["n"] += len(vpos)

            # kurtosis over vocab, averaged over a few positions
            for p in vpos[:max_pos_var].tolist():
                a["kurt"] += excess_kurtosis(logits[p]); a["kn"] += 1

            # top-1 autocorrelation across adjacent valid positions
            t1 = logits[vpos].argmax(-1)
            a["ac_hit"] += int((t1[1:] == t1[:-1]).sum()); a["ac_n"] += max(len(t1) - 1, 0)

            # subsample raw activations for the variance metric
            sub = vpos[torch.linspace(0, len(vpos) - 1, min(max_pos_var, len(vpos))).long()]
            act_samples[l].append(h[sub].cpu())

    rows = []
    for l in layers:
        a = acc[l]
        J = lens.jacobians[l].float()
        s = torch.linalg.svdvals(J)
        eff_dim = float(s.sum() ** 2 / (s.pow(2).sum() + 1e-9))  # participation ratio

        H = torch.cat(act_samples[l], 0)  # [N, d]
        Hc = H - H.mean(0, keepdim=True)
        total_var = float(Hc.pow(2).sum())
        # top-k read directions = right singular vectors of J_l (what the lens reads)
        Vt = torch.linalg.svd(J, full_matrices=False).Vh[:JSPACE_K]  # [k, d]
        proj = Hc @ Vt.T  # [N, k]
        read_var_frac = float(proj.pow(2).sum() / (total_var + 1e-9))

        rows.append(dict(
            layer=l, depth=round(depth_percent(l, model.n_layers), 1),
            jlens_top1=a["j1"] / max(a["n"], 1),
            jlens_top5=a["j5"] / max(a["n"], 1),
            logitlens_top1=a["l1"] / max(a["n"], 1),
            kurtosis=a["kurt"] / max(a["kn"], 1),
            eff_dim=eff_dim,
            read_var_frac=read_var_frac,
            top1_autocorr=a["ac_hit"] / max(a["ac_n"], 1),
        ))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--lens", default=None)
    ap.add_argument("--n-eval", type=int, default=8)
    ap.add_argument("--int8", action="store_true", help="load 32B via int8 quantizer")
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    from _common import get_corpus
    # held-out eval slice (fitting used prompts[:64])
    prompts = get_corpus(128)[64 : 64 + args.n_eval]

    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(key)
    else:
        model = load_model(key)
    lens = JacobianLens.from_pretrained(str(lens_path))
    df = compute(model, lens, prompts)

    out = ROOT / "results" / "metrics" / f"{tag}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    pd.set_option("display.width", 140, "display.max_columns", 20)
    print(df.to_string(index=False))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
