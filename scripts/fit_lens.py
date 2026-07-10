#!/usr/bin/env python
"""Fit a Jacobian lens on a Qwen3 model and save it.

Fits ~25 evenly spaced source layers (the paper's convention) over a
WikiText-103 corpus, with resumable checkpointing. Cost is dominated by the
model backward pass: ceil(d_model/dim_batch) backward passes per prompt.

    python scripts/fit_lens.py 1.7b --n-prompts 64
    python scripts/fit_lens.py 8b  --n-prompts 100 --dim-batch 16
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

import jlens
from _common import MODELS, evenly_spaced_layers, get_corpus, load_model, resolve_tag

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="key (1.7b/8b/32b) or HF id")
    ap.add_argument("--n-prompts", type=int, default=64)
    ap.add_argument("--k-layers", type=int, default=25)
    ap.add_argument("--dim-batch", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=128)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--int8", action="store_true", help="fit on the int8 model (control)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    key = args.model
    tag = resolve_tag(key, int8=args.int8)
    out = Path(args.out) if args.out else ROOT / "out" / "lenses" / f"{tag}.pt"
    ckpt = out.with_suffix(".ckpt.pt")
    out.parent.mkdir(parents=True, exist_ok=True)

    jlens.configure_logging()
    print(f"loading {MODELS.get(key, key)} {'(int8)' if args.int8 else ''} ...", flush=True)
    if args.int8:
        from int8_model import load_int8_model
        model = load_int8_model(key, report=True)  # compile not composed with int8
    else:
        model = load_model(key, compile=args.compile)
    layers = evenly_spaced_layers(model.n_layers, args.k_layers)
    print(f"  {model!r}", flush=True)
    print(f"  fitting {len(layers)} source layers: {layers}", flush=True)
    print(f"  VRAM after load: {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    print(f"fetching {args.n_prompts} WikiText prompts ...", flush=True)
    prompts = get_corpus(args.n_prompts)
    print(f"  got {len(prompts)} prompts", flush=True)

    t0 = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=layers,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        checkpoint_path=str(ckpt),
        checkpoint_every=4,
    )
    dt = time.perf_counter() - t0
    lens.save(str(out))
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"saved {out}  ({dt:.0f}s, {dt/max(len(prompts),1):.1f}s/prompt, "
          f"peak {peak:.2f} GiB)", flush=True)
    print(f"  {lens!r}", flush=True)


if __name__ == "__main__":
    main()
