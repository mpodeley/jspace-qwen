#!/usr/bin/env python
"""Phase 0 smoke test: real Qwen3 forward+backward on the gfx1151 iGPU.

Confirms (a) the model loads in bf16 on the ROCm GPU, (b) jlens.from_hf finds
the Qwen layout, and (c) jacobian_for_prompt produces a finite, non-zero J_l
via autograd through the actual model. This is the Phase 0 exit criterion.
"""

import sys
import time

import torch
import transformers

import jlens

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-1.7B"


def main() -> None:
    assert torch.cuda.is_available(), "ROCm GPU not visible to torch"
    print(f"loading {MODEL} in bf16 on cuda ...", flush=True)
    t0 = time.perf_counter()
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL)
    model = jlens.from_hf(hf, tok)
    print(f"  loaded in {time.perf_counter()-t0:.0f}s -> {model!r}", flush=True)
    print(f"  weights VRAM: {torch.cuda.memory_allocated()/2**30:.2f} GiB", flush=True)

    mid = model.n_layers // 2
    prompt = (
        "The history of the Roman Empire spans over a thousand years, from the "
        "founding of the city to the fall of its western half in the fifth century. "
        "Its institutions, roads, and law shaped the whole of the Mediterranean world."
    )
    print(f"computing J_l for source layer {mid} (dim_batch=8) ...", flush=True)
    t0 = time.perf_counter()
    jacs, seq_len, n_valid = jlens.jacobian_for_prompt(
        model, prompt, source_layers=[mid], dim_batch=8, max_seq_len=128
    )
    J = jacs[mid]
    dt = time.perf_counter() - t0
    print(f"  seq_len={seq_len} n_valid={n_valid}  took {dt:.0f}s", flush=True)
    print(f"  J shape={tuple(J.shape)} finite={torch.isfinite(J).all().item()} "
          f"||J||/sqrt(d)={J.norm().item()/model.d_model**0.5:.3f} "
          f"nonzero={J.abs().sum().item()>0}", flush=True)
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"  peak VRAM: {peak:.2f} GiB", flush=True)
    ok = torch.isfinite(J).all().item() and J.abs().sum().item() > 0
    print("SMOKE:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
