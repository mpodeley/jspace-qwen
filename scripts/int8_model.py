#!/usr/bin/env python
"""Weight-only int8 loader for large Qwen3 (so 32B fits the 48 GB VRAM wall).

bitsandbytes/torchao don't run on gfx1151, so we do our own symmetric
per-output-channel int8 quantization with a custom autograd Function.

Why a custom Function (not just `x @ (w.to(bf16)*scale).T`)? The J-lens fit
retains one graph and runs ceil(d_model/dim_batch) backward passes against it.
A naive dequant would store the *bf16* weight in that retained graph for every
linear -> ~64 GB, defeating the point. `Int8LinearFn` instead saves only the
int8 weight (by reference, no copy) and re-dequantizes inside backward, so the
graph holds int8 (~32 GB) + activations. All the J-lens needs is grad wrt the
activation; weights are frozen.

Weights are streamed and quantized straight from the safetensors shards, so we
never materialize the full 64 GB bf16 model (won't fit in 48 GB system RAM).
Embeddings / lm_head / all norms stay bf16 for a faithful readout.
"""

from __future__ import annotations

import re
from typing import Any

import torch
import transformers
from torch import nn

import jlens

# block linear weights we quantize: attention + MLP projections
_QUANT_RE = re.compile(
    r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


class Int8LinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_int8, scale, bias):
        ctx.save_for_backward(w_int8, scale)
        w = w_int8.to(x.dtype) * scale  # [out, in], transient
        y = x @ w.t()
        if bias is not None:
            y = y + bias
        return y

    @staticmethod
    def backward(ctx, grad_y):
        w_int8, scale = ctx.saved_tensors
        w = w_int8.to(grad_y.dtype) * scale  # recomputed, freed after
        grad_x = grad_y @ w  # [.., out] @ [out, in] -> [.., in]
        return grad_x, None, None, None


class Int8Linear(nn.Module):
    def __init__(self, out_features: int, in_features: int, bias: bool):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer(
            "weight_int8", torch.empty(out_features, in_features, dtype=torch.int8)
        )
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.bfloat16))
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=torch.bfloat16) if bias else None
        )

    def load_bf16_weight(self, w: torch.Tensor) -> None:
        """Quantize a bf16 [out,in] weight into this module's buffers."""
        w = w.float()
        scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        q = (w / scale).round().clamp_(-127, 127).to(torch.int8)
        self.weight_int8.copy_(q)
        self.scale.copy_(scale.to(torch.bfloat16))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Int8LinearFn.apply(x, self.weight_int8, self.scale, self.bias)


def quantize_error(w: torch.Tensor) -> float:
    """Relative Frobenius error of round-tripping a weight through int8."""
    w = w.float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
    deq = (w / scale).round().clamp_(-127, 127) * scale
    return float((deq - w).norm() / (w.norm() + 1e-9))


def _shard_paths(name: str) -> list[str]:
    """Local safetensors shard paths for a model (downloads if needed)."""
    import json
    from pathlib import Path

    from huggingface_hub import hf_hub_download, snapshot_download

    idx = None
    try:
        idx = hf_hub_download(name, "model.safetensors.index.json")
    except Exception:
        pass
    if idx is None:  # single-shard model
        return [hf_hub_download(name, "model.safetensors")]
    d = snapshot_download(name, allow_patterns=["*.safetensors"])
    files = sorted(set(json.loads(Path(idx).read_text())["weight_map"].values()))
    return [str(Path(d) / f) for f in files]


def load_int8_model(
    name_or_key: str, *, device: str = "cuda", report: bool = False
) -> jlens.HFLensModel:
    from accelerate import init_empty_weights

    from _common import MODELS

    name = MODELS.get(name_or_key, name_or_key)
    config = transformers.AutoConfig.from_pretrained(name)
    with init_empty_weights():
        hf = transformers.AutoModelForCausalLM.from_config(config, dtype=torch.bfloat16)

    # 1. swap block Linears for empty Int8Linear (real buffers, on device)
    int8_mods: dict[str, Int8Linear] = {}
    for mod_name, mod in list(hf.named_modules()):
        if isinstance(mod, nn.Linear) and re.search(
            r"\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)$",
            mod_name,
        ):
            q = Int8Linear(mod.out_features, mod.in_features, mod.bias is not None)
            q.weight_int8 = q.weight_int8.to(device)
            q.scale = q.scale.to(device)
            if q.bias is not None:
                q.bias = q.bias.to(device)
            parent = hf.get_submodule(mod_name.rsplit(".", 1)[0])
            setattr(parent, mod_name.rsplit(".", 1)[1], q)
            int8_mods[mod_name + ".weight"] = q

    # 2. stream shards: quantize block linears, materialize everything else bf16
    from safetensors import safe_open

    n_q = 0
    errs = []
    for path in _shard_paths(name):
        with safe_open(path, framework="pt", device=device) as f:
            for key in f.keys():
                t = f.get_tensor(key)
                if _QUANT_RE.search(key):
                    qmod = int8_mods[key]
                    if report and n_q % 64 == 0:
                        errs.append(quantize_error(t))
                    qmod.load_bf16_weight(t)
                    n_q += 1
                else:
                    # materialize a meta param/buffer (norms, embed, lm_head, q/k_norm)
                    mod_path, leaf = key.rsplit(".", 1)
                    mod = hf.get_submodule(mod_path)
                    tb = t.to(torch.bfloat16)
                    if leaf in mod._parameters:
                        mod._parameters[leaf] = nn.Parameter(tb, requires_grad=False)
                    else:
                        mod._buffers[leaf] = tb
    # tied embeddings: if lm_head wasn't in the checkpoint, tie to embed_tokens
    if hf.get_output_embeddings().weight.is_meta:
        hf.get_output_embeddings().weight = hf.get_input_embeddings().weight

    if report:
        vram = torch.cuda.memory_allocated() / 2**30
        print(f"int8: quantized {n_q} linears | mean rel-err "
              f"{sum(errs)/max(len(errs),1):.4f} | VRAM {vram:.1f} GiB", flush=True)

    return jlens.from_hf(hf, transformers.AutoTokenizer.from_pretrained(name))


if __name__ == "__main__":
    import sys

    model = load_int8_model(sys.argv[1] if len(sys.argv) > 1 else "8b", report=True)
    print(repr(model))
