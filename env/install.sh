#!/usr/bin/env bash
# Reproducible environment for jspace-qwen on AMD Strix Halo (Radeon 8060S, gfx1151).
#
# No distrobox / no ROCm system install needed: the host kernel already exposes
# the GPU compute nodes (/dev/kfd and /dev/dri/renderD128 are world-accessible),
# and the ROCm userspace ships *inside* the PyTorch wheels (TheRock gfx1151
# nightly index), pulled in as the `rocm[libraries]` pip package.
#
# Key gotchas encountered while building this:
#   - Do NOT add PyPI as an equal index for torch: uv picks the higher version
#     number, which is the CUDA build (torch 2.13.0 + nvidia-*). Install torch
#     from the gfx1151 index *only*.
#   - torch needs `rocm[libraries]==<same a-version>` and a matching
#     `triton==...+rocm...`, both alpha builds -> require
#     --index-strategy unsafe-best-match --prerelease=allow
#     (PyPI has a decoy `rocm` package that trips uv's confusion guard).
#   - bitsandbytes / torchao are unsupported on gfx1151 -> stay in bf16.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
GFX_INDEX="https://rocm.nightlies.amd.com/v2/gfx1151"
ROCM_VER="7.13.0a20260513"   # pin: torch build == rocm build == triton rocm build

cd "$ROOT"
uv venv --python 3.12 .venv
export VIRTUAL_ENV="$ROOT/.venv"

# 1. ROCm PyTorch (gfx1151 index ONLY, no-deps so PyPI can't shadow it)
uv pip install --no-deps torch --index-url "$GFX_INDEX/"

# 2. ROCm runtime libraries + matching triton (alpha, cross-index best match)
uv pip install "rocm[libraries]==$ROCM_VER" "triton==3.6.0+rocm$ROCM_VER" \
  --index-url "$GFX_INDEX/" --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match --prerelease=allow

# 3. Generic torch runtime deps + the jlens stack (all from PyPI)
uv pip install filelock typing-extensions sympy networkx jinja2 fsspec setuptools \
  numpy "transformers>=5.5" huggingface_hub datasets accelerate safetensors \
  sentencepiece pandas pyarrow
uv pip install --no-deps -e ./jacobian-lens

echo
echo "Verifying GPU is visible ..."
.venv/bin/python - <<'PY'
import torch
assert torch.cuda.is_available(), "ROCm GPU not visible"
p = torch.cuda.get_device_properties(0)
print("OK:", torch.__version__, "| hip", torch.version.hip,
      "|", torch.cuda.get_device_name(0), getattr(p, "gcnArchName", "?"),
      "|", round(p.total_memory / 2**30, 1), "GiB VRAM")
PY
echo "Done. Run: .venv/bin/python scripts/00_smoke.py"
