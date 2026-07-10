# Reproducible setup — AMD Strix Halo (Radeon 8060S, gfx1151)

The J-lens needs **autograd on the GPU** (it is built from backward passes), so
inference-only stacks like llama.cpp/Vulkan do not apply. This project runs on
an AMD **Ryzen AI Max+ "Strix Halo"** APU (Radeon 8060S iGPU, `gfx1151`) under
Fedora Aurora — no NVIDIA, no CUDA.

## No distrobox, no system ROCm

The host kernel already exposes the GPU compute nodes (`/dev/kfd` and
`/dev/dri/renderD128` are world-accessible), and the ROCm userspace ships
*inside* the PyTorch wheels from AMD's TheRock gfx1151 nightly index. So a plain
`uv` virtualenv is enough — the full recipe is in `env/install.sh`.

```bash
bash env/install.sh          # creates .venv, installs ROCm torch + jlens stack
.venv/bin/python scripts/00_smoke.py   # forward+backward of Qwen3-1.7B on the iGPU
```

### Gotchas worth knowing

- **Install torch from the gfx1151 index only.** If you add PyPI as an equal
  index, `uv` picks the higher *version number*, which is the CUDA build
  (`torch 2.13.0` + `nvidia-*`) — useless on AMD.
- torch depends on `rocm[libraries]==<same version>` and a matching
  `triton==...+rocm...`, both alpha builds, so you need
  `--index-strategy unsafe-best-match --prerelease=allow` (PyPI ships a decoy
  `rocm` package that trips uv's dependency-confusion guard).
- **Stay in bf16.** `bitsandbytes` and `torchao` do not run on gfx1151.
- Verified stack: `torch 2.10.0+rocm7.13`, HIP 7.13, `gfx1151`, autograd OK.

## The 48 GB memory wall (and how 32B still fits)

The machine's 96 GB is BIOS-split into **48 GB system RAM + 48 GB dedicated
VRAM**. Measured behaviour: PyTorch/HIP hits a hard **48 GB** ceiling — the
~23 GB of GTT is not used automatically, so despite the "unified" firmware
label there is no spill past the VRAM carveout without a reboot.

Consequences for the three-model sweep:

| model | bf16 weights | fits 48 GB VRAM? | how we run it |
|---|---|---|---|
| Qwen3-1.7B | ~3.4 GB | yes | bf16 |
| Qwen3-8B | ~16 GB | yes | bf16 |
| Qwen3-32B | ~64 GB | **no** | **custom int8** |

For 32B we use a **weight-only int8** quantizer (`scripts/int8_model.py`).
Because the J-lens only needs gradients *with respect to activations* (weights
are frozen), a symmetric per-output-channel int8 weight with an on-the-fly bf16
dequantization is fully differentiable in the way the lens requires. Two details
make it work on this hardware:

1. **A custom `autograd.Function`** re-dequantizes the weight inside `backward`
   from the persistent int8 buffers, so the retained multi-pass Jacobian graph
   stores int8 (~32 GB), not bf16 (~64 GB) — otherwise the saving is lost.
2. Weights are **stream-quantized straight from the safetensors shards**, so the
   full 64 GB bf16 model is never materialized (it would not fit in 48 GB of
   system RAM). Embeddings, the LM head, and all norms stay bf16 for a faithful
   readout.

This keeps the flagship 32B in the study without a BIOS reboot, at the cost of a
mild quantization confound (we measure the int8 model, not the exact bf16 one).
