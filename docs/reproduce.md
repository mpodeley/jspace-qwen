# How to reproduce

Everything runs on a single AMD Strix Halo (Radeon 8060S, gfx1151); no CUDA, no
cloud. See [setup](setup.md) for why the environment is a plain `uv` venv.

## 1. Environment

```bash
git clone <this repo> && cd jspace-qwen
git clone https://github.com/anthropics/jacobian-lens
bash env/install.sh                     # ROCm torch + jlens stack
.venv/bin/python scripts/00_smoke.py    # forward+backward of Qwen3-1.7B on the iGPU
```

## 2. Fit the lenses

```bash
export HF_HOME=$PWD/models
# bf16 models
.venv/bin/python scripts/fit_lens.py 1.7b --n-prompts 64 --compile
.venv/bin/python scripts/fit_lens.py 8b   --n-prompts 64 --compile
# int8 (fits the 48 GB VRAM wall); also used as the quantization control at 8B
.venv/bin/python scripts/fit_lens.py 8b   --int8 --n-prompts 64
.venv/bin/python scripts/fit_lens.py 32b  --int8 --n-prompts 48
```

Fitting is dominated by the model backward pass (`ceil(d_model/dim_batch)`
backward passes per prompt) and is resumable — a crash re-runs from the last
checkpoint. Lenses land in `out/lenses/<tag>.pt`.

## 3. Measure

```bash
for m in 1.7b 8b; do
  .venv/bin/python scripts/metrics.py     $m          # per-layer J-space metrics
  .venv/bin/python scripts/lens_eval.py   $m          # pass@k vs logit-lens
  .venv/bin/python scripts/causal_swap.py $m          # causal band intervention
done
# int8 variants add --int8 (targets the <tag>-int8 lens)
.venv/bin/python scripts/metrics.py 8b --int8
.venv/bin/python scripts/metrics.py 32b --int8
```

Results are written to `results/metrics/` and `results/ablation/`.

## 4. Operator/operand experiments + geometry dump

The paper's figures and the interactive explorer draw on the operator/operand runs:

```bash
for m in 1.7b 8b; do
  .venv/bin/python scripts/operator_paradigm.py  $m --domain relations  # all-pairs swap
  .venv/bin/python scripts/operator_factorize.py $m --domain relations  # ANOVA + held-out
  .venv/bin/python scripts/op_geometry_dump.py   $m --domain relations  # geometry -> results/geometry/
done
```

`op_geometry_dump.py` persists the factorization/PCA/cosine geometry (print-only in the
other two) to `results/geometry/*.json|npz`, and writes the explorer's data bundle
`docs/interactive/declension.data.js` (from the 1.7B relations run).

## 5. Figures and site

```bash
.venv/bin/python scripts/plots.py        # -> docs/figs/*.png (incl. op_geometry/op_swap/op_syncretism)
.venv/bin/python -m mkdocs serve         # preview the site (incl. the interactive explorer)
```

## The whole batch

`scripts/run_batch.sh` chains the fits + analyses for the sweep and is resumable;
`scripts/validate.sh <tag>` waits for a running fit and then runs the three
measurements for one model.
