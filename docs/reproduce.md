# How to reproduce

Everything runs on a single AMD Strix Halo (Radeon 8060S, gfx1151); no CUDA, no
cloud. See [setup](setup.md) for why the environment is a plain `uv` venv.

## 1. Environment

```bash
git clone https://github.com/mpodeley/jspace-qwen.git && cd jspace-qwen
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

Two further checks from the review pass:

```bash
.venv/bin/python scripts/operator_templates.py 1.7b   # cross-frame transfer (3 context frames)
.venv/bin/python scripts/operator_lexical.py 1.7b     # cross-LEXICALIZATION transfer
.venv/bin/python scripts/op_dose.py 1.7b              # alpha dose-response + off-task KL
.venv/bin/python scripts/op_minimal.py 1.7b           # band sweep + activation patching + greedy exact match
```

## 4b. The reviewer round: composition, positions, nulls, layers, a second domain

```bash
bash scripts/run_revision.sh 1.7b     # the whole battery below, sequentially
bash scripts/run_revision.sh 8b
```

Individually:

```bash
# Does a state COMPOSED from the factorization's parts generate the answer?
# (the decomposition ladder, incl. the leave-one-cell-out reconstruction)
.venv/bin/python scripts/op_patch_decomp.py 1.7b --domain relations

# WHERE must the vector land, and what do flips hide? (positions x layer scope,
# with target rank / top-1 / exact match / on- and off-task KL; plus the
# dose x position sweep on generation)
.venv/bin/python scripts/op_positions.py 1.7b --domain relations

# The competitive null battery (permuted operator labels, operator-subspace random,
# wrong layer, other-relation direction, shuffled answers)
.venv/bin/python scripts/op_nulls.py 1.7b --domain relations --seeds 20

# Where is the operator most READABLE vs most CAUSAL?
.venv/bin/python scripts/op_layer_sweep.py 1.7b --domain relations

# Is the direction a signed axis? (negative alphas)
.venv/bin/python scripts/op_dose.py 1.7b --alphas -12 -8 -4 -2 -1 -0.5 0.5 1 2 4 6 8 12

# The non-geographic domain (animals -> class/habitat/diet/covering)
.venv/bin/python scripts/build_op_datasets.py --only animals   # tokenizer-screening gate
.venv/bin/python scripts/operator_paradigm.py 1.7b --domain animals
```

!!! note "Reproducibility box"
    All experiments are seeded (`--seed 0` default; the RNG seeds only the matched-norm
    random controls — direction building is deterministic). Checkpoints: `Qwen/Qwen3-1.7B`,
    `Qwen/Qwen3-8B` (bf16), `google/gemma-2-9b` for the cross-architecture run. Per-operand
    long-form values are persisted as `results/ablation/*_long.parquet`; every figure
    regenerates from `scripts/plots.py`. License MIT; contact `mpodeley@gmail.com`.

## 5. Figures and site

```bash
.venv/bin/python scripts/plots.py        # -> docs/figs/*.png (incl. op_geometry/op_swap/op_syncretism)
.venv/bin/python -m mkdocs serve         # preview the site (incl. the interactive explorer)
```

## The whole batch

`scripts/run_batch.sh` chains the fits + analyses for the sweep and is resumable;
`scripts/validate.sh <tag>` waits for a running fit and then runs the three
measurements for one model.
