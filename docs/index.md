# The J-space on Qwen3

Reproducing Anthropic's **global-workspace / Jacobian-lens** result
([Gurnee, Sofroniew, Lindsey et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html))
on open-weights **Qwen3**, and measuring **when** the *J-space* emerges and
**when** it is causally important — run entirely on a single AMD Strix Halo APU.

## What this is

The paper introduces the **Jacobian lens (J-lens)**, which reads out what an
internal activation is "poised to make the model say", and the **J-space**: a
small, sparse subspace of the residual stream that behaves like a *global
workspace* — it holds a few concepts at a time, accounts for <10% of activation
variance, and is organised by depth into sensory → workspace → motor regimes.

The original study is on Claude. This project asks the question the paper
cannot: on open weights, **does the J-space exist at every scale, or does it
emerge and sharpen as models grow?** We run the identical pipeline on Qwen3
**1.7B, 8B, and 32B** and compare.

- [**Method**](method.md) — the J-lens math and every metric we compute.
- [**Setup**](setup.md) — the AMD/ROCm recipe, the 48 GB memory wall, and the
  custom int8 path that keeps 32B in the study.
- [**Results — scale**](results-scale.md) · [**Results — causal**](results-causal.md)
- [**How to reproduce**](reproduce.md)

## Preliminary finding (1.7B)

The smallest model already shows a clean split between *structure* and
*legibility*:

- **The causal workspace signature is present.** Swapping a bridge entity's
  J-lens direction across the mid-layer *workspace* band flips the model's
  two-hop answer (flip-rate 0.30) far more than the early (0.17) or late (0.04)
  bands — while a matched-norm **random control flips nothing (0.00)**.
- **The readout advantage has not yet emerged.** On the two-hop pass@k eval the
  J-lens only matches the logit-lens baseline at 1.7B.
- The J-space read directions hold ~2% of residual variance, and top-1
  readout **persistence peaks inside the workspace band** — both workspace
  signatures.

Whether the readout advantage and a sharper workspace emerge at 8B and 32B is
the scale question this project answers.

!!! note "Status"
    Work in progress. 1.7B complete; 8B (bf16 + int8 quantization control) and
    32B (int8) fitting in progress. Figures and numbers update as the sweep
    completes.
