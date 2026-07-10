# The J-space on Qwen3

Reproducing Anthropic's **global-workspace / Jacobian-lens** result
([Gurnee, Sofroniew, Lindsey et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html))
on open-weights **Qwen3**, and measuring **when** the *J-space* emerges and
**when** it is causally important — run entirely on a single AMD Strix Halo APU.

!!! tip "Interactive: **[Concept Flooding →](waterflood/)**"
    A visual, didactic piece — watch an injected idea *diffuse* through the model
    and, at the output, either **condense into a word** or stay a **silent
    thought**, drawn in the language of reservoir-simulation flooding.
    *(Currently a synthetic-data preview of the UI; real Qwen data swaps in shortly.)*

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

## Headline (1.7B → 8B)

- **The causal workspace sharpens with scale.** Swapping a bridge entity's
  J-lens direction in the mid-layer *workspace* band flips the two-hop answer
  with rate **0.30 → 0.55** from 1.7B to 8B, and *localizes* (the early band
  collapses 0.17 → 0.04); the matched-norm control stays ~0.
- **The J-space concentrates** (read variance fraction 0.027 → 0.014).
- **But the readout advantage does not emerge**: on pass@k the J-lens only ties
  the logit-lens on Qwen3 at both scales — an honest departure from the Claude
  result.
- **int8 preserves the J-space** (structural metrics within ~1–2%, correlations
  ≥ 0.98), so the 32B-int8 run is trustworthy.

!!! note "Status"
    1.7B and 8B complete (both bf16 + int8). 32B (int8) is the next scale point.
    Figures and numbers update as it completes.
