# Results — scale

Does the J-space exist at every scale, or emerge and sharpen with size? We fit
the identical J-lens on Qwen3 1.7B, 8B, and 32B and compare per-layer metrics
(reindexed to a 0–100 depth scale) and the paper's pass@k lens-quality eval.

![J-space metrics vs depth, across model scale](figs/scale_metrics.png)

## Per-layer metrics

The workspace band (38–92% depth) is shaded. Reading the 1.7B curves:

- **`read_var_frac`** — the J-lens read directions hold a small, *shrinking*
  fraction of residual variance (~5% early → ~1.8% deep), consistent with the
  paper's "<10%" J-space.
- **`top1_autocorr`** — top-1 readout persistence **peaks inside the workspace
  band**, the signature of content that is held across positions.
- **`jlens_top1`** — next-token accuracy stays near zero until the motor regime,
  then rises steeply toward the output.
- **`eff_dim`** rises with depth (low-rank transport early, near-full-rank late);
  **`kurtosis`** dips through the mid layers.

## Lens quality (pass@k)

![Lens quality: J-lens vs logit-lens](figs/lenseval.png)

pass@k = fraction of hidden two-hop bridge entities recovered at lens rank ≤ k
(min over layers). **At 1.7B the J-lens only matches the logit-lens** — the
readout advantage the paper reports for Claude has not yet emerged. The scale
hypothesis is that this gap opens at 8B and 32B.

## Quantization control (8B bf16 vs int8)

Because 32B is run in int8 (the [48 GB wall](setup.md)), we fit the 8B in *both*
precisions and compare, to measure the confound directly.

*Early signal: the int8 per-prompt Jacobian norms match bf16 to within ~1%.
The full bf16-vs-int8 figure lands with the 8B runs.*

!!! note "In progress"
    8B (bf16 + int8) and 32B (int8) are fitting. Curves for those models overlay
    onto the figures above as they complete.
