# Method: the Jacobian lens and the J-space

This project reproduces the method of Anthropic's
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Gurnee, Sofroniew, Lindsey et al., 2026) on open-weights Qwen3 models, using
the authors' reference implementation
[`anthropics/jacobian-lens`](https://github.com/anthropics/jacobian-lens).

## The Jacobian lens (J-lens)

A *lens* reads out what an intermediate residual-stream activation is "poised to
make the model say". The **logit lens** does this by decoding a mid-layer
residual $h_\ell$ directly with the model's unembedding $W_U$. The **Jacobian
lens** first *transports* $h_\ell$ into the final-layer basis with the average
input–output Jacobian of the network:

$$
\text{lens}_\ell(h) = \operatorname{softmax}\big(W_U\,\operatorname{norm}(J_\ell\, h)\big),
\qquad
J_\ell = \mathbb{E}\!\left[\frac{\partial h_{\text{final}}}{\partial h_\ell}\right].
$$

The expectation is taken over a corpus of pretraining-like prompts, over source
positions, and over all current-and-future target positions. Intuitively, for
every vocabulary token the J-lens finds the internal activity pattern that makes
the model more likely to emit that token *at some point in the future* — it
measures **causal influence on the output**, not mere presence.

Computing $J_\ell$ requires backward passes through the model: for each of the
$d_{\text{model}}$ output dimensions a one-hot cotangent is injected at every
valid target position and backpropagated to layer $\ell$. This is why the whole
pipeline needs autograd on the GPU (not just inference) — see
[the setup page](setup.md).

## From the logit lens to the Jacobian lens

It helps to build the J-lens up from the simplest readout.

The **logit lens** ([nostalgebraist, 2020](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens))
decodes a mid-layer residual *in place* with the model's own output head:

$$
\text{logit-lens}(h_\ell) = \operatorname{softmax}\big(W_U\,\operatorname{norm}(h_\ell)\big).
$$

It asks "if the model stopped at layer $\ell$ and read out now, what would it
say?" The catch is a **basis mismatch**: the residual stream at layer $\ell$ is
not yet in the coordinate system the unembedding expects — the remaining layers
still rotate and rescale it. So the logit lens is reasonable in late layers
(basis already close to final) but noisy and systematically *under-reads*
content in the early and middle layers — exactly where the workspace lives.

The **Jacobian lens** fixes the basis by *transporting* $h_\ell$ with the
model's own average input–output Jacobian before decoding:

$$
\text{J-lens}(h_\ell) = \operatorname{softmax}\big(W_U\,\operatorname{norm}(J_\ell\, h_\ell)\big).
$$

Because $J_\ell$ is the model's first-order sensitivity — how a nudge to $h_\ell$
actually propagates to the output, averaged over real text — three things follow:

- **It reads in the right basis.** Instead of assuming the mid-layer basis equals
  the final one, it carries $h_\ell$ forward with the network's own derivative.
- **It measures influence, not presence.** A direction that exists in $h_\ell$
  but that the downstream network ignores contributes nothing to $J_\ell h_\ell$;
  the lens filters for what actually reaches the output.
- **It is derived, not trained.** Unlike the *tuned lens* (a learned affine probe
  fit to predict the next token), $J_\ell$ comes from the model itself, so it
  cannot hallucinate content that is not causally there.

| lens | transport into final basis | trained? | main failure mode |
|---|---|---|---|
| logit lens | none ($J_\ell = I$) | no | basis mismatch → under-reads mid layers |
| tuned lens | learned affine $A_\ell h + b_\ell$ | yes (fit to next token) | can hallucinate / overfit |
| **Jacobian lens** | model's average Jacobian $J_\ell$ | no (derived) | needs backward passes to build |

**The logit lens is exactly the special case $J_\ell = I$** — turn off the
transport and the J-lens *is* the logit lens. In this codebase that is a single
flag, `lens.apply(..., use_jacobian=False)`, so every "J-lens vs logit-lens"
comparison in the results is controlled: same activations, same unembedding, the
only difference is whether $h_\ell$ is carried through the model's Jacobian
first. If the J-lens beats the logit lens, the improvement is due to the
transport and nothing else.

## The J-space

The **J-space** is a small, sparse subspace of the residual stream spanned by
the J-lens vectors that are active at a given moment (typical sparsity
$k \le 25$). Its defining properties, which we measure across scale:

- it accounts for a **small fraction of activation variance** (never more than
  ~10%, varying by layer);
- far more model components read from and write to it than to ordinary
  directions (up to ~100× in parts of the network);
- it is organised by depth into a **sensory** regime (early layers), a
  **workspace** regime (the middle band, where abstract content persists), and a
  **motor** regime (late layers, aligned with the output).

## What we measure

**Per-layer metrics** (`scripts/metrics.py`), reindexed to a 0–100 depth scale:

| metric | what it captures |
|---|---|
| `jlens_top1/5`, `logitlens_top1` | next-token accuracy of the readout; J-lens vs logit-lens baseline |
| `read_var_frac` | fraction of residual variance in the J-lens' top-$k$ read directions (the "<10%" claim) |
| `eff_dim` | participation ratio of $J_\ell$'s singular values (effective dimensionality of the workspace) |
| `kurtosis` | excess kurtosis of the lens logits (a few concepts dominate → workspace signature) |
| `top1_autocorr` | persistence of the top-1 readout across adjacent positions |

**Lens quality** (`scripts/lens_eval.py`): on the paper's shipped two-hop set,
pass@$k$ = the fraction of hidden bridge entities (e.g. *Brazil* in "the country
where the Amazon ends") that appear at lens rank $\le k$ at their best layer.

**Causal importance** (`scripts/causal_swap.py`): using the shipped `probe-swap`
set, we move the residual component along a bridge entity's J-lens direction
onto a different entity's direction, across a layer band, and check whether the
model's answer flips accordingly — versus a matched-norm random control. If the
**workspace band** flips (while the control and the early/late bands do not), the
J-space is causally steering the computation.

## The scale question

The paper studies Claude. Our contribution is to ask, on open weights, **when**
these properties appear: we run the identical pipeline on Qwen3 **1.7B, 8B, and
32B** and compare the metrics above across model size — does the J-space exist
already at 1.7B, or does it sharpen (lower variance fraction, cleaner
workspace band, stronger causal flips) as models scale?
