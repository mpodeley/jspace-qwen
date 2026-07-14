# jspace-qwen — Operator–operand factorization in LLM residual streams: causal influence and compositional sufficiency

Relational operations (*currency-of*, *capital-of*, …) are represented in the residual
stream of Qwen3-1.7B/8B and Gemma-2-9B as **causally manipulable operator directions** that
factorize from the operand, transfer to held-out entities, unseen prompt frames, fully
re-lexicalized wordings, and a second (animal-taxonomy) domain — and the factorization is
**behaviorally sufficient**: a state *composed* from its additive parts (grand mean + operand
+ operator), written into one position, makes the model produce the target answer at its own
competence ceiling. The steering vector is *identically* the ANOVA operator component, and the
old "steering can't generate" result was an overdose artifact — at the calibrated dose it
generates at ceiling too. Arithmetic and comparison-logic operators do **not** factorize
under this setup. All experiments run on a single AMD Strix Halo APU (no CUDA, no cloud).

- **Showcase site (plain-language, EN/ES): https://mpodeley.github.io/assembled-thought/** —
  this repo remains the code & reproducibility home.
- **Learn the field (in Spanish):** [Interpretabilidad Mecanicista](https://mpodeley.github.io/interpretabilidad-mecanicista/) —
  a plain-language mech-interp course, curated resources, and research blog.
- **New to all this?** [The paper, explained simply](https://mpodeley.github.io/jspace-qwen/explained/)
  ([español](https://mpodeley.github.io/jspace-qwen/es/explained/)) — for any curious reader.
- **Site / paper / evidence log:** https://mpodeley.github.io/jspace-qwen/
- **Interactive explorer** (3Blue1Brown-style, real model data):
  https://mpodeley.github.io/jspace-qwen/explorer/
- **Reproduce:** [docs/reproduce.md](docs/reproduce.md) — env, seeds, exact checkpoint ids
  (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `google/gemma-2-9b`), every figure regenerable from
  `scripts/`.
- **Preprint builds:** `paper/` (single-column) and `paper/acl/` (ACL two-column).

Headline numbers (operator-level cluster-bootstrap 95% CIs): all-pairs swap contrast
**+22.6 [+14.0, +32.1]** (Qwen3-1.7B) / **+26.0 [+17.9, +32.8]** (Qwen3-8B) /
**+30.4 [+23.8, +34.8]** (Gemma-2-9B); held-out-operand transfer **+20.0 / +22.8 / +26.6**;
animals domain **12/12** flips (+18.1 / +25.8). **Composition:** the assembled state
`μ+operand+operator` generates at **51.8%** vs the donor's 50.9% and the model's own 53%
ceiling (8B: 62/65/68; leave-one-cell-out 36/50%; swapping the operand component redirects
the answer). **Nulls:** permuted relation labels and operator-subspace random directions
abolish the margin effect (≈0 at both scales); query-token-only injection ≈ all-positions at
60× lower off-task cost. **Audit:** even at the overdose the target wins a forced choice at
80% — overdosing destroys fluency, not information. The J-lens/J-space readout shows **no
legibility advantage** over the logit lens under matched controls.

## How to cite

```bibtex
@misc{podeley2026operator,
  title  = {Operator--Operand Factorization in LLM Residual Streams:
            Causal Influence and Compositional Sufficiency},
  author = {Podeley, Matias},
  year   = {2026},
  url    = {https://mpodeley.github.io/jspace-qwen/},
  note   = {Code: https://github.com/mpodeley/jspace-qwen}
}
```

MIT license. Author: Matias Podeley, independent researcher — mpodeley@gmail.com
