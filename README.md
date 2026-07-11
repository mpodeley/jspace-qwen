# jspace-qwen — Causal operator–operand factorization in Qwen3

Relational operations (*currency-of*, *capital-of*, …) are represented in Qwen3's residual
stream as **causally manipulable operator directions** that factorize from the operand,
transfer to held-out entities and unseen prompt frames, and are distinct from the word that
realizes them. Arithmetic and comparison-logic operators do **not** factorize — the
structure is specific to relational retrieval. All experiments run on a single AMD Strix
Halo APU (no CUDA, no cloud).

- **Site / paper / evidence log:** https://mpodeley.github.io/jspace-qwen/
- **Interactive explorer** (3Blue1Brown-style, real model data):
  https://mpodeley.github.io/jspace-qwen/explorer/
- **Reproduce:** [docs/reproduce.md](docs/reproduce.md) — env, seeds, exact checkpoint ids
  (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`), every figure regenerable from `scripts/`.
- **Preprint builds:** `paper/` (single-column) and `paper/acl/` (ACL two-column).

Headline numbers (operator-level cluster-bootstrap 95% CIs): all-pairs swap contrast
**+22.6 [+14.0, +32.1]** (1.7B) / **+26.0 [+17.9, +32.8]** (8B); held-out-operand transfer
**+20.0 / +22.8**; cross-frame transfer **180/180** (1.7B) and **100/100** (8B) flips;
flip fraction **1.00 [1.00, 1.00]** throughout. The J-lens/J-space readout shows **no
legibility advantage** over the logit lens under matched controls — the contribution is
causal structure, reported with its controls.

## How to cite

```bibtex
@misc{podeley2026operator,
  title  = {Causal Operator--Operand Factorization in the Residual Stream of Qwen3},
  author = {Podeley, Matias},
  year   = {2026},
  url    = {https://mpodeley.github.io/jspace-qwen/},
  note   = {Code: https://github.com/mpodeley/jspace-qwen}
}
```

MIT license. Contact: mpodeley@gmail.com
