# LessWrong / AI Alignment Forum — draft crosspost

*Personalize before posting. The AF/LW audience rewards the controlled nulls and the
calibration — lead with what replicates and what doesn't. Embed the GIFs and link the
interactive explorer.*

**Title:** The workspace declines concepts by role: operator/operand factorization in an
open-weights LLM

---

**TL;DR.** In Qwen3 (1.7B/8B), a relational operation (*currency-of*, *capital-of*, …) is
carried by a low-dimensional, causally-manipulable **operator direction** that factorizes
from the **operand** (the entity). Adding `v(to) − v(from)` flips the answer for all 20
ordered relation pairs (random control ≈ 0); the direction generalizes to held-out operands;
and the operation is separable from the *word it produces* (language ≠ demonym as directions
despite both emitting "Italian"). I frame this as **declension** — the model marks a concept
by its grammatical role, with syncretism and fusion as in a Latin case paradigm. Two things I
want to flag for this audience specifically: (1) the factorization is **specific to relational
retrieval** — arithmetic and logic operators fail the same tests, consistent with the
"bag of heuristics" picture; (2) the effect is **causal, not a readout advantage** — a
Jacobian-lens "J-space" readout is *not* more legible than the raw residual under matched
controls, including a spectrum-matched random projection. I expected the opposite and report
the null.

**Why post here.** This started as a J-space legibility study and the headline readout claim
turned out to be a controlled null. The positive result is the causal operator/operand
structure. I think the negative half is as informative as the positive, and this is the venue
that treats it that way.

## The claim, in one picture

[embed: promo/declension.gif]

Read the workspace state `H[country, case]` and watch where it "lives" along the sequence: at
the country token the cloud is organized by operand (which country, 59% of variance); by the
query token it has reorganized into clean case clusters (which relation, 86%). A two-way ANOVA
gives `H ≈ μ + operand + operator`, ~90% additive, ~10% interaction (the "fusion"), operand
and operator subspaces near-orthogonal (principal angles 41–82°).

## The operator is manipulable

[embed: promo/injection.gif]

Add `v(to) − v(from)` over the workspace band and read the answer-token logit difference. All
20 ordered pairs flip sign (mean +21 at 1.7B, +25 at 8B); a matched-norm random direction does
nothing. The direction built on half the operands transfers to the held-out half — a genuine
operator, not interpolation.

## Operation ≠ realization

`language-of` and `demonym-of` emit the identical word "Italian" yet are distinct directions.
A "pure desinence" built where the two share their output (so the word cancels) still installs
the relation causally (−3.6 → +12.5). The syncretism sits at the exponent, not the case — as
declension predicts.

## What does *not* factorize (the cross-domain test)

The identical pipeline on arithmetic (+ × −) and comparison-logic has 2–4× the interaction and
**fails** held-out generalization at both scales — the operator is entangled with its operands.
This matches the "arithmetic as a bag of heuristics / Fourier features" line and reconciles
with Christ et al. (NeurIPS 2025), whose *add-N* cut generalizes precisely because it varies a
linear numeric parameter rather than the function.

## The negative half (well controlled)

Four readout comparisons — bridge-entity pass@k, a surface-form logit difference, a number
probe, and a concept-plane trajectory — show no J-space advantage over the logit-lens. A
spectrum-matched random projection reads number as well as the fitted lens; the concept
"channel" doesn't survive a random-plane baseline. Legibility ≠ causal importance.

## Reproduce / explore

Everything runs on a single AMD Strix Halo APU (no CUDA). Interactive explorer (scrub the
declension, fire the injection, includes a case-grammar primer):
https://mpodeley.github.io/jspace-qwen/explorer/ · paper + code + working findings:
https://mpodeley.github.io/jspace-qwen/

Feedback and adversarial takes very welcome — especially on the cross-domain negative and
whether the declension framing earns its keep.
