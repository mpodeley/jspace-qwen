# LessWrong / AI Alignment Forum — draft crosspost

*Personalize before posting. Numbers current as of 2026-07-12 (reviewer round 2: composition
ladder, dose × position, competitive nulls, layer sweep, animals domain, generation audit —
all replicated at 8B). The AF/LW audience rewards the controlled nulls and the calibration —
lead with what replicates, what doesn't, and what we got wrong ourselves (the overdose).*

**Title:** A thought assembled from averaged parts makes the model speak: operator/operand
factorization and compositional sufficiency in open-weights LLMs

---

**TL;DR.** In Qwen3 (1.7B/8B) and Gemma-2-9B, a relational operation (*currency-of*,
*capital-of*, …) is carried by a causally manipulable **operator direction** that factorizes
from the **operand** (the entity). Adding `v(to) − v(from)` flips the sign of the answer-logit
margin for all 20 ordered relation pairs in all three models (matched-norm random control ≈ 0);
the direction generalizes to held-out operands, unseen prompt frames, and **fully re-lexicalized
wordings** ("currency of" → "money used in": 40/40, transfer ≈ within); and the operation is
separable from the *word it produces* (language ≠ demonym as directions despite both emitting
"Italian"). The headline: the factorization is **behaviorally sufficient** — a state *composed*
from its averaged additive parts (`μ + operand + operator`), patched into one position, makes
the model *produce* the answer at its own competence ceiling (52% vs 53% clean at 1.7B; 62% vs
68% at 8B; components that never saw the target cell still reach 36–50%; swapping the operand
part redirects the answer to that operand). Four things to flag for this audience specifically:
(1) we falsified our *own* earlier claim — "additive steering can't generate (≤0.5%)" was an
**overdose artifact**: the steering vector is identically the ANOVA operator component, and at
the calibrated dose it generates at ceiling (51.3%), with generation an inverted-U peaked
exactly where the algebra says the state is on-manifold; (2) a **generation audit** (raw texts
classified 5 ways + a forced-choice sequence-log-prob readout) shows overdosing destroys
*fluency*, not *information* — the target still wins a forced choice at 80% under 94.6% token
loops; (3) the factorization is **specific to relational retrieval** under our setup — a
curated animals domain replicates everything while arithmetic and logic fail the same tests;
the decisive null (directions rebuilt under **permuted relation labels**) abolishes the effect
at both scales; (4) the effect is **causal, not a readout advantage** — a Jacobian-lens
"J-space" readout is *not* more legible than the raw residual under matched controls. I
expected the opposite and report the null.

**Why post here.** This started as a J-space legibility study and the headline readout claim
turned out to be a controlled null. The positive result is the causal operator/operand
structure. I think the negative half is as informative as the positive, and this is the venue
that treats it that way.

## The claim, in one picture

[embed: promo/declension.gif]

Read the residual state `H[country, case]` and watch where it "lives" along the sequence: at
the country token the cloud is organized by operand (which country, 59% of variance); by the
query token it has reorganized into clean case clusters (which relation, 82–86%). A two-way
ANOVA gives `H ≈ μ + operand + operator`, ~90% additive, ~10% interaction (the "fusion"),
operand and operator subspaces well separated at the query token (principal angles 41–85°).

## The operator is manipulable

[embed: promo/injection.gif]

Add `v(to) − v(from)` over the workspace band and read the answer-token logit margin. All 20
ordered pairs flip sign in all three models — swap-minus-random contrast **+22.6 [+14.0, +32.1]**
(Qwen3-1.7B), **+26.0 [+17.9, +32.8]** (Qwen3-8B), **+30.4 [+23.8, +34.8]** (Gemma-2-9B;
operator-level cluster-bootstrap 95% CIs — the 20 pairs share 5 directions and 12 operands, so
pairs are *not* independent observations and the bootstrap resamples operators/operands, not
pairs). A matched-norm random direction produces a flat, nonspecific shift. The direction built
on half the operands transfers to the held-out half (+20.0 / +22.8 / +26.6) — a genuine
operator, not interpolation.

## Not the prompt, not the wording

Built on one context frame, the direction flips prompts in unseen frames (1.7B 180/180, 8B
100/100). The stronger control: build the direction on "The currency of X is" and test on
prompts re-lexicalized as "The money used in X is" (every relation re-worded) — 40/40 both
ways at both Qwen3 scales, with the transfer contrast (+22.59) matching the within-wording
contrast (+22.62). The direction encodes the relation, not its surface form.

## The composition ladder — and the overdose we caught ourselves in

Decompose the donor activation by the exact factorization and patch partial reconstructions
back in: `μ + operand + operator` generates at the donor's own level (51.8% vs 50.9%, clean
ceiling 53%; 8B: 62.1% vs 65.2%, ceiling 68%) while the interaction term alone does nothing
(7.1%) and a norm-matched magnitude control does less (4.5%). Leave-one-cell-out components
(never saw the target cell) still generate at 35.7% / 50.4%; swapping the operand component
redirects the answer to *that* operand (34.4% / 48.2% vs ~2% baseline). And the honest
reversal: our earlier "additive steering flips margins but can't generate (≤0.5%)" was an
**overdose** — the steering vector IS the operator component, and generation follows an
inverted-U peaked exactly at the on-manifold dose (51.3% at α≈0.1 band-wide; 38.4% at α=1
single-layer; same peak doses at 8B: 56.2% / 45.1%). Past the peak the margin keeps climbing
while generations collapse into token loops (94.6% degraded by manual classification) — yet
the target still wins a forced-choice among the operand's answers at 80.4%. **Overdosing
destroys fluency, not information.** Also: query-token-only injection matches all-position
injection at 60× lower off-task KL — a localized edit, not a global perturbation.

## Operation ≠ realization

`language-of` and `demonym-of` emit the identical word "Italian" yet are distinct directions.
A "pure desinence" built where the two share their output (so the word cancels) still installs
the relation causally (clean −3.6/−2.9/−3.0 → +12.5/+8.1/+8.5 across the three models). The
syncretism sits at the exponent, not the case — as declension predicts.

## What does *not* factorize (the cross-domain test)

The identical pipeline on arithmetic (+ × −) and comparison-logic has 2–4× the interaction and
**fails** held-out generalization at both Qwen3 scales — the operator is entangled with its
operands. This matches the "arithmetic as a bag of heuristics / Fourier features" line and
reconciles with Christ et al. (NeurIPS 2025), whose *add-N* cut generalizes precisely because
it varies a linear numeric parameter rather than the function (add-N is also the most
collinear family: 76% of variance on one line).

## The negative half (well controlled)

Four readout comparisons — bridge-entity pass@k, a surface-form logit difference, a number
probe, and a concept-plane trajectory — show no J-space advantage over the logit-lens. A
spectrum-matched random projection reads number as well as the fitted lens; the concept
"channel" doesn't survive a random-plane baseline. Legibility ≠ causal importance.

## Reproduce / explore

Everything runs on a single AMD Strix Halo APU (no CUDA). Interactive explorer (scrub the
declension, fire the injection, includes a case-grammar primer):
https://mpodeley.github.io/jspace-qwen/explorer/ · plain-language explainer:
https://mpodeley.github.io/jspace-qwen/explained/ · paper + code + working findings:
https://mpodeley.github.io/jspace-qwen/

Feedback and adversarial takes very welcome — especially on the cross-domain negative and
whether the declension framing earns its keep.
