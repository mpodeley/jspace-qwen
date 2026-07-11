# Findings — J-lens readout vs. causal routing on Qwen3

*This is the chronological lab notebook — technical and unpolished by design. For the story, start at
[Explained simply](explained.md); for the claims with their controls, see
[Evidence & controls](robustness.md).*

*Working evidence document, 2026-07-10. Not a fixed thesis. It lays out every
measurement, its controls, and the interpretive options — the framing decision
is deliberately left open.*

## The question that started this

The project set out to test a sharpened version of the global-workspace / J-lens
result on Qwen3: **does the J-lens readout carry information the logit-lens
lacks — specifically morphosyntactic (number, casing, a/an) structure?** The
motivation was twofold: the standing `pass@k` result where the J-lens only *ties*
the logit-lens, and a concept-plane figure (`flow.py`) where the J-lens path
looked visibly richer than the logit-lens path.

The short version of what we found: **every correlational "one lens reads more
than the other" comparison is null under controls. The one robust, controlled
effect is causal — intervening on the workspace reroutes downstream computation.**

---

## Part 1 — Readout comparisons (all null under controls)

All four ask a version of "does the J-lens readout expose X better than the
logit-lens readout?" All were run on Qwen3-1.7B (8B where noted).

### 1.1 Identity: `pass@k` on the multihop bridge entity

Does the lens surface the unstated bridge entity (e.g. "Brazil")? Existing result,
`results/metrics/*_lenseval.json`:

| model | jlens pass@1 | logit pass@1 | jlens pass@10 | logit pass@10 |
|---|---:|---:|---:|---:|
| 1.7b | 0.087 | 0.100 | 0.362 | 0.300 |
| 8b | 0.075 | 0.150 | 0.450 | 0.475 |

A tie, drifting slightly toward the logit-lens at 8B. `pass@k` scores next-token
*identity*, a lexical-semantic axis.

### 1.2 Surface form: two-token logit difference

Conditioning on the model getting the clean case right, does the lens rank the
correct surface form above a *real* competing form (`mice` vs `mouse`, `are` vs
`is`)? Normalized logit difference, per band, 1.7B (`lens_eval.py --set
lemma-form`, agreement subset, 10/12 clean):

| band | J-lens | logit-lens |
|---|---:|---:|
| early | 0.035 | 0.135 |
| workspace | 0.083 | 0.119 |
| **late** | **0.272** | **0.316** |

The number distinction is ~0 in the workspace for both and rises only in the late
band — consistent with "form decided late." But the logit-lens matches or beats
the J-lens in every band. (An earlier version used non-word competitors like
"mouses"; that was confounded — the non-word has intrinsically low logit, so the
difference was trivially positive. Fixed to real competitors.)

### 1.3 Number probe (the "rescue" attempt)

The two-token difference is a 1-D readout; maybe a probe over a richer set of
logit dimensions sees a J-lens advantage the difference misses. L2 logistic probe
for grammatical number over number-diagnostic logit axes, grouped k-fold by noun,
Hewitt–Liang selectivity = AUC(real) − AUC(shuffled). 1.7B, 198 items
(`syntax_probe.py`):

| band | J selectivity | logit selectivity | **randproj** selectivity | J AUC | logit AUC |
|---|---:|---:|---:|---:|---:|
| early | 0.447 | 0.400 | 0.351 | 0.993 | 0.926 |
| workspace | 0.440 | 0.486 | 0.457 | 0.986 | 0.998 |
| late | 0.498 | 0.511 | 0.485 | 1.000 | 1.000 |

Number is decodable at ~ceiling from **all three** lenses in **every** band. The
decisive control: a **spectrum-matched random projection** (`randproj`, no fitted
directional structure) reads number as well as the fitted J-lens. So the probe is
not measuring anything specific to the fitted J — number is ubiquitously present,
and the lens choice is irrelevant.

### 1.4 The concept-plane channel (the visual that motivated all this)

`flow.py` projects the residual (transported by the J-lens, and untransported for
the logit-lens) onto the 2-D plane of two concept tokens (Italy, euro) and the
J-lens path looked far richer. Is that plane-specific, or generic? Path length on
the real plane vs 20 random concept planes, 1.7B (scratch measurement):

| plane | J-lens path | logit-lens path | J/logit ratio |
|---|---:|---:|---:|
| real (Italy × euro) | 75.1 | 60.2 | 1.25 |
| random (n=20) | 52.2 ± 15.8 | 31.9 ± 4.4 | 1.64 |

The J-lens does travel farther than the logit-lens on the real plane — but it
travels farther on random planes too, by a *larger* ratio. The "channel" is a
generic J-lens magnitude effect (transport through `J_l` rescales the residual in
any direction), not a concept-specific readout. If anything the logit-lens is
*relatively more* active on the real plane than on random ones.

**Part 1 summary.** Four independent readout comparisons — identity, form, a
number probe, and the concept-plane trajectory — show no J-lens advantage over
the logit-lens. Two are pinned by strong controls: a random projection matches
the fitted J-lens on number, and the concept "channel" doesn't survive a
random-plane baseline. On Qwen3, under controls, the J-space is not a specially
*readable* subspace.

---

## Part 2 — Causal intervention (real, survives controls)

### 2.1 Semantic reroute (existing `causal_swap`)

Rewriting the bridge entity's J-lens direction across a band and reading the
two-hop answer. `results/ablation/*_swap.parquet`:

| model | early flip | workspace flip | late flip | control (workspace) |
|---|---:|---:|---:|---:|
| 1.7b | 0.174 | **0.304** | 0.043 | 0.000 |
| 8b | 0.043 | **0.553** | 0.021 | 0.043 |

The workspace swap flips the answer far more than the early or late bands, and a
matched-norm random control flips ~nothing. This sharpens with scale (0.30 →
0.55) and localizes to the workspace band.

### 2.2 The pointer reroutes the concept — but not its morphology

`syntax_swap.py run_aan`: swap the workspace concept pointer (`cat → elephant`)
and read what the model does. 1.7B, 23/40 clean.

- **The noun reroutes cleanly.** Force the determiner, read the noun: `cat →
  elephant`, `dog → owl`, `spider → ant`, `frog → apple`. The workspace node *is*
  a rerouteable concept pointer, even for concepts only *inferred* from a riddle
  (never named in the prompt). This is the "channel" intuition, confirmed
  causally.
- **The determiner does not follow.** After swapping to a vowel-onset noun, the
  determiner stays `" a"` — the model produces ungrammatical "a elephant". The
  `a/an` choice was committed before the noun was dereferenced, so rewriting the
  pointer changes the noun but not the already-decided determiner.

So morphology does **not** follow the pointer downstream — the opposite of the
naive prediction, and itself a dissociation (concept is rerouteable at the node;
its surface realization is not, at least via this node).

### 2.3 The relational "declension paradigm" (`operator_paradigm.py`)

The strongest positive result. Reframe: the workspace holds not a bare concept but
a concept marked by its pending *relational role* — "Italy-in-the-currency-case"
vs "Italy-in-the-capital-case". The relation (currency-of, capital-of, language-of,
demonym-of, continent-of) is a **case**; the country is a **stem**. This is the
function-vector view (Todd et al. 2024) with a morphological reading, and it
connects to relational linearity (Hernandez et al. 2024).

Build each case as a direction: the mean, over stems, of that case's deviation
from the stem's average over cases. Then intervene on a `from` prompt with
`v(to) − v(from)` and read the answer-token logit shift (1.7B, 12 countries):

| swap | clean (from wins) | +operator | +random (matched norm) |
|---|---:|---:|---:|
| currency → capital | −7.9 | **+31.5** | −1.2 |
| capital → currency | −12.0 | **+33.0** | −2.5 |
| capital → language | −8.5 | **+32.5** | +3.0 |
| language → capital | −7.6 | **+34.0** | −2.1 |
| continent → language | −1.9 | **+23.0** | +1.4 |

**All 20 ordered case-swaps flip the sign; the matched-norm random control does
not** (its 8/20 sign-crossings are ±2 noise against a +10…+34 effect). The
relations are a structured set of causally-manipulable case-directions.

Two findings sharpen the "declension" reading:

- **Operation ≠ realization.** `language-of` and `demonym-of` produce the *same
  output word* (Italian = the language and the demonym), yet they are **distinct
  case-directions** in the causal representation, and the swap *between* them is
  weak (+3.9, vs +20…+34 for other pairs) precisely because they share an
  exponent. The model separates the grammatical role from its surface form — and
  the syncretism (shared form) shows up exactly where declension predicts: at the
  level of the exponent, not the case.

- **J-space shows the syncretism, not cleaner cases (1.7B; does NOT replicate at
  8B).** Building the case directions in the J-lens readout `unembed(J h)` vs the
  logit-lens readout `unembed(h)`: at 1.7B the cases are **less** separable in
  J-space (mean |off-diagonal cosine| 0.38 vs 0.25) and reorganize into output-form
  families — the form-sharing cases pulled together. That fit the story that `J_l`
  maps toward the output so J-space reflects realization. **But at 8B the two
  spaces are comparable (0.245 vs 0.259)**, so this specific readout-geometry claim
  is 1.7B-specific and should not be leaned on. The causal case structure below is
  what replicates.

Both the swap efficacy (**20/20 flips, random ~0**) and the factorization (§2.4)
replicate at 8B; only the J-space-geometry sub-claim does not.

### 2.4 The representation factorizes into operator ⊕ operand (`operator_factorize.py`)

Two-way decomposition of the workspace tensor `H[country, case] = μ + stem(c) +
case(k) + interaction`. Variance shares and principal angles between the stem- and
case-subspaces, at a mid-workspace layer:

| read position | stem (operand) | case (operator) | interaction (fusion) | 1.7B / 8B |
|---|---:|---:|---:|---|
| query token ("is") | 5% | **86% / 82%** | 9% / 13% | case-dominant |
| country token | **59% / 55%** | 31% / 34% | 9% / 11% | stem-dominant |

Three things, replicating 1.7B→8B:

- **The representation is ~90% additive** in operator + operand — `H ≈ μ + stem +
  case` is a good model, so relations compose roughly linearly. The ~9–13%
  **interaction** is the *fusion*: where stem and ending do not cleanly concatenate
  (the phonological fusion of a declension). Stem- and case-subspaces are largely
  separate (principal angles 41–82°).

- **The declension happens along the sequence.** At the country token the
  representation is stem-dominant (~55–59% country); by the query token it has
  flipped to case-dominant (~82–86% operation). So this is **not template echo** —
  the bare concept enters, and the workspace *declines* it into an operation-marked
  form as it flows to the query position. (The causal swaps are the stronger
  control: adding `v(case)` changes the output.)

- **A pure desinence is isolable.** `v = mean_country[h(language) − h(demonym)]`,
  built where the two cases emit the SAME word (Italian), cancels the exponent and
  leaves a bare case marker. Added to a currency prompt it installs the relation
  (logit(language) − logit(currency): 1.7B −3.6 → +12.5; 8B −2.9 → +8.1) — the
  ending, stripped of its form, is still causally functional.

### 2.5 Cross-domain: the factorization is specific to relations (`op_core.py`)

The operator framework is domain-general (`scripts/op_core.py`; `--domain
relations|arithmetic|logic`). Applying the identical pipeline to **arithmetic**
(+, ×, − over single-digit operands) and **comparison-logic** (greater/less/equal
over number pairs) tests how far the factorization generalizes. The key
discriminator is **held-out-operand generalization**: build `v(op)` from half the
operands, swap the other half — a real operator transfers, mere interpolation does
not.

| domain | operator variance | interaction (fusion) | held-out generalization | swap vs random |
|---|---:|---:|---|---|
| **relational** (1.7B) | 86% | 9% | **20/20** flip (+19) | +21 vs ~0 |
| **relational** (8B) | 82% | 13% | **20/20** flip (+22) | +25 vs ~0 |
| **arithmetic** (1.7B) | 55% | 23% | 2/6 (−0.4, ≤ random) | +0.4 vs +0.3 |
| **arithmetic** (8B) | 42% | 25% | 1/6 (−0.6, ≤ random) | +1.4 vs +0.0 |
| **logic/compare** (1.7B) | 33% | 34% | 2/6 (−0.5, ≤ random) | +1.2 vs +0.2 |
| **logic/compare** (8B) | 25% | 45% | 0/6 (−2.8, < random) | −0.6 vs −0.1 |

The gradient is monotone at both scales — relational (clean) ≫ arithmetic > logic
(worst: highest interaction, zero held-out generalization). Relational operators
factorize cleanly (case-dominant, low interaction) and
**generalize to held-out operands** at both scales. Arithmetic and logic operators
have **2–4× the interaction** and **fail to generalize** (held-out flip ≤ random) —
the operator is entangled with its operands, not a transferable direction. This
matches the literature's picture of arithmetic as a bag of heuristics / Fourier
computation (Nikankin et al. 2025; Nanda et al. 2023; Kantamneni & Tegmark 2025)
rather than a linear operator, and extends it with a quantitative factorization and
a generalization test that pinpoint *why* it is not operator-linear. The held-out
test is the discriminator: a clean in-sample swap (arithmetic swaps do flip
in-sample) is **not** evidence of a real operator unless it generalizes.

A [draft write-up is at **paper.md**](paper.md).

### 2.6 Reconciling with Christ et al. (2510.26543): add-N vs. two-operand arithmetic

Christ et al. (Oct 2025) report that arithmetic **generalizes** across held-out
relations when each relation is an *add-N* (fixed addend N). That is the opposite
sign to our §2.5 arithmetic negative — so we reproduce their cut on Qwen3 and
reconcile. In *add-N* the operator **is the addend N**, over a single number
operand; because a number is a linear quantity, "generalizing across N" is
number-line interpolation, not operator structure. Our `+ × −` cut instead varies
the **function** over two operands.

Measured (`--domain arith_addN` vs `arithmetic`), the reconciliation is directional
but honest — the single-digit BPE constraint keeps the add-N grid tiny (5 operands),
so we cannot reproduce their strong multi-digit generalization:

| cut | held-out generalization (1.7B / 8B) | operator-set collinearity (top-1 var) |
|---|---|---:|
| relations (5 operations) | 20/20 · 20/20 | 0.39 (spread paradigm) |
| **add-N** (operator = addend) | 5/12 (+0.10) · 6/12 (+0.05) | **0.76 (most 1-D / number-line)** |
| `+ × −` (operator = function) | 2/6 (−0.43) · 1/6 (−0.55) | 0.64 |

So add-N sits between: it generalizes *better* than `+ × −` (positive vs clearly
negative) and its operator directions are the **most collinear** (76% of the
operator-set variance on one line) — consistent with add-N being a linear numeric
family. The genuine-operation cut (`+ × −`) is entangled and generalizes worst.
This matches Christ's positive (their cut varies a linear parameter) and our
negative (ours varies the function) without contradiction; the constraint is that
Qwen3 BPE limits us to single-digit arithmetic, so the add-N generalization is weak
in absolute terms — the *ordering*, not the magnitude, is the point.

### 2.7 Statistical treatment and paraphrase-frame transfer (2026-07-10, evening)

Two reviewer-driven upgrades, both landing the same conclusions with proper uncertainty:

- **Cluster bootstrap.** The 20 ordered swaps reuse 5 operator directions over 12 shared operands — not
  20 independent observations. Per-operand values are now persisted (`*_operator_swap_long.parquet`)
  and CIs are cluster-bootstrapped at two levels (operands within pair; operators as top-level clusters
  via a dyadic node bootstrap — `op_core.bootstrap_pair_ci` / `bootstrap_family_ci`). Operator-level
  swap−random contrast: **1.7B +22.6 [+14.0, +32.1]; 8B +26.0 [+17.9, +32.8]**; held-out-operand
  contrast **+20.0 [+10.8, +29.5] / +22.8 [+14.0, +31.0]**; flip fraction **1.00 [1.00, 1.00]**
  everywhere. No per-pair operand-bootstrap CI crosses zero (see `figs/op_swap_dist.png`).

- **Cross-frame transfer** (`operator_templates.py`). Three paraphrase frames (declarative / QA /
  discourse-prefixed) holding the `{op} of {a}` unit fixed. 1.7B, all 9 build→test combinations:
  **180/180 flips** (off-diagonal 120/120, mean contrast **+24.2**), contrast frame-invariant to ~0.02
  while clean baselines shift (−7.0/−8.1/−7.9) — the frames are genuinely different prompts, the
  operator's causal effect is a property of the relation, not the template. Kills the template-echo
  reading beyond the reading-position control. **8B replicates: 100/100 flips on the tested combos
  (80/80 cross-frame, mean contrast +28.7), declarative-built direction frame-invariant
  (+26.0/+26.0/+26.0), QA-built strongest (+35.3).**

### 2.8 Cross-architecture replication: Gemma-2-9B (2026-07-10, night)

The full pipeline transfers unchanged to `google/gemma-2-9b` (different corpus, SentencePiece
tokenizer with 0.95 single-token answer coverage, soft-capped logits). **Everything replicates,
with the largest effect sizes of the three models:** all-pairs swap 20/20, operator-level contrast
**+30.4 [+23.8, +34.8]** (vs +22.6 / +26.0 for Qwen3-1.7B/8B); held-out operands 20/20 at
**+26.6 [+20.6, +33.2]**; factorization at the query token stem 8.4% / **case 84.8%** / fusion
**6.8%** (lowest of the three); along-sequence shift present (stem 37.6%→8.4%); pure desinence
−3.0 → **+8.5**. Zero per-pair CIs cross zero. The factorization is not a Qwen idiosyncrasy —
paper claims upgraded to two-family evidence (title now "…residual stream of LLMs").

### 2.9 Cross-lexicalization transfer (2026-07-11, reviewer round)

The frame test (§2.7) held the `{op} of {a}` unit fixed — a reviewer rightly noted that proves
*context* invariance, not *wording* invariance. New test (`operator_lexical.py`,
`data/relations_lex.json`): re-lexicalize every relation (currency of → money used in; capital of →
seat of government in; language of → language primarily spoken in; demonym → name for someone from;
continent of → world region containing) and transfer directions both ways. **1.7B: 40/40 flips, transfer
contrast +22.59 vs +22.62 within-formulation. 8B: 40/40, +26.0/+27.6.** Clean baselines differ across
formulations; the causal effect does not. The direction is a property of the relation, not its wording —
the lexical confound is closed.

### 2.10 Minimal band, activation patching, and margins vs. generation (2026-07-11)

`op_minimal.py` (1.7B): a **single mid-workspace layer** flips 20/20 margins (+6.8 [+3.0, +8.9]) at
4.4× lower off-task KL (4.6 vs 20.6 nats); half band +12.8 at 10.9 nats. Greedy honesty check: the
additive injection at ANY width essentially never makes the model emit answer_B in 3 greedy tokens (≤0.5%),
while a **query-position activation patch** (real donor state, one position) does — **51% vs the
model's own 53% clean accuracy** (ceiling). Random control 0%. Reading: the averaged difference
direction rotates relative preference; the state-level patch reroutes the actual answer. Artifacts:
`1.7b_relations_minimal.parquet` + `_minimal_margins.json`.

### 2.11 Decomposing the donor: a COMPOSED state generates (2026-07-11, reviewer round 2)

The §2.10 gap ("the patch has more information than the direction") demanded an answer to *which*
information. `op_patch_decomp.py` decomposes the donor at the query position by the exact two-way
factorization `H = μ + stem(operand) + case(operator) + inter` and patches partial reconstructions
(μ included in every variant — the hook replaces the state, so a variant must be a plausible
full-magnitude residual). 1.7B relations, 224 (pair, operand) cells, clean greedy ceiling 53%:

| patched state | says target | says source | says other | Δmargin | rank(to) | top-1 |
|---|---:|---:|---:|---:|---:|---:|
| magnitude control (μ + random, norm-matched) | 4.5% | 33.5% | 0.0% | +6.6 | 2889 | 0% |
| interaction only (μ + inter) | 7.1% | 35.3% | 0.0% | +8.5 | 906 | 0% |
| operand only (μ + stem) | 7.6% | 37.5% | 0.0% | +6.9 | 275 | 0% |
| operator only (μ + case) | 20.1% | 14.7% | 1.8% | +12.8 | 174 | 0% |
| **operator + operand (μ + stem + case)** | **51.8%** | 13.4% | 1.8% | +12.9 | 90 | 31.7% |
| operator + operand, **held-out cell** | **35.7%** | 18.8% | 1.8% | +11.5 | 98 | 17.9% |
| operator + **wrong** operand (μ + stem′ + case) | 20.1% | 11.6% | **34.4%** | +12.4 | 670 | 0% |
| full donor (μ + stem + case + inter) | 50.9% | 12.5% | 1.8% | +13.8 | 79 | 31.2% |

**The reviewer's hypothesis inverts.** The expected result was that generation needs the non-additive
interaction term the averaged direction lacks. It does not: the purely additive composition
`μ + stem + case` generates **at the donor's own level** (51.8% vs 50.9%), i.e. at the model's clean
ceiling; the interaction term alone does nothing (7.1%). The two controls close the loopholes:
**leave-one-cell-out** (stem and case rebuilt without ever seeing cell (o,to)) still generates at
35.7% — composition, not leakage — and **swapping the stem redirects the answer to the swapped
operand** (34.4% says-other vs 1.8% baseline): the state is compositional in exactly the way the
factorization claims.

**And the direction IS the component.** `v(k)` from `op_dirs` equals the ANOVA main effect `case[k]`
**identically** (max relative deviation 8.5e-08 across the band — same algebra, verified numerically).
So steering and patching apply the *same object* in two different ways: steering **adds**
`α·(case[to] − case[frm])` on top of a state that already contains `case[frm]` (at α=4 over a 13-layer
band: a far off-manifold overshoot), while the composition **replaces** the state with
`μ + stem + case[to]`. The §2.10 dissociation is therefore a fact about the *mode and dose* of
intervention, not about missing information — the additive factorization is behaviorally sufficient.
(The dose × position sweep in `op_positions.py` tests the α-artifact reading directly.)
Artifacts: `1.7b_relations_patch_decomp.parquet` + `.json` (band), `_patch_decomp_single.*` (one layer).

**Single-layer variant.** Replacing the query state at ONE mid-workspace layer: `μ + case[to]` *alone*
generates at 44.6% ≈ full donor 40.6% — the operand is re-read from the unpatched layers and positions
(consistently, the stem-swap no longer redirects there: 1.3% vs 34.4% at the band). The operator
injection point is local; the operand identity is distributed and recoverable.

**8B replication (2026-07-11 evening): every number sharpens.** Clean ceiling 68%; composed
`μ+stem+case` **62.1%** ≈ donor 65.2%; **leave-one-cell-out 50.4%** (1.7B: 35.7%); **stem-swap
redirect 48.2%** (1.7B: 34.4%); interaction-only 3.6%; magnitude control 2.2%. Artifact:
`8b_relations_patch_decomp.parquet` + `.json`.

### 2.12 Dose × position: steering DOES generate at the calibrated dose (2026-07-11)

`op_positions.py` (1.7B relations, 224 cells/condition). The α-artifact reading of §2.11 is confirmed
quantitatively — the peak generation dose lands exactly where the algebra predicts (on-manifold at
`α ≈ 1` for one layer; `α ≈ 1/n_layers ≈ 0.1` for the 13-layer band, since band additions accumulate):

| condition | α | greedy exact match | Δmargin |
|---|---:|---:|---:|
| band, all positions | **0.1** | **51.3%** | +16.7 |
| band, all positions | 4.0 (old default) | 0.4% | +28.9 |
| band, query only | 0.1 | 39.3% | +13.4 |
| single layer, query only | **1.0** | **38.4%** | +12.1 |
| single layer, query only | 0.1 | 3.1% | +0.3 |
| single layer, query only | 4.0 | 9.8% | +13.7 |

An inverted-U in generation centered at the predicted dose, while the margin keeps climbing
monotonically past it: **overdosing keeps shifting preference but pushes the state off-manifold and
kills generation** (on-task KL at band/α=4: 29 nats). The old "≤0.5% exact match" was an overdose
artifact. At the calibrated dose, additive steering generates at the model's ceiling — level (iii)
is reached by *addition*, not only by replacement.

**Position specificity (α=4 battery).** Query-token-only injection ≈ all-positions on the margin
(+28.7 vs +28.9, sign-correct 98%) at **60× lower off-task cost** (KL 0.29 vs 18.4 nats/token);
operand-token and sentence-initial ("wrong") controls do ~nothing (+1.8 / +2.3, sign+ 4% / 16%).
Single layer + query position: target rank 570 → 69, off-task KL 0.04. The margin effect is a
**localized edit of the queried relation**, not a global perturbation — the reviewer's objection is
answered directly. Artifacts: `1.7b_relations_positions.parquet` + `.json`, `_posdose.parquet`.

### 2.13 The competitive null battery: semantics is everything; the margin has parts (2026-07-11)

`op_nulls.py` (1.7B relations, α=4; 20 redraws per stochastic null; the script proves its swap loop
identical to `op_core.measure_swaps_long` before running, max Δ 0.0). Contrast vs matched-norm random,
operator-level cluster bootstrap:

| null | contrast [95% CI] | flips | reading |
|---|---:|---:|---|
| **real swap** | **+22.62 [+14.0, +32.1]** | 1.00 | |
| permuted labels (per-operand) | +0.68 [−0.8, +3.9] | 0.45 | **the decisive null: → 0** |
| permuted labels (global) | +0.57 [−2.1, +3.0] | 0.35 | → 0 |
| random in operator **subspace** | +0.75 [−2.2, +3.5] | 0.45 | right home, wrong content: → 0 |
| wrong layer (early band) | +9.77 [+5.2, +13.3] | 1.00 | additions **persist** downstream |
| other-relation direction | +11.09 [+6.7, +17.1] | 0.80 | −case(source) alone ≈ half the margin |
| shuffled answers | +21.12 [+11.2, +32.2] | 1.00 | +case(target) boosts the whole **category** |

**Group A (semantic nulls) all go to zero** — permuting the operator labels preserves every statistical
property of the extraction (same residuals, same averaging, same norms) and kills the effect entirely;
even a random direction *inside the operator subspace* does nothing. The contrast is carried by the
specific, correctly-labeled content.

**Group B (structural probes) stay nonzero for mechanistically informative reasons**, not as confounds:
(1) the margin decomposes like the representation does — injecting `case(other) − case(source)` keeps
the *uninstall-source* half and yields ≈ half the effect (+11 of +22.6); (2) `+case(target)` raises the
whole answer **category** (any capital city), which is why the margin toward a *shuffled* operand's
answer also moves — **operand specificity lives in generation, not in the margin** (the composed-state
patch says the *correct* operand's answer at 51.8% and the stem-swap redirects it, §2.11); (3) a vector
added at an early layer **persists** in the residual stream into the workspace, so "wrong layer" is not
actually absent from the right layer — per-layer locality is measured properly by the layer sweep
(§2.14), where building and injection co-vary. Artifacts: `1.7b_relations_nulls.parquet` + `.json`.

### 2.14 Layer sweep: represented early, causal late, readable everywhere (2026-07-11)

`op_layer_sweep.py` (1.7B relations; the direction is built AND injected at each single source layer).
Three per-layer profiles of the same object:

- **Trivial readability:** leave-one-operand-out operator classification is **100% at every layer,
  including layer 0** — attention mixes the operator token into the query position immediately, so
  decodability-style evidence locates nothing.
- **Representation:** the ANOVA operator share peaks **early** — 92.4% at L5 (18.5% depth) — and decays
  monotonically to 56.5% at the end of the stream.
- **Causal leverage:** the single-layer swap contrast is ≈0 early, rises through the band, and peaks
  **late** — +21.28 [+13.8, +28.2] at L24 (88.9% depth), essentially the full band's +22.6 from ONE
  layer.

The two informative curves peak ~70 depth-points apart. Bonus diagnosis: op_minimal's "single
mid-workspace layer" (L17, +6.8) was far below the late-band optimum — the minimal high-effect
intervention is one LATE layer at the query position. Artifacts:
`1.7b_relations_layersweep.parquet` + `.json`.

### 2.15 The animals domain: everything replicates, one floor caveat (2026-07-11)

An independently curated non-geographic domain (`data/animals.json`: class/habitat/diet/covering × 12
animals, tokenizer-screened by `build_op_datasets.py --only animals`, 3 paraphrase frames). 1.7B:

- **Swap:** 12/12 ordered pairs flip; contrast **+18.08 [+13.53, +22.81]**, flip fraction 1.00.
- **Held-out operands:** 12/12, **+18.04 [+13.59, +22.74]** — transfer ≡ within.
- **Cross-frame:** 72/72 flips, mean +17.26, frame-invariant to two decimals (+17.50 / +17.50).
- **Nulls:** permuted per-operand **−0.53 [−5.1, +3.0]**, global +1.10, subspace +0.17 — all at zero.
  Structural probes replicate quantitatively: other-relation retains ≈half (+8.5 of +18.1 = 47%; the
  countries figure is 49%), shuffled answers ≈ full (+18.1, category-level boost), wrong layer +10.7.
- **Factorization:** query position operand 16.3% / operator 61.7% / **interaction 22.0%**; entity
  token operand-dominant 79.4%. More fusion than countries (9%) — yet transfer is perfect, while
  arithmetic (23% interaction) fails held-out: the domain discriminator is TRANSFER, not fusion share.
- **Minimal band:** single layer 12/12 (+6.88), full band +18.08; off-task KL 6.3 vs 19.5 nats.
- **Floor caveat:** clean greedy accuracy is only **6%** (multi-token answers; single-token coverage
  0.62), so the generation-level readout is uninformative here — composed patch 2.1% = full donor 2.1%
  (consistent, at floor), Δmargin discriminates as usual (+6.8/+7.3 vs +3.5 magnitude control).
  Sufficiency in animals is attested by margins/rank, not exact match.

Artifacts: `1.7b_animals_{operator_swap,heldout,templates,nulls,minimal,patch_decomp,positions,dose}*`.

### 2.16 The generation audit: classify what it SAYS, and force the choice (2026-07-11, reader round)

A reader pushed on the generation metrics: containment-at-k=3 and a two-token margin could miss
syntactically adapted answers, and nobody had classified the raw generations. `op_audit.py` (1.7B
relations, 224 cells/condition): k=8 greedy with **raw texts persisted**, a 5-way classification
(target / source / other-operand / other-relation / degraded), and **forced choice** = length-normalized
full-sequence log-prob among the operand's gold answers *under the same intervention*:

| condition | target | source | degraded | **forced choice** |
|---|---:|---:|---:|---:|
| clean | 3.6% | 58.5% | 36.6% | 2.7% |
| patch composed (μ+stem+case) | **57.6%** | 11.6% | 29.9% | 76.3% |
| patch donor | 57.6% | 9.8% | 31.7% | 84.4% |
| add band α=4 (old default) | 0.4% | 0.0% | **94.6%** | **80.4%** |
| add band α=0.1 (calibrated) | **59.4%** | 0.9% | 38.4% | **87.5%** |
| add 1 layer, query, α=1 | 40.6% | 21.9% | 35.3% | 76.3% |

**Three findings.** (1) The overdose reading is *confirmed by inspection*, not assumed: at α=4 the
generations are token loops ("dollar dollar dollar…", "出处出处…") — category often correct, fluency
destroyed — NOT surface variants the metric missed. (2) At the calibrated dose, additive steering's
free generation (59.4% fluent target) matches the composed patch (57.6%) and the clean prompt's own
source-rate (58.5%): full behavioral sufficiency under the strictest reading. (3) **Forced choice
dissociates information from fluency**: the target wins among the gold answers at 80.4% *even at the
overdose* — overdosing destroys the generation manifold, not the encoded preference. Caveat, both
ways: containment can also over-count (2–5% of cells contain target AND source — "the Swedish krona"
contains demonym-"Swedish" as a modifier); worst-case reclassification moves target rates by ≤5 points
and changes nothing. Artifacts: `1.7b_relations_audit.parquet` (with raw texts) + `.json`.

---

## The open interpretive question (deliberately unresolved)

The evidence splits cleanly into a negative half and a positive half:

- **Negative (well controlled):** the J-space is not more *readable* than the raw
  residual on Qwen3. Four comparisons agree; a random projection matches the
  fitted J-lens; the concept channel is a magnitude artifact.
- **Positive (well controlled):** the workspace is causally a *concept-routing
  node* — intervening on it reroutes the downstream concept (and the noun that
  realizes it), robustly, localized to the band, sharpening with scale. And
  relations are a structured paradigm of causally-manipulable **case-directions**
  (§2.3): 20/20 case-swaps flip, random ~0, with operation separated from
  realization (language ≠ demonym as cases despite the same output word).

Framings this could support, none yet chosen:

1. **Legibility ≠ causal importance.** The headline is the dissociation itself:
   a subspace can be causally load-bearing without being more linearly readable
   than the stream it sits in. Novel, honest, contrarian to the readout framing.
2. **The workspace declines concepts by role.** Center on §2.3: relational
   computation as a case paradigm (function vectors with paradigm geometry, and
   the operation/realization split). The nulls become the "not via a special
   readout" controls. This is the richest positive story.
3. **The negative result alone.** "The J-lens readout advantage does not replicate
   on Qwen3 under controls" — modest, defensive, solid.

Open threads that could change the picture before deciding:
- Everything here is 1.7B (causal also 8B). Does the readout null hold at 8B/32B,
  does the causal reroute keep sharpening, and does the case paradigm survive at
  scale and generalize to held-out stems (a case-ending applies to any stem)?
- `a/an` is a poor morphology probe in this model (determiner not look-ahead). A
  clean downstream-realization test would need a construction where the surface
  form provably depends on the dereferenced concept.
- The case paradigm is causal/representational; whether it is specifically a
  *J-space* phenomenon is answered "no" so far (§2.3: cases are cleaner in the
  raw residual; J-space reflects output form).

## Reproducing these numbers

```
python scripts/lens_eval.py 1.7b --set lemma-form      # 1.2
python scripts/syntax_probe.py 1.7b                     # 1.3
python scripts/syntax_probe.py 1.7b --lens out/lenses/1.7b-randproj.pt
python scripts/syntax_swap.py 1.7b                      # 2.2
python scripts/causal_swap.py 1.7b                      # 2.1 (existing)
python scripts/operator_paradigm.py 1.7b               # 2.3 (also 8b)
python scripts/operator_factorize.py 1.7b              # 2.4 (also 8b)
```

Relational stimuli: `data/relations.json`.

Datasets: `data/morphosyntax/`. Control lenses: `out/lenses/*-randproj.pt`,
`*-permvocab.json` (`scripts/control_lens.py`).
