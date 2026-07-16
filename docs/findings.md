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
Two later programs sharpen that into a shape the original question did not
anticipate: the causal vector's *readable* part is not its lever (§2.18), and the
operator half of the factorization has removable tissue while the operand half
does not, at either position we have looked (Part 3).

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

**8B replication (evening):** query-only +32.7 ≈ all +33.2 at 63× lower off-task KL (0.35 vs 21.9);
operand/wrong ≈ 0; single/query rank 748 → 67. The dose inverted-U reproduces at the SAME peak doses:
band α=0.1 → 56.2%, single-layer α=1 → 45.1%. Artifacts: `8b_relations_positions*`, `8b_*_posdose`.

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

**8B replication (night):** semantic nulls even tighter at zero — per-operand +0.15 [−1.8, +3.1],
global +0.09, subspace +0.43, vs real +26.0. The other-relation probe retains ≈half at all three data
points: 49% (1.7B countries), 47% (1.7B animals), 52% (8B). Artifacts: `8b_relations_nulls*`.

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

**8B replication:** the same profile almost to the point — case share peaks at L8 (22.9% depth,
84.7%), causal at L31 (88.6% depth, +23.1); decodability 100% everywhere. Artifact:
`8b_relations_layersweep*`.

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

**8B animals (night):** swap 12/12, contrast +25.8 [+9.8, +34.8] (sharpens over 1.7B's +18.1); the
generation floor persists at scale (clean ceiling 8%; composed 7.6% ≈ donor 8.3% — the
composed=donor=ceiling identity holds even at the floor, and rank still discriminates: 62 composed vs
1670 magnitude control). Nulls replicate: per-operand −0.69 [−4.3, +3.2], subspace −1.79, global +2.47
(CI spans 0); other-relation retains ≈half (52%) — the fourth consistent data point for the margin
decomposition. Artifacts: `8b_animals_*`.

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

### 2.17 Round 3 (user review): equivalence, robustness partitions, LOO-operator, Wang & Xu (2026-07-12)

- **Missing prior integrated:** Wang & Xu 2025 (arXiv:2504.14496), "Functional Abstraction of
  Knowledge Recall" — subject=input argument, relation=function body, object=return value, verified by
  component patching. Same group as Wang et al. 2024. New Related-Work paragraph states the delta:
  they exchange components across positions; we quantify the joint decomposition of ONE prediction
  state, show steering vector ≡ operator main effect, competitive nulls, generative sufficiency.
- **Paired equivalence (`op_equivalence.py`, recompute):** composed−donor +0.9pp [−6.9,+8.3] (1.7B),
  −3.1pp [−12.5,+5.2] (8B); interaction adds −0.9pp [−8.3,+6.9]. TOST at ±5pp does NOT certify
  (5 operator clusters = underpowered) — paper now says "no detectable gap", not "matches".
  Held-out-cell composed − donor = −15pp with CI excluding zero at 8B: quantified honestly.
- **Held-out beyond one split (`op_heldout_parts.py`):** 5 shuffled partitions — every pair flips in
  every partition, contrast range [+20.0, +22.9]; LOO-country 220/224 flips, range [+11.5, +28.1]
  (weakest: Turkey 16/18, Brazil 18/20).
- **Leave-one-operator-out (recompute):** dropping any operator keeps flips at 1.00; contrast range
  [+18.8, +25.8] (1.7B), [+23.3, +28.9] (8B). No single operator carries the paradigm.
- **Editorial:** three named generation readouts (short-decode containment / long-decode
  classification / forced-choice sequence probability), formal SS shares + "two-way effects
  decomposition" (not inferential ANOVA), "fusion"→"interaction (cell residual)" outside §5,
  "a state"→"per-layer factorized residual trajectory", "relation-conditioned retrieval operator",
  abstract −25%, §4.9 → 2-sentence provenance note, exponent glossed.
- **Format:** real title in main*.tex (was stale!), claims table → wrapped p{} columns, adjustbox
  shrink-only (resizebox was ENLARGING narrow tables), appendix → \onecolumn with non-floating
  tables/figures pinned to their prose, print figures redesigned at ~7in with 9pt+ type and no
  web footers. Discovered: tectonic writes no main.log — the "4 overfulls" tracked all day were a
  stale log from Jul 10; real count is 0.
- **Out-of-sample dose (`op_dose_oos.py`):** the strongest possible outcome — α* identical in ALL
  five calibration/evaluation partitions (0.1 band, 1.0 single-layer); frozen-α* generation on the
  disjoint half 60.8% [56.2, 67.3] (band) / 42.2% [38.6, 48.2] (single layer); the evaluation-optimal
  dose coincides with the calibrated one in 10/10 splits. The on-manifold dose is a transferable
  property, not a post-hoc pick. 8B held-out partitions also replicate (all flip, [+22.8, +25.6];
  LOO 222/224). Artifacts: `1.7b_relations_dose_oos.*`, `8b_relations_heldout_parts.*`.

### 2.18 The vocab-semantics battery: the case vector reads, but its readable part is not the lever (2026-07-14)

Todd et al. (ICLR'24, Tab. 5) decode function vectors to answer-space exemplars; Nadaf (2026) finds FV
projections "universally incoherent" despite >0.9 steering accuracy. Both are correlational
projections. We have the causal object — the case vector *is* the steering vector *is* the operator
main effect (§2.11) — so we can ask the question causally: split the vector into the part a logit lens
can see and the part it cannot, and inject each **alone at the calibrated dose** (§2.12).

- **P1, it reads, and what it reads changes with depth (`op_vocab_portrait.py`).** The case vector's
  diagonal contrast (its own relation's answers vs. the other relations') rises monotonically with
  depth in all three models: L11 +1.81 → L24 **+7.94** (1.7B), L14 +3.03 → L31 **+10.05** (8B),
  L17 +5.07 → L37 **+12.01** (Gemma-2-9B). Controls at the lever layer are flat: permuted-label
  vectors max +1.32/+1.45/+1.64, random-subspace max +2.76/+3.59/+5.52, and a **random-projection
  lens reads nothing at all** (+0.11 at 1.7B, −0.08 at 8B). So Nadaf's "incoherent" is too strong on
  these models — but see the next bullet for what the coherence is worth.
- **P2, the headline: the readable part is not the lever (`op_vocab_causal.py`).** Split
  `case = c_ans ⊕ c_rest`, where `c_ans` is the exact orthogonal projection onto the span of the
  γ-weighted unembedding rows of the relation's answer tokens — *the only part a logit lens could ever
  attribute to "answer directions"*. It holds a **mean 7.3% of the vector's energy** (range 1.4–14.9%
  over 20 pairs) at 1.7B, 5.3% (1.6–9.8%) at 8B, 2.9% (1.3–5.4%) at Gemma. At the calibrated
  single-layer dose:

  | model | `c_ans` alone | `c_rest` alone | full vector |
  |---|---|---|---|
  | Qwen3-1.7B | +0.9 [+0.2, +1.4], **0% flip** | +10.8 [+6.5, +14.7], 90% flip | +12.1 [+7.0, +16.5], 100% |
  | Qwen3-8B | +0.8 [+0.5, +1.1], **0% flip** | +11.6 [+8.0, +15.1], 85% flip | +13.2 [+9.0, +17.1], 90% |
  | Gemma-2-9B | +1.1 [+0.3, +1.5], **0% flip** | +16.1 [+11.2, +20.9], 100% flip | +17.3 [+11.9, +22.1], 100% |

  The readable part's margin effect is *positive and tiny* — the CI excludes zero, so this is not "it
  does nothing", it is "it does ~1/13th of the job and flips not a single pair in any model". The
  complement carries 89% / 88% / 93% of the full margin effect. **Decodable, but not the lever:
  identity travels in the ~93% no logit lens can read.**
- **The generation split does NOT scale, and we said it did.** Exact-match generation under the donor
  patch: `c_ans` 8.9% (1.7B) / **29.5% (8B)** / 4.9% (Gemma); `c_rest` 26.8 / 54.5 / 59.8; full 57.6 /
  77.7 / 63.4. The 8B is the odd one out and **we have no account of why**. This retracts the
  scale-trend reading of the 1.7B→8B pair: with a third model it is not a trend, it is an outlier.
- **A third face of the dose artifact (§2.12).** At the 4× band overdose, `c_ans` *alone* inflates the
  margin +34.6 [+19.3, +52.9] at 1.7B and **+65.7 [+51.9, +79.7]** at 8B — more than the full vector.
  But so does a norm-matched vector built from the **wrong relations'** answers (+17.8 [+4.7, +30.7],
  65% flips), and so do random vectors in the answer-token span (+5.4, +4.7). At high dose, margins in
  vocabulary-aligned directions are **mechanically inflatable**; the calibrated dose is the honest
  readout, and any "the answer directions matter" claim measured at overdose is measuring the dose.
- **The purest test: a marker with no answers in it.** Build `language − demonym` from only the 8
  countries where both relations share an answer word ("Italian"), so the marker carries almost no
  answer-token energy (**4.96%** of ‖m‖² at 1.7B, 2.36% at 8B, 1.71% at Gemma, vs. random expectations
  of 0.83 / 0.40 / 0.44%). Inject it on the 4 held-out countries where the answers *differ*: it moves
  preference in the predicted direction in **100% of cells, in both directions, in all three models**
  (Δ lang margin +12.27 [+9.76, +14.70] / +15.56 [+12.36, +18.77] / +17.60 [+13.74, +20.31] pushing
  one way; −11.09 / −13.27 / −16.12 pushing the other). You can steer **which question is asked**
  without touching **which word comes out**.

Artifacts: `{1.7b,8b,gemma-2-9b}_relations_vocab_portrait_summary.json`, `*_vocab_causal_summary.json`,
`*_marker_causal_summary.json`. Every number above is checked by
`python scripts/verify_numbers.py` (VOCAB/MARKER blocks).

---

## Part 3 — The lesion study: tissue, not vectors (2026-07-14/15)

Parts 1 and 2 are divided by method, and this is a third method. Everything above intervenes on the
residual stream at **block** granularity — the right unit for steering, the wrong one for asking which
*pieces* implement the operation. Neuropsychology's move is: localize a function with a contrast,
remove the tissue, and measure whether the deficit is **selective**. That needs units you can remove.

Two differences from every experiment above. The units are **heads and MLP neurons**, hooked inside
the block (`layers[l].self_attn.o_proj` pre-hook = the concatenated per-head outputs;
`layers[l].mlp.down_proj` pre-hook = `act_fn(gate(x)) * up(x)`, *the* MLP neurons). And a lesion is
**not a steer**: the unit is silenced at *every token position of every prompt*, pinned to its
WikiText mean, and the prompt asks the **true** question. We are not pushing the model toward an
answer; we removed tissue and are watching what breaks.

### 3.1 The infrastructure, and its ground-truth check (`lesion_core.py`)

Zeroing 16 head slices at `o_proj`'s input equals zeroing that layer's attention output to 4 decimal
places — the hook points are the units they claim to be. Every battery run begins with an assert that
an **empty lesion is a bit-exact no-op** (`lesion_battery.py:196`, tolerance 1e-4): if the harness
perturbs the model by existing, nothing downstream means anything.

### 3.2 The criticality map: a tiny brainstem and a large redundant cortex

Ablate every head **alone** and record the damage. Brains have this structure — a millimetre of
brainstem is fatal, a lobe of "silent" cortex can be lost quietly — and if LLMs share it, a lesion
study must know where the infrastructure is *before* claiming any deficit is about function.

Qwen3-1.7B: **2 of 448 heads** exceed 2× perplexity. L1H5 alone costs **×19.7**, L0H3 **×4.4**; the
median head costs **×1.003**. That is the whole map. It is why every control below is
**depth-matched**: a globally random draw that happens to include L1H5 "beats" any targeted lesion for
reasons that have nothing to do with function (§3.6 is that prediction coming true).

**This map is 1.7B-only.** At 8B the depth-matching rationale is currently justified by analogy, not
measurement. Artifacts: `1.7b_criticality_summary.json`.

### 3.3 The anchor: a function with a known answer (induction heads)

Before trusting the method on an unknown network, run it on one the field has already solved. Top-48
induction heads, mean-ablated: the induction score collapses **+12.17 → +2.59 (−79%)** while WikiText
perplexity is spared (**×1.24**), arithmetic is untouched (**100%**), and relational retrieval is
*unharmed* (65.0% vs. a 61.7% baseline). The depth-matched controls do nothing to copying: layer
+11.70, magnitude +12.14. The method removes a known function selectively.

### 3.4 The operator network lesions cleanly, at both scales

Lesioning operator-selective neurons produces the pre-registered signature — the model answers the
**right country under the wrong relation** — at **every lesion size**, dose-ordered, in both models.
The test is a one-sided Fisher exact of the class's cell count against the **pooled control null** for
its kind (36 control runs = 2160 cells), Bonferroni-corrected over all 42 tests (α = 1.2e-03):

| model | k | accuracy | control band | `other_relation` | vs. null | p |
|---|---|---|---|---|---|---|
| 1.7B | 32 | 40.0% | [60.0, 61.7] | 6/60 | 1.1% | 8.9e-05 |
| 1.7B | 128 | 36.7% | [60.0, 63.3] | 7/60 | 1.1% | 9.2e-06 |
| 1.7B | **512** | **18.3%** | [45.0, 60.0] | **10/60** | 1.1% | **5.5e-09** |
| 1.7B | 2048 | 16.7% | [61.7, 63.3] | 8/60 | 1.1% | 8.6e-07 |
| 8B | 128 | 71.7% | [80.0, 81.7] | 3/60 | 0.1% | 3.5e-04 |
| 8B | **512** | **46.7%** | [73.3, 81.7] | **5/60** | 0.1% | **6.4e-07** |
| 8B | 2048 | 35.0% | [75.0, 80.0] | 4/60 | 0.1% | 1.6e-05 |

Perplexity is spared throughout (×1.06–×1.22 at 1.7B, ×1.03–×1.14 at 8B) and arithmetic is intact, so
the gate the method pre-registers — *"perplexity spared; otherwise the deficit is damage, not
localization"* — passes on every row above.

**An honesty correction to `b65dbc7`.** Its message says the three matched controls "do nothing". They
do not all do nothing: at k=512 the **magnitude control drops to 45.0%** against a 61.7% baseline — a
16.7-point hit. The defensible statement is that the targeted lesion sits **26.7 points below the
worst matched control**, which is still the result, stated at its real size. The scoring now prints
the control band on every row so this cannot be rounded off again.

### 3.5 The operand network does not lesion — at either read position. The dissociation is one-sided.

The pre-registered operand prediction is `other_operand`: the relation survives, the entity is lost —
another country's capital. It essentially never happens.

- **At the query position:** null at k=32/128/512 in both models (accuracy inside the control band).
  The one exception is 1.7B k=2048, and it is instructive: **4/60 cells vs. a 0.4% null, p=2.0e-04** —
  formally significant, and the predicted class. But accuracy moves only 61.7% → 53.3%, and those 4
  cells (6.7%) **exactly equal the single highest of the 36 control runs**. So the pooled test calls
  it enriched and the control envelope calls it ordinary; both are true, and it does not replicate at
  8B (unstructured at every k). This is weaker than `b65dbc7`'s "NOT lesionable", and weaker than a
  result: it is a hint at one scale, at the edge.
- **At the entity token — the pre-registered test of `b65dbc7`'s account, and it fails.** That commit
  proposed a mechanism: the entity reaches the query position by attention *from* the entity token, so
  it is redundantly coded — erase it at the query and the model re-imports it next forward pass. The
  operator has no second home. **Prediction: the operand must be lesionable where it is not
  redundant.** We localized it there and lesioned it at 7 sizes in 2 models:

  | model | k | accuracy | ppl | signature | detail |
  |---|---|---|---|---|---|
  | 1.7B | 512 | 46.7% | ×1.07 | `unstructured` | no class clears the null |
  | 1.7B | **2048** | **11.7%** | **×1.48** | `unstructured` | `degraded` 81.7%, `other_operand` **1.7%** |
  | 8B | 512 | 48.3% | ×1.04 | `other_operand` | 2/60, p=7.2e-04 — 2 cells |
  | 8B | **2048** | **28.3%** | ×1.08 | **`other_relation`** | **6/60, p=2.4e-08** — the *operator* signature |

  At 1.7B the accuracy collapse is real but it is **destruction**: ×1.476 perplexity, the largest cost
  of any targeted neuron lesion in the battery, with 81.7% of cells degraded and the predicted class
  at one cell. At 8B the deficit is cleaner *and points the wrong way*: six cells of the operator's
  signature to the operand's one.
- **The deficit is ranking-specific, so this is not generic entity-position fragility.** At 1.7B
  k=2048 the depth-matched controls for the same network do nothing (68.3 / 58.3 / 61.7%). The ranking
  selects something causally load-bearing at the entity token. It is not the operand.
- **The readout was stacked in favour of the prediction that failed.** `op_audit.classify` tests
  `other_operand` **before** `other_relation`, over ~2.8× more candidate strings per cell (10.33 vs.
  3.73). The operand had priority and a wider net, and lost anyway.

So: the operator has removable tissue; the operand does not, at either place we have looked for it.
**Two hypotheses are now dead** — query-position, and entity-position — and the mechanism of the
operand half is an open question, not a footnote.

### 3.6 What the head-level rows can and cannot say

The head arm at 1.7B is confounded, **exactly as §3.2 predicted it would be**, and the confound is the
evidence rather than the bug:

- The **globally random control** — kept only to expose the depth confound — catches L1H5 at k=16 and
  k=48 and posts perplexity **×38.4** and **×78.4**. That row is the argument for depth-matching, in
  the artifact, on demand.
- `operand/top heads k=48` swallows L0H3: accuracy 0.0%, arithmetic 0.0%, perplexity **×485**. The
  gate marks it `uninterpretable` automatically, and the summary now records *which* critical head it
  caught rather than leaving the reader to guess.
- **Excluding the sink heads would not rescue the regime.** `operand/magnitude heads k=48` costs
  **×17.5 perplexity while containing no head that individually exceeds ×1.1**. Joint 48-head lesions
  are superadditive in a way the single-head screen cannot predict, so the k=48 head regime at 1.7B is
  simply beyond the screen's warrant. It is reported, not interpreted.
- **At 8B the head arm is null but clean** — every head lesion stays under ×1.04 perplexity and no
  targeted row produces a structured class. Uninformative for the opposite reason. Whatever the
  operator network is, at 8B it is not a small set of heads.

### 3.7 Two rankings, not four networks

A methodological wart that is load-bearing for how the geometry reads. `lesion_localize.py` defines
`operand := −operator` and `operand_entity := −operator_entity` on the same per-unit contrast. So
**`jaccard(operator, operand) = 0.000` is a tautology, not a finding** — no "the networks are
spatially segregated" claim can rest on it, and the number is now excluded from the printed summary
rather than left to be misread. There are two rankings and two read positions, not four independent
networks.

The Jaccards that *are* informative are cross-position, and they are strikingly stable across scale:
`operand|operand_entity` = **0.185 / 0.188** (1.7B / 8B), `operator|operator_entity` = 0.096 / 0.133.
The same function at two positions shares ~1/5 of its tissue.

The geometry that survives is real. Over neurons, layer entropy (1 = distributed, 0 = single-layer
area) against a 0.99 random-unit null:

| network | 1.7B | 8B | CoM depth (1.7B / 8B) |
|---|---|---|---|
| operator | 0.64 | 0.60 | 89% / 92% |
| operand | 0.59 | 0.56 | 90% / 92% |
| operator_entity | 0.69 | 0.58 | 87% / 93% |
| operand_entity | 0.71 | 0.77 | 84% / 81% |
| text_symbolic | **0.97** | **0.96** | 47% / 41% |

The relational networks are **concentrated** and sit at 84–93% depth — the paper's causal lever, found
again by a method that knows nothing about the paper. The text-vs-symbolic contrast is **fully
distributed** (0.96–0.97, indistinguishable from the null) and sits mid-stack. **Whether a function
has an "area" or a "network" depends on the function** — and that is a claim about this model, from
this model's own data.

### 3.8 What is still open

- **`operator_entity` is localized and never lesioned.** The column exists, the region stats now
  exist, the lesion runs do not. It is the missing cell of the 2×2 (~3h34m of sweep at measured
  rates). It would not change §3.5 — the depth-matched controls already exclude generic
  entity-position fragility — but the 2×2 is not closed until it runs.
- **No 8B criticality screen**, so §3.2's map, and the depth-matching it licenses, is 1.7B-only.
- **The per-cell generations are not persisted.** `run()` classifies each of the 60 cells and keeps
  only the class *distribution*; the raw text is discarded. So no classification above can be
  re-audited without a re-run — which is exactly the objection §2.16 was written to answer, reopened
  at a new granularity. Fix before the next run.
- **The `--networks` / `--stage` writes used to clobber.** They wrote the current run's rows over the
  whole battery, which is why the P1 anchor rows (§3.3) are absent from the 84-run artifact and had to
  be read back from `out/lesion/1.7b/anchor2.log`. Now merged on the run's identity; §3.3's numbers
  return to the artifact on the next anchor pass.

Artifacts: `{1.7b,8b}_relations_lesion_summary.json`, `*_localizer_summary.json`,
`1.7b_criticality_summary.json`. Checked by `python scripts/verify_numbers.py` (LESION block).

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

Part 3 adds a third category that fits neither half: causal, well controlled, and
**asymmetric**. The two halves of a factorization that is symmetric in the algebra
(`H = μ + stem[o] + case[k] + inter`) are not symmetric in the tissue — one has a
removable network and the other does not, at either read position. Nothing in the
readable/causal split anticipates that, and we cannot currently explain it.

Framings this could support, none yet chosen:

1. **Legibility ≠ causal importance.** The headline is the dissociation itself:
   a subspace can be causally load-bearing without being more linearly readable
   than the stream it sits in. Novel, honest, contrarian to the readout framing.
2. **The workspace declines concepts by role.** Center on §2.3: relational
   computation as a case paradigm (function vectors with paradigm geometry, and
   the operation/realization split). The nulls become the "not via a special
   readout" controls. This is the richest positive story. **Part 3 both
   strengthens and narrows it:** the paradigm is a paradigm of *operators* — they
   are what has an identifiable, removable, dose-ordered network (§3.4) — while
   the operand's mechanism is now an open question rather than a symmetric
   partner (§3.5). "Declines concepts by role" would have to be a claim about the
   case-endings, not about the stems.
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
- **Where does the operand live?** Two hypotheses are dead (query position §3.5,
  entity token §3.5) and the second died having been *predicted* by the first's
  post-mortem. The factorization's operand term is causally real — held-out
  transfer, compositional sufficiency, §2.4 and §2.11 — yet it has no tissue we
  can find. Either it is genuinely distributed in a way the operator is not, or
  the localizer contrast (a per-unit variance decomposition) is the wrong
  instrument for it. Deciding that is the next real experiment, and it is not
  another lesion sweep.

## Reproducing these numbers

```
python scripts/lens_eval.py 1.7b --set lemma-form      # 1.2
python scripts/syntax_probe.py 1.7b                     # 1.3
python scripts/syntax_probe.py 1.7b --lens out/lenses/1.7b-randproj.pt
python scripts/syntax_swap.py 1.7b                      # 2.2
python scripts/causal_swap.py 1.7b                      # 2.1 (existing)
python scripts/operator_paradigm.py 1.7b               # 2.3 (also 8b)
python scripts/operator_factorize.py 1.7b              # 2.4 (also 8b)
python scripts/op_vocab_portrait.py 1.7b               # 2.18 P1 (also 8b, gemma-2-9b)
python scripts/op_vocab_causal.py 1.7b                 # 2.18 P2 + the marker
python scripts/lesion_localize.py 1.7b                 # 3.1, 3.7 (also 8b)
python scripts/lesion_battery.py 1.7b --stage screen   # 3.2
python scripts/lesion_battery.py 1.7b --stage anchor   # 3.3
python scripts/lesion_battery.py 1.7b --stage dissoc   # 3.4-3.6 (also 8b)
```

Two of those re-score without a model, from the tracked summaries — the `.parquet`
artifacts are gitignored, so this is what a fresh clone can actually run:

```
python scripts/lesion_battery.py 1.7b --rescore         # re-score the battery, no GPU
python scripts/lesion_localize.py 1.7b --rescore        # recompute the geometry, no GPU
python scripts/verify_numbers.py                        # every number above, against results/
```

Relational stimuli: `data/relations.json`.

Datasets: `data/morphosyntax/`. Control lenses: `out/lenses/*-randproj.pt`,
`*-permvocab.json` (`scripts/control_lens.py`).
