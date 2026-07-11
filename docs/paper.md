# Causal operator–operand factorization in the residual stream of LLMs

*Subtitle: relational computation as declension — the model marks concepts by their functional
role. The case metaphor is our explanatory device; the claims below stand on the causal
interventions and the measured factorization.*

[:material-file-document: Download the PDF](assets/paper.pdf){ .md-button .md-button--primary }
[:material-play-circle: Interactive explorer](explorer.md){ .md-button }

*Draft, 2026-07-10. Qwen3-1.7B/8B + Gemma-2-9B. All numbers reproducible from `scripts/` and `data/`; see
["How to reproduce"](reproduce.md) and the [working findings log](findings.md). An
[interactive explorer](explorer.md) animates the three central results (declension, injection,
syncretism) and, for readers new to grammatical case, explains the linguistic analogy.*

## Abstract

We study how open-weights transformers (Qwen3-1.7B/8B, Gemma-2-9B) represent a **relational operation**
applied to an entity — *currency-of*, *capital-of*, *language-of*, *demonym-of*, *continent-of* applied
to a country.
We find that the operation is carried by a **manipulable operator direction** in the residual stream
that is largely **separable from the operand**. Adding `v(op_B) − v(op_A)` to a prompt performing `op_A`
converts the model's answer to `op_B` for all 20 ordered operator pairs (matched-norm random controls
null), and the direction **generalizes to operands never used to build it**, to **unseen paraphrase
frames** (every build→test frame combination flips), and **across architectures** — the full pipeline
replicates on Gemma-2-9B with the largest effect sizes of the three models tested. A two-way analysis of
variance factorizes the representation as `H[operand, operator] = μ + operand + operator + interaction`,
~90% additive with a ~10% **fusion** interaction, operator and operand occupying largely orthogonal
subspaces; the operation-dominant representation *emerges along the sequence* (operand-dominant at the
entity token → operator-dominant at the query token). Strikingly, the operation is **separable from its
surface realization**: *language-of* and *demonym-of* are distinct operator directions even though both
emit "Italian". As an explanatory device we borrow a morphological metaphor — the model **declines** a concept by its
functional role (case), with **syncretism** (shared surface form) and **fusion** (non-additive
stem+ending) as in a Latin declension. Testing generalization across domains, the factorization is
**specific to relational retrieval**: for **arithmetic** (+, ×, −) and **comparison-logic** operators,
the interaction term is 2–4× larger and the operator **fails to generalize to held-out operands**,
consistent with arithmetic being computed by a "bag of heuristics" rather than a linear operator. A
control that instead varies a *linear numeric parameter* (add-N) rather than the function generalizes
better and is near-collinear, reconciling a contrary recent report (Christ et al. 2025). Finally,
the structure is **causal, not a readout advantage**: the Jacobian-lens ("J-space") readout (Gurnee,
Sofroniew, Lindsey et al. 2026) is not more
legible than the raw residual on any of our metrics under matched controls.

## 1. Introduction

A line of work shows that a *task* or *relation* can be captured by a single addable vector in an LLM's
residual stream (task vectors, Hendel et al. 2023; function vectors, Todd et al. 2024) and that many
relations are approximately linear operators (LRE, Hernandez et al. 2024; Merullo et al. 2024). What has
not been characterized is the **joint structure of operator and operand**: whether the operation
*factorizes* from its argument, whether that factorization *generalizes*, whether the operation is
separable from the *word it produces*, and whether any of this holds *beyond* relational facts, in
arithmetic and logic. We answer these on Qwen3 and Gemma-2, and offer a morphological reading — relational computation
as **declension** — as an organizing metaphor that anticipates the specific asymmetries we observe
(syncretism at the level of the surface form, fusion in the interaction term).

## 2. Related work and what is new here

**Operation as an addable vector.** Task Vectors (Hendel et al., EMNLP 2023) and Function Vectors (Todd
et al., ICLR 2024) extract one vector per ICL task and add/patch it to trigger the task. In-Context
Vectors (Liu et al., ICML 2024) and ActAdd (Turner et al. 2023) steer a task/style similarly. These
bundle operator *and* operand into one unfactored "task" direction and (for FVs) show *partial*
cross-task arithmetic without null controls. We contribute the **factorization** (operator ⊕ operand),
an **exhaustive all-pairs swap with matched-norm nulls**, and a **held-out-operand generalization** test.

**Relation as a linear operator.** LRE (Hernandez et al., ICLR 2024) fits a full affine map `W·s + b` per
relation. That represents the relation as a *monolithic* operator; we show a **low-dimensional additive
operator *direction*** that is arithmetically composable and **factorable from the operand**, and we
quantify the operand/operator/interaction variance split, which LRE does not.

**Additive factual recall.** Summing Up the Facts (Chughtai et al. 2024) decomposes factual recall into
additive *circuits* (subject vs relation contributions). Ours is a **representational two-way ANOVA of
the residual state** with an explicit, measured **interaction/fusion term** — a different object.

**Attribute subspaces / reading position.** RAVEL (Huang et al., ACL 2024) disentangles multiple
*attributes of one entity* into subspaces (operators sharing an operand); we measure the orthogonal axis,
**operator vs operand**. The operand→operator shift along the sequence restates the subject-enrichment →
attribute-extraction dynamics of Geva et al. (EMNLP 2023) in operator/operand terms — our least novel point,
included as confirmation.

**Global workspace and the Jacobian lens.** Gurnee, Sofroniew, Lindsey et al. (2026) introduce the
Jacobian lens and identify a mid-depth "global workspace" band of the residual stream. We adopt their
depth band as the site of intervention and their lens as scaffolding; our contribution is orthogonal —
the operator/operand structure of what that band *holds* — and we additionally report a controlled null
on the lens's readout advantage for our task (§4.8).

**Arithmetic and logic.** Arithmetic in LLMs is reported as a "bag of heuristics" (Nikankin et al. 2025)
and Fourier/helical procedures (Nanda et al. 2023; Zhong et al. 2023; Kantamneni & Tegmark 2025), with the
operand *value* cleanly linear (Gurnee & Tegmark 2024). Truth is a steerable direction (Marks & Tegmark
2023) and logical-operator heads exist (Hong et al. 2024). No prior work reports a **cross-domain
operator/operand factorization**; we provide one and show it **breaks down** for arithmetic and logic —
itself a result.

**What is new here is the combination, not any single ingredient.** That relations admit addable
vectors (task/function vectors) and linear decoders (LRE) is established; what has not been done is the
joint, quantified analysis: (1) an operator ⊕ operand **factorization of the residual state itself**
(two-way ANOVA with a measured interaction/fusion term and operator/operand subspace angles); (2) an
**exhaustive all-pairs causal swap** with matched-norm nulls; (3) **held-out-operand generalization** as
the discriminator between a real operator and interpolation among build examples; (4) the
**operation-vs-realization dissociation** (language ≠ demonym as directions despite identical output);
and (5) a **cross-domain test** showing the factorization is specific to relational retrieval and breaks
down for arithmetic and logic. The declension/case reading is our *presentation* of these measurements —
an organizing metaphor that anticipates the observed asymmetries (syncretism at the exponent, fusion in
the interaction term) — not an additional empirical claim.

## 3. Method

**Setup.** Qwen3-1.7B/8B throughout, plus Gemma-2-9B for the cross-architecture replication (§4.5); all
run in bf16 on a single AMD Strix Halo APU (see [setup](setup.md)). A relation is rendered into a template; the canonical frame is
`"The {op} of {a} is"`, and §4.4 additionally uses two paraphrase frames (question–answer and
discourse-prefixed) that hold the `{op} of {a}` unit fixed while varying the surrounding frame, all
ending in *is* so answer scoring is comparable.
Reading position is the query token unless stated; the depth "workspace" band (38–92% of layers) is where
the J-space paper locates the global workspace and where we build operator directions.

**Operator direction.** For operator `k`, `v(k) = mean_operand[ h(operand, k) − mean_op h(operand, ·) ]`
at the workspace layers — the deviation of an operator from the operand's average over operators, averaged
over operands (the function-vector construction; `scripts/op_core.py:op_dirs`).

**Swap and controls.** Efficacy of `k_A → k_B` is the change in `logit(answer_B) − logit(answer_A)` at the
query position when `α·(v(k_B) − v(k_A))` is added over the band, versus a **matched-norm random**
direction. Answers are scored by their distinguishing token (bare digit for arithmetic, since Qwen3
splits " 3" into [space, 3]); a tokenization guard drops, within each operator pair, the operands whose
two answers collide at that token (for relations this leaves 4 of 12 operands on the syncretic
demonym↔language pairs and all 12 elsewhere).

**Factorization.** `H[operand, operator] = μ + operand(o) + operator(k) + interaction`; we report the
variance share of each term and the principal angles between the operand- and operator-subspaces
(`scripts/op_core.py:factorize`).

**Generalization.** Build `v(op)` from half the operands; test the swap on the held-out half. This
distinguishes a genuine operator from interpolation among the examples used to build it.

**Statistical treatment.** The 20 ordered swaps are **not** 20 independent observations: they are built
from 5 operator directions (each participating in 8 ordered pairs) and evaluated on the same shared
operand set (12 operands, 4 on the syncretic pairs after the tokenization guard).
We therefore report cluster-bootstrap percentile intervals (10,000 replicates) at two levels. Within a
pair, the **operand** is the resampling unit (per-pair 95% CIs). Across the paradigm, the **operator** is
the top-level cluster: a dyadic node bootstrap resamples the operator set with replacement, weights each
ordered pair by the product of its endpoints' multiplicities, and resamples operands within surviving
pairs (`op_core.bootstrap_pair_ci` / `bootstrap_family_ci`; per-operand values persisted in the
`*_long.parquet` artifacts).

**J-space controls.** We repeat readout-geometry in the J-lens readout `unembed(J·h)` vs the logit-lens
readout `unembed(h)`, and re-run efficacy with **spectrum-matched random-projection** and
**permuted-vocabulary** null lenses (`scripts/control_lens.py`).

## 4. Results

### 4.1 Relational operators are manipulable directions

All 20 ordered operator swaps flip the answer (mean swap **+21** logit units on 1.7B, **+25** on 8B) while
the matched-norm random control is ~0. Representative (1.7B):

| swap | clean | +operator | +random |
|---|---:|---:|---:|
| currency → capital | −7.9 | **+31.5** | −1.2 |
| capital → currency | −12.0 | **+33.0** | −2.5 |
| capital → language | −8.5 | **+32.5** | +3.0 |
| continent → language | −1.9 | **+23.0** | +1.4 |

![All-pairs operator swap: clean (baseline, "from" wins, blue) vs. injecting v(to)−v(from) ("to" wins, red). 20/20 pairs flip sign; matched-norm random control ≈ 0.](figs/op_swap.png)

*Injecting the operator difference `v(to) − v(from)` flips every one of the 20 ordered pairs: clean (left,
"from" wins, blue) → swapped (right, "to" wins, red). The matched-norm random control moves nothing.*

Under the operator-level cluster bootstrap (§3, the level that respects that pairs share directions), the
swap−random contrast is **+22.6, 95% CI [+14.0, +32.1]** at 1.7B and **+26.0 [+17.9, +32.8]** at 8B; the
flip fraction is **1.00 [1.00, 1.00]** at both scales — every replicate flips every pair. Per-pair
operand-bootstrap CIs never cross zero:

![Per-pair distributions: swap values per operand with operand-bootstrap 95% CIs, matched-norm random control, and clean baselines. No CI crosses zero; the weakest pairs are exactly the syncretic ones (demonym↔language and their neighbours).](figs/op_swap_dist.png)

*Every ordered swap, with uncertainty. Orange dots = per-operand swap values (median 12/pair; the
tokenization guard leaves 4 for the syncretic demonym↔language pairs); gray × = matched-norm random
control; black bar = operand-bootstrap 95% CI; blue ○ = clean baseline mean. The weakest effects are
precisely the syncretic pairs (`demonym ↔ language` and swaps into `demonym`) — where the two operations
share their surface form.*

**Dose–response and collateral cost.** Sweeping the intervention strength α (1.7B) separates the effect
into its parts: the **operator-specific** component (swap − random) rises and **saturates at ≈+23 by the
default α = 4**, while the matched-norm random control contributes a **nonspecific** margin shift of
≈+6 that is flat in α — any large perturbation degrades the clean answer's dominance somewhat, which is
why all headline numbers are swap − random contrasts. The intervention is *answer*-surgical, not
*distribution*-surgical: with the same hook active on unrelated WikiText, per-token KL(clean ‖
intervened) grows from 7.9 nats (α = 0.5) to 21 nats (α = 12) — 18.4 at the default — so the band-wide
edit substantially distorts off-task text. We report this as an honest cost of the band-wide,
all-position intervention rather than tuning it away; targeted (single-position) variants are the
natural mitigation and are left to future work.

![Dose-response: the operator-specific effect saturates at the default dose α=4 while the nonspecific random shift stays flat; per-token KL on unrelated text grows monotonically — the edit is answer-surgical, not distribution-surgical.](figs/op_dose.png)

### 4.2 The representation factorizes into operator ⊕ operand

Two-way ANOVA at a mid-workspace layer (1.7B / 8B):

| read position | operand | operator | interaction (fusion) |
|---|---:|---:|---:|
| query token | 5% / 6% | **86% / 82%** | 9% / 13% |
| entity token | **59% / 55%** | 31% / 34% | 9% / 11% |

~90% additive; operand/operator subspaces largely orthogonal (principal angles 41–85° at the query
token, across all three models). The operation-
dominant representation **emerges along the sequence**: the entity enters operand-dominant and is
*declined* into an operator-marked form by the query position (the reading-position control rules out
template echo; the causal swaps are the stronger control).

![PCA of the 60 workspace vectors H[operand, operator], colored by operator. At the country token the cases intermix (operand-organized, stem 59%); at the query token they separate into clean case clusters (operator-organized, case 86%). The faint web links each country's five case-forms — short at the country token, splayed at the query token. Bottom: the variance split, 1.7B and 8B.](figs/op_geometry.png)

*Where operand and operator live. Colored by operator throughout: at the **country token** the colors
intermix (the cloud is organized by operand); at the **query token** the same points separate into five
case clusters. Each country's five case-forms (faint web) start together and are pulled apart — the
concept is **declined** along the sequence.*

### 4.3 Operation ≠ realization (syncretism)

*language-of* and *demonym-of* emit the **same word** (Italian) yet are **distinct operator directions**;
the swap between them is weak precisely because they share an exponent. A **pure desinence** built from
the shared-output pairs (`mean[h(language) − h(demonym)]`, exponent cancelled) still installs the relation
causally (1.7B: −3.6 → +12.5). The syncretism appears at the *exponent* level, not the *case* level — as
in Latin declension.

![Left: operator-direction cosines — all five operators are distinct directions (no off-diagonal near ±1); language and demonym, boxed, both emit "Italian" yet differ (cos −0.26). Right: the pure desinence, built where the two share an output word, still installs the relation causally (−3.6 → +12.5).](figs/op_syncretism.png)

*Operation ≠ realization. Every operator is a distinct direction — including language and demonym (boxed),
which emit the identical word "Italian". A desinence built precisely where the two **share** their output
word (so the word cancels) still installs the relation: the case is separable from the word that realizes it.*

### 4.4 Operators generalize to held-out operands and across prompt frames

**Held-out operands.** `v(op)` built on 6 operands and applied to the 6 held-out operands still flips
**20/20** swaps — a genuine, transferable operator, not interpolation. Operator-level cluster bootstrap
on the held-out contrast: **+20.0 [+10.8, +29.5]** at 1.7B, **+22.8 [+14.0, +31.0]** at 8B; flip
fraction **1.00 [1.00, 1.00]** at both.

**Cross-frame transfer (paraphrase robustness).** To rule out template echo we render every relation in
three paraphrase frames — declarative (`The {op} of {a} is`), question–answer (`Q: What is the {op} of
{a}? A: It is`), and discourse-prefixed (`It is well known that the {op} of {a} is`) — and test every
build→test frame combination: `v(op)` built on frame *i*, all-pairs swap run on frame *j*. On 1.7B, **all
9 combinations flip 20/20** (180/180; off-diagonal transfer 120/120, mean contrast **+24.2**), and the
contrast is *frame-invariant to two decimals*: directions built on the declarative frame produce +22.6 on
all three test frames. The clean baselines do shift across frames (−7.0 / −8.1 / −7.9), confirming the
frames are genuinely different prompts; the operator's causal effect does not. At 8B the pattern
replicates — **100/100 flips over the tested combinations (80/80 cross-frame, mean contrast +28.7)**,
with the declarative-built direction again frame-invariant (+26.0 / +26.0 / +26.0). The operator
direction is a property of the **relation**, not of the prompt that elicits it.

### 4.5 Cross-architecture replication (Gemma-2-9B)

The entire pipeline transfers unchanged to **Gemma-2-9B** — a different pretraining corpus, tokenizer
(SentencePiece; answer single-token coverage 0.95), and architecture family (soft-capped logits, GQA):

| measure | Qwen3-1.7B | Qwen3-8B | **Gemma-2-9B** |
|---|---:|---:|---:|
| all-pairs swap contrast | +22.6 [+14.0, +32.1] | +26.0 [+17.9, +32.8] | **+30.4 [+23.8, +34.8]** |
| flips (swap > 0) | 20/20 | 20/20 | **20/20** |
| held-out-operand contrast | +20.0 [+10.8, +29.5] | +22.8 [+14.0, +31.0] | **+26.6 [+20.6, +33.2]** |
| operator variance @ query | 86% | 82% | **84.8%** |
| interaction (fusion) | 9% | 13% | **6.8%** |
| pure desinence (clean → +v) | −3.6 → +12.5 | −2.9 → +8.1 | **−3.0 → +8.5** |

Every qualitative signature replicates — all-pairs flips with matched-norm nulls ≈ 0, held-out-operand
transfer, case-dominant factorization at the query token with single-digit fusion, the along-sequence
shift (stem 37.6% → 8.4% from the entity token to the query token), and the exponent-free desinence.
Effect sizes are, if anything, largest in Gemma. The operator–operand factorization is not an
idiosyncrasy of one model family.

### 4.6 The factorization is domain-specific (arithmetic and logic do not)

Extending the identical pipeline to arithmetic (+, ×, −) and comparison-logic operators, on 1.7B:

| domain | operator variance | interaction (fusion) | held-out generalization | swap vs random |
|---|---:|---:|---|---|
| **relational** | 86% | 9% | **20/20** flip | +21 vs ~0 |
| **arithmetic** | 55% | 23% | 2/6 (≤ random) | +0.4 vs +0.3 |
| **logic (compare)** | 33% | 34% | 2/6 (≤ random) | +1.2 vs +0.2 |

The gradient replicates and sharpens at **8B**: relational held-out generalization 20/20 (interaction
13%); arithmetic interaction 25%, held-out 1/6; logic interaction 45%, held-out **0/6** (swap below
random). Monotone at both scales: relational ≫ arithmetic > logic.

Relational operators factorize cleanly and generalize; arithmetic and logical operators are **entangled
with their operands** (2–4× the interaction) and do **not** form a transferable direction — consistent
with arithmetic being a bag of heuristics / Fourier computation rather than a linear operator. The
**held-out generalization test is the discriminator** between a real operator and memorized interpolation.

### 4.7 Reconciling with a contrary report: add-N vs. two-operand arithmetic

Christ et al. (2025) report the *opposite* sign for arithmetic: an operator built from an **add-N**
relation (a fixed addend `N` applied to a single number) **does** generalize to held-out relations. The
two results do not conflict — they cut arithmetic along different axes. In add-N the operator **is the
addend `N`** over one numeric operand, so "generalizing across `N`" is interpolation along the number
line (the operand *value* is itself linear; Gurnee & Tegmark 2024), not transfer of a *function*. Our
`+ × −` cut varies the **function** over two operands. Running both cuts on Qwen3
(`op_core.py --domain arith_addN` vs `arithmetic`; `scripts/operator_collinearity.py`) places add-N
exactly between relations and the genuine-function cut:

| cut | held-out generalization (1.7B / 8B) | operator-set collinearity (top-1 var) |
|---|---|---:|
| relations (5 operations) | 20/20 · 20/20 | 0.39 (spread paradigm) |
| **add-N** (operator = addend) | 5/12 · 6/12 (+0.10 / +0.05) | **0.76 (most 1-D / number-line)** |
| `+ × −` (operator = function) | 2/6 · 1/6 (−0.43 / −0.55) | 0.64 |

add-N generalizes **better** than `+ × −` (positive vs. clearly negative) and its operator directions are
the **most collinear** of the three families — 76% of the operator-set variance on a single line —
consistent with add-N being a linear numeric family rather than a set of distinct operations. This
reproduces Christ et al.'s positive (their cut varies a linear parameter) alongside our negative (ours
varies the function) with no contradiction. The caveat is quantitative: Qwen3 BPE restricts us to
single-digit results, so the add-N grid is small (5 operands) and its generalization is weak in absolute
terms — the **ordering** (relations ≫ add-N > `+ × −`), not the magnitude, is the claim.

### 4.8 This is causal structure, not a J-space readout advantage

On four readout comparisons (bridge-entity pass@k, a surface-form logit difference, a number probe, and a
concept-plane trajectory), the Jacobian-lens readout of Gurnee, Sofroniew, Lindsey et al. 2026 — whose
global-workspace result motivated this project, and whose "workspace" band we intervene on throughout —
shows **no readout advantage** over the logit-lens under matched
controls; a spectrum-matched random projection reads number as well as the fitted J-lens, and the
concept-plane "channel" does not survive a random-plane baseline (see [findings §Part 1](findings.md)). The
operator/operand structure lives in the **causal** organization of the residual, and is **not** more
legible in the J-space than in the raw stream (its 1.7B readout-geometry difference does not replicate at
8B). We therefore make no claim of a special readable subspace; the contribution is the operator/operand
*causal* factorization.

## 5. Limitations

- **Scale and family.** 1.7–9B models from two families (Qwen3, Gemma 2) — both decoder-only
  transformers; no ≥30B model. Relational linearity holds for a subset of relations even in-domain
  (cf. LRE ~48%).
- **Relation coverage.** Five hand-chosen country relations, English-only. Prompt variation covers three
  paraphrase *frames* (§4.4) but all frames share the `{op} of {a}` phrasing unit; phrase-level
  paraphrases, more relations, and other entity types remain open.
- **Arithmetic coverage.** Qwen3 BPE splits multi-digit numbers, so we restrict to single-digit results;
  the arithmetic null is on that regime and at 1.7B/8B, where arithmetic competence is itself limited.
- **Reading-position result** restates known subject-enrichment→attribute-extraction dynamics.
- **No J-space claim.** The readout-geometry difference is 1.7B-specific; the paper stands on the causal
  factorization, not on a J-lens readout advantage.

## 6. Conclusion

An LLM represents a relational operation as a low-dimensional, causally-manipulable operator direction
that factorizes from the operand, generalizes to unseen operands, and separates the operation from the
word that realizes it — a structure well described as **declension**. This factorization is not universal:
it is clean for relational retrieval and breaks down for arithmetic and logic, whose operators are
entangled with their operands. The structure is causal; it is not a more-readable subspace.

## References

Christ et al. 2025, *The Structure of Relation Decoding Linear Operators in Large Language Models*, NeurIPS (Spotlight), arXiv:2510.26543 ·
Chughtai et al. 2024, *Summing Up the Facts*, arXiv:2402.07321 ·
Geva et al. 2023, *Dissecting Recall of Factual Associations*, EMNLP ·
Gurnee, Sofroniew, Lindsey et al. 2026, *Verbalizable Representations Form a Global Workspace in Language Models*, Transformer Circuits Thread ·
Gurnee & Tegmark 2024, *Language Models Represent Space and Time*, ICLR ·
Hendel et al. 2023, *In-Context Learning Creates Task Vectors*, EMNLP Findings, arXiv:2310.15916 ·
Hernandez et al. 2024, *Linearity of Relation Decoding in Transformer LMs* (LRE), ICLR, arXiv:2308.09124 ·
Hong et al. 2024, *A Implies B: Circuit Analysis for Propositional Logic*, arXiv:2411.04105 ·
Huang et al. 2024, *RAVEL*, ACL, arXiv:2402.17700 ·
Kantamneni & Tegmark 2025, *Language Models Use Trigonometry to Do Addition*, arXiv:2502.00873 ·
Liu et al. 2024, *In-Context Vectors*, ICML, arXiv:2311.06668 ·
Marks & Tegmark 2023, *The Geometry of Truth*, arXiv:2310.06824 ·
Merullo et al. 2024, *Language Models Implement Simple Word2Vec-style Vector Arithmetic*, NAACL, arXiv:2305.16130 ·
Nanda et al. 2023, *Progress Measures for Grokking*, ICLR, arXiv:2301.05217 ·
Nikankin et al. 2025, *Arithmetic Without Algorithms*, ICLR, arXiv:2410.21272 ·
Todd et al. 2024, *Function Vectors in Large Language Models*, ICLR, arXiv:2310.15213 ·
Turner et al. 2023, *Activation Addition (ActAdd)*, arXiv:2308.10248 ·
Zhong et al. 2023, *The Clock and the Pizza*, NeurIPS, arXiv:2306.17844.
