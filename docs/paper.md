# The workspace declines concepts by role: operator/operand factorization in an open-weights LLM

*Draft, 2026-07-10. Qwen3-1.7B/8B. All numbers reproducible from `scripts/` and `data/`; see
["How to reproduce"](reproduce.md) and the [working findings log](findings.md).*

## Abstract

We study how an open-weights transformer (Qwen3) represents a **relational operation** applied to an
entity — *currency-of*, *capital-of*, *language-of*, *demonym-of*, *continent-of* applied to a country.
We find that the operation is carried by a **manipulable operator direction** in the residual stream
that is largely **separable from the operand**. Adding `v(op_B) − v(op_A)` to a prompt performing `op_A`
converts the model's answer to `op_B` for all 20 ordered operator pairs (matched-norm random controls
null), and the direction **generalizes to operands never used to build it**. A two-way analysis of
variance factorizes the representation as `H[operand, operator] = μ + operand + operator + interaction`,
~90% additive with a ~10% **fusion** interaction, operator and operand occupying largely orthogonal
subspaces; the operation-dominant representation *emerges along the sequence* (operand-dominant at the
entity token → operator-dominant at the query token). Strikingly, the operation is **separable from its
surface realization**: *language-of* and *demonym-of* are distinct operator directions even though both
emit "Italian". We frame this with a morphological metaphor — the model **declines** a concept by its
functional role (case), with **syncretism** (shared surface form) and **fusion** (non-additive
stem+ending) as in a Latin declension. Testing generalization across domains, the factorization is
**specific to relational retrieval**: for **arithmetic** (+, ×, −) and **comparison-logic** operators,
the interaction term is 2–4× larger and the operator **fails to generalize to held-out operands**,
consistent with arithmetic being computed by a "bag of heuristics" rather than a linear operator. Finally,
the structure is **causal, not a readout advantage**: the Jacobian-lens ("J-space") readout is not more
legible than the raw residual on any of our metrics under matched controls.

## 1. Introduction

A line of work shows that a *task* or *relation* can be captured by a single addable vector in an LLM's
residual stream (task vectors, Hendel et al. 2023; function vectors, Todd et al. 2024) and that many
relations are approximately linear operators (LRE, Hernandez et al. 2024; Merullo et al. 2024). What has
not been characterized is the **joint structure of operator and operand**: whether the operation
*factorizes* from its argument, whether that factorization *generalizes*, whether the operation is
separable from the *word it produces*, and whether any of this holds *beyond* relational facts, in
arithmetic and logic. We answer these on Qwen3, and give a morphological reading — relational computation
as **declension** — that predicts the specific asymmetries we observe.

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

**Arithmetic and logic.** Arithmetic in LLMs is reported as a "bag of heuristics" (Nikankin et al. 2025)
and Fourier/helical procedures (Nanda et al. 2023; Zhong et al. 2023; Kantamneni & Tegmark 2025), with the
operand *value* cleanly linear (Gurnee & Tegmark 2024). Truth is a steerable direction (Marks & Tegmark
2023) and logical-operator heads exist (Hong et al. 2024). No prior work reports a **cross-domain
operator/operand factorization**; we provide one and show it **breaks down** for arithmetic and logic —
itself a result.

**Genuinely new here:** (1) operation-vs-realization dissociation (language ≠ demonym as directions
despite identical output); (2) the operator×operand ANOVA with a named fusion term; (3) the
declension/case-paradigm framing; (4) all-pairs swap with matched-norm nulls; (5) operator/operand
subspace orthogonality; (6) the cross-domain generalization test that separates relational (clean) from
arithmetic/logic (entangled).

## 3. Method

**Setup.** Qwen3-1.7B and 8B, run on a single AMD Strix Halo APU (see [setup](setup.md); an int8 path
keeps larger models in range). A relation is rendered into a uniform template `"The {op} of {a} is"`.
Reading position is the query token unless stated; the depth "workspace" band (38–92% of layers) is where
the J-space paper locates the global workspace and where we build operator directions.

**Operator direction.** For operator `k`, `v(k) = mean_operand[ h(operand, k) − mean_op h(operand, ·) ]`
at the workspace layers — the deviation of an operator from the operand's average over operators, averaged
over operands (the function-vector construction; `scripts/op_core.py:op_dirs`).

**Swap and controls.** Efficacy of `k_A → k_B` is the change in `logit(answer_B) − logit(answer_A)` at the
query position when `α·(v(k_B) − v(k_A))` is added over the band, versus a **matched-norm random**
direction. Answers are scored by their distinguishing token (bare digit for arithmetic, since Qwen3
splits " 3" into [space, 3]); a tokenization guard drops operator pairs that collide at that token.

**Factorization.** `H[operand, operator] = μ + operand(o) + operator(k) + interaction`; we report the
variance share of each term and the principal angles between the operand- and operator-subspaces
(`scripts/op_core.py:factorize`).

**Generalization.** Build `v(op)` from half the operands; test the swap on the held-out half. This
distinguishes a genuine operator from interpolation among the examples used to build it.

**J-space controls.** We repeat readout-geometry in the J-lens readout `unembed(J·h)` vs the logit-lens
readout `unembed(h)`, and re-run efficacy with **spectrum-matched random-projection** and
**permuted-vocabulary** null lenses (`scripts/control_lens.py`).

## 4. Results

### 4.1 Relational operators are manipulable directions

All 20 ordered operator swaps flip the answer (mean swap **+21** logit units on 1.7B, **+25** on 8B) while
the matched-norm random control is ~0. Representative (1.7B):

| swap | clean (from wins) | +operator | +random |
|---|---:|---:|---:|
| currency → capital | −7.9 | **+31.5** | −1.2 |
| capital → currency | −12.0 | **+33.0** | −2.5 |
| capital → language | −8.5 | **+32.5** | +3.0 |
| continent → language | −1.9 | **+23.0** | +1.4 |

### 4.2 The representation factorizes into operator ⊕ operand

Two-way ANOVA at a mid-workspace layer (1.7B / 8B):

| read position | operand | operator | interaction (fusion) |
|---|---:|---:|---:|
| query token | 5% / 6% | **86% / 82%** | 9% / 13% |
| entity token | **59% / 55%** | 31% / 34% | 9% / 11% |

~90% additive; operand/operator subspaces largely orthogonal (principal angles 41–82°). The operation-
dominant representation **emerges along the sequence**: the entity enters operand-dominant and is
*declined* into an operator-marked form by the query position (the reading-position control rules out
template echo; the causal swaps are the stronger control).

### 4.3 Operation ≠ realization (syncretism)

*language-of* and *demonym-of* emit the **same word** (Italian) yet are **distinct operator directions**;
the swap between them is weak precisely because they share an exponent. A **pure desinence** built from
the shared-output pairs (`mean[h(language) − h(demonym)]`, exponent cancelled) still installs the relation
causally (1.7B: −3.7 → +12.5). The syncretism appears at the *exponent* level, not the *case* level — as
in Latin declension.

### 4.4 Operators generalize to held-out operands

`v(op)` built on 6 operands and applied to the 6 held-out operands still flips **20/20** swaps (mean
**+19** on 1.7B) — a genuine, transferable operator, not interpolation.

### 4.5 The factorization is domain-specific (arithmetic and logic do not)

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

### 4.6 This is causal structure, not a J-space readout advantage

On four readout comparisons (bridge-entity pass@k, a surface-form logit difference, a number probe, and a
concept-plane trajectory), the J-lens readout shows **no advantage** over the logit-lens under matched
controls; a spectrum-matched random projection reads number as well as the fitted J-lens, and the
concept-plane "channel" does not survive a random-plane baseline (see [findings §Part 1](findings.md)). The
operator/operand structure lives in the **causal** organization of the residual, and is **not** more
legible in the J-space than in the raw stream (its 1.7B readout-geometry difference does not replicate at
8B). We therefore make no claim of a special readable subspace; the contribution is the operator/operand
*causal* factorization.

## 5. Limitations

- **Scale.** 1.7B and 8B; no 32B lens yet. Relational linearity holds for a subset of relations even
  in-domain (cf. LRE ~48%); our five relations are hand-chosen and English-only.
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

Chughtai et al. 2024, *Summing Up the Facts*, arXiv:2402.07321 ·
Geva et al. 2023, *Dissecting Recall of Factual Associations*, EMNLP ·
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
