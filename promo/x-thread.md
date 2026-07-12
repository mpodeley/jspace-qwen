# X / Twitter thread — draft

*Ready-to-post. Numbers current as of 2026-07-12 (reviewer round 2: composition ladder,
dose × position, competitive nulls, animals domain, generation audit — replicated at 8B).
Personalize the voice, fill handles (verify before tagging), attach the GIFs where marked.
Live site: https://mpodeley.github.io/jspace-qwen/ · explorer:
https://mpodeley.github.io/jspace-qwen/explorer/*

---

**1/**  An LLM stores a *relation* the way Latin stores a noun's role: it **declines** the
concept by case. The country is the stem; "currency-of / capital-of / language-of" is the
case ending — an addable direction that factorizes off the operand.

New work on Qwen3 + Gemma-2 👇 [attach: promo/declension.gif]

**2/**  Read the residual state H[country, case] as it flows through the model. At the
country token it's organized by **operand** (which country). By the query token it has been
*declined* into an **operator**-organized form (which relation). Two-way ANOVA: ~90%
additive, operand ⊕ operator, ~10% "fusion". [attach: op_geometry.png]

**3/**  Is the operator a *manipulable* direction? Add v(to) − v(from) to a prompt and the
answer-logit margin flips sign — **all 20 ordered pairs, in all three models** (contrast
+22.6 / +26.0 / +30.4, cluster-bootstrap CIs). A matched-norm random direction does ≈ nothing.
[attach: promo/injection.gif]

**4/**  Not the prompt, not the wording. The direction built on one prompt frame transfers to
unseen frames (180/180, 100/100 flips), and — the strong version — a direction built on
"the currency of X" flips prompts phrased "the money used in X": full re-lexicalization,
**40/40**, transfer ≈ within-wording (+22.59 vs +22.62). It's a property of the *relation*.

**5/**  The best experiment: **build the thought from parts**. Take the factorization's three
averaged ingredients — a generic base + an "Italy" part + a "capital-of" part — write the
assembled state into ONE position, and the model *says* "Rome" at its own competence ceiling
(52% vs 53% clean; 8B: 62% vs 68%). Parts that never saw the target cell still hit 36-50%.
Swap in the "France" part → it says *Paris*. The thought is made of parts, and the parts suffice.

**5b/**  And the twist: we'd earlier reported that additive steering can't make the model
*speak* (≤0.5% exact match). That was an **overdose**. The steering vector IS the ANOVA
operator component — applied at the calibrated dose it generates at ceiling too (51% at
α≈0.1). Push harder and the margin keeps climbing while fluency dies (94.6% token loops) —
yet the target still wins a forced choice at 80%. Overdosing destroys fluency, not information.

**6/**  The best part: the operation is separable from the *word it produces*.
`language-of` and `demonym-of` both emit "Italian", yet they're distinct directions. Build a
"pure desinence" from exactly the cases where they share the word — it still installs the
relation causally (−3.6 → +12.5). Grammatical case ≠ its exponent.

**7/**  Does this generalize beyond geography? Yes — and beyond retrieval? No, and both are
results. A curated **animals domain** (class/habitat/diet/covering) replicates everything:
12/12 swaps, held-out 12/12, permuted-label null at zero. But **arithmetic** (+ × −) and
**comparison logic** fail the same tests: no held-out generalization at either Qwen3 scale.
The discriminator isn't fusion share — it's whether the operator direction *transfers*.

**8/**  And an honest negative: I went in expecting the Jacobian-lens "J-space" readout to be
*more legible* than the raw residual. Under matched controls (incl. a spectrum-matched random
projection) it isn't. The structure here is **causal**, not a readout advantage.

**9/**  Everything is reproducible on a single AMD Strix Halo APU (no CUDA, no cloud). There's
an **interactive 3blue1brown-style explorer** — scrub the declension, fire the injection, see
the syncretism — and a plain-language explainer anyone can read:
https://mpodeley.github.io/jspace-qwen/explorer/ ·
https://mpodeley.github.io/jspace-qwen/explained/

**10/**  Paper + code + findings: https://mpodeley.github.io/jspace-qwen/
Builds on function vectors (Todd et al.), LRE (Hernandez et al.), relation embeddings
(Wang et al.), and reconciles with Christ et al. (NeurIPS'25). Feedback very welcome — first
paper in this area, would love your eyes on it. 🙏

*(Consider tagging: the function-vectors / LRE authors, and the mech-interp community —
verify exact handles first.)*
