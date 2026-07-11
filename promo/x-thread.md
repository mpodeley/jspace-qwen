# X / Twitter thread — draft

*Ready-to-post. Numbers current as of 2026-07-11 (includes Gemma-2-9B replication, the
re-lexicalization control, and the minimal-intervention results). Personalize the voice,
fill handles (verify before tagging), attach the GIFs where marked. Live site:
https://mpodeley.github.io/jspace-qwen/ · explorer:
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

**5/**  How much machinery does this need? **One layer.** The same directions at a single
mid-workspace layer flip 20/20 margins at 4.4× lower off-task KL. And an honest split: the
additive direction rotates *preference* (≤0.5% greedy exact-match), while patching the real
activation at one position reroutes the *generated answer* — 51% vs the model's own 53% ceiling.

**6/**  The best part: the operation is separable from the *word it produces*.
`language-of` and `demonym-of` both emit "Italian", yet they're distinct directions. Build a
"pure desinence" from exactly the cases where they share the word — it still installs the
relation causally (−3.6 → +12.5). Grammatical case ≠ its exponent.

**7/**  Does this generalize beyond facts? No — and that's a result. The identical pipeline on
**arithmetic** (+ × −) and **comparison logic** fails: 2–4× the fusion, no held-out
generalization. Consistent with arithmetic = "bag of heuristics", not a linear operator.
Monotone at both Qwen3 scales: relations ≫ arithmetic > logic.

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
