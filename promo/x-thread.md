# X / Twitter thread — draft

*Ready-to-post. Personalize the voice, fill handles (verify before tagging), and attach the
GIFs where marked. Live site: https://mpodeley.github.io/jspace-qwen/ · explorer:
https://mpodeley.github.io/jspace-qwen/explorer/*

---

**1/**  An LLM stores a *relation* the way Latin stores a noun's role: it **declines** the
concept by case. The country is the stem; "currency-of / capital-of / language-of" is the
case ending — an addable direction that factorizes off the operand.

New work on Qwen3 👇 [attach: promo/declension.gif]

**2/**  Read the workspace state H[country, case] as it flows through the model. At the
country token it's organized by **operand** (which country). By the query token it has been
*declined* into an **operator**-organized form (which relation). A two-way ANOVA:
~90% additive, operand ⊕ operator, ~10% "fusion". [attach: op_geometry.png]

**3/**  Is the operator really a *manipulable* direction? Add v(to) − v(from) to a prompt and
the answer flips — for **all 20** ordered relation pairs. Matched-norm random control ≈ 0.
[attach: promo/injection.gif]

**4/**  The best part: the operation is separable from the *word it produces*.
`language-of` and `demonym-of` both emit "Italian", yet they're distinct directions. Build a
"pure desinence" from exactly the cases where they share the word — it still installs the
relation causally (−3.6 → +12.5). Grammatical case ≠ its exponent.

**5/**  Does this generalize beyond facts? No — and that's a result. The identical pipeline on
**arithmetic** (+ × −) and **comparison logic** fails: 2–4× the fusion, no held-out
generalization. Consistent with arithmetic = "bag of heuristics", not a linear operator.
Monotone at 1.7B and 8B: relations ≫ arithmetic > logic.

**6/**  And an honest negative: I went in expecting the Jacobian-lens "J-space" readout to be
*more legible* than the raw residual. Under matched controls (incl. a spectrum-matched random
projection) it isn't. The structure here is **causal**, not a readout advantage.

**7/**  Everything is reproducible on a single AMD Strix Halo APU (no CUDA, no cloud), and
there's an **interactive 3blue1brown-style explorer** — scrub the declension, fire the
injection, see the syncretism — plus a primer on grammatical case for non-linguists:
https://mpodeley.github.io/jspace-qwen/explorer/

**8/**  Paper + code + findings: https://mpodeley.github.io/jspace-qwen/
Builds on function vectors (Todd et al.), LRE (Hernandez et al.), and reconciles with
Christ et al. (NeurIPS'25). Feedback very welcome — first paper in this area, would love
your eyes on it. 🙏

*(Consider tagging: the function-vectors / LRE authors, and the mech-interp community —
verify exact handles first.)*
