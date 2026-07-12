# Relational operations factorize from their operands in LLMs

<p class="hero-sub">How does a language model represent <i>"the capital of Italy"</i>? We find it
splits the thought in two: <b>the thing</b> (Italy) and <b>the operation applied to it</b>
(capital-of) — and the operation lives as its own <b>direction</b> — an arrow in the model's space of internal
states — that you can grab, move, and transplant.</p>

## The finding, conceptually

Latin marks a noun's *role* with a case ending — `ros-a / ros-am`, same stem, different job.
Mid-network, an LLM does something structurally similar with factual relations. As
*"The currency of Italy is"* flows through the model, the sentence's internal state is well
described by a sum:

```
state ≈ μ + operand(Italy) + operator(currency-of) + small interaction
```

Three things make this more than a curve fit:

1. **The operator is causal.** Add the direction difference `v(capital) − v(currency)` to the
   residual stream and the model stops saying *euro* and says *Rome* — for **every** ordered
   pair of relations we test. A random direction of the same size does nothing.
2. **It transfers.** Built from half the countries, it flips the other half. Built from one
   phrasing, it flips paraphrases it never saw. Built on Qwen, the same recipe works on Gemma.
3. **The operation is not its output word.** *Language-of* and *demonym-of* both end in
   "Italian" — yet they are distinct directions, and a marker built exactly where the two
   share their word still installs the relation. The model separates the grammatical role from
   the surface form, the way declension separates case from ending (*syncretism*).

And the boundary is just as informative: run the identical pipeline on **arithmetic** (+, ×, −)
or **comparison logic**, and the operator is entangled with its operands and refuses to
transfer — consistent with "arithmetic as a bag of heuristics", and evidence that the clean
factorization is a property of *relational retrieval*, not of prompting in general.

<div class="hero-gif" markdown>
[![The declension morph: the same 60 workspace states, organized by operand at the country token, reorganize by operator at the query token](figs/declension.gif)](explorer.md)
</div>

<div class="hero-buttons" markdown>
[:material-play-circle: Interactive result](explorer.md){ .md-button .md-button--primary }
[:material-school: Explained simply](explained.md){ .md-button }
[:material-file-document: Paper (PDF)](assets/paper.pdf){ .md-button }
[:material-flask: Evidence & controls](robustness.md){ .md-button }
[:material-github: Code](https://github.com/mpodeley/jspace-qwen){ .md-button }
</div>

## The numbers

<div class="stats" markdown>
<div class="stat"><span class="n">20/20 × 3</span><span class="d">ordered operator swaps flip the
answer margin in Qwen3-1.7B, Qwen3-8B and Gemma-2-9B; permuted-label nulls ≈ 0</span></div>
<div class="stat"><span class="n">52% ≈ 53%</span><span class="d">a state <em>assembled</em> from
μ + operand + operator makes the model produce the answer at its own ceiling (8B: 62% vs 68%)</span></div>
<div class="stat"><span class="n">80%</span><span class="d">even at a 4× overdose — where fluent
generation collapses — the target still wins a forced choice: dose destroys fluency, not
information</span></div>
</div>

All headline effects carry operator-level cluster-bootstrap 95% CIs and none crosses zero —
see [Evidence & controls](robustness.md) for every claim next to the control that could have
killed it.

!!! note "Methodological note — the honest null that started this"
    This project began as a replication of the J-space / global-workspace readout claim
    ([Gurnee, Sofroniew, Lindsey et al., 2026](https://transformer-circuits.pub/2026/workspace/index.html)).
    Under matched controls (including a spectrum-matched random projection), the **J-lens readout
    did not outperform the logit lens** on any of our metrics. The structure above is *causal*
    organization, not a privileged readable subspace — the negative half is documented with the
    same rigor in the [archive](method.md) and [working log](findings.md).

---

**Reproducibility.** Everything runs on one AMD Strix Halo APU (no CUDA). Seeds, exact
checkpoints (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `google/gemma-2-9b`), per-operand long-form
artifacts, and every figure's generator are in the repo — see [How to reproduce](reproduce.md).
MIT license · Matias Podeley · <mpodeley@gmail.com> ·
[How to cite](https://github.com/mpodeley/jspace-qwen#how-to-cite)
