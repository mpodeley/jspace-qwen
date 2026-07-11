# Relational operations factorize from their operands in LLMs

<p class="hero-sub">Causal interventions reveal <b>transferable operator directions</b> for factual
relations — replicated on <b>Qwen3-1.7B/8B and Gemma-2-9B</b> — while arithmetic and logic remain
entangled with their operands.</p>

<div class="stats" markdown>
<div class="stat"><span class="n">20/20 × 3</span><span class="d">operator swaps flip the answer, in
all three models — matched-norm random controls ≈ 0</span></div>
<div class="stat"><span class="n">82–86%</span><span class="d">of workspace variance at the query
token is the operator; ~7–13% interaction ("fusion")</span></div>
<div class="stat"><span class="n">180/180</span><span class="d">swaps flip under held-out operands
and unseen paraphrase frames — a real operator, not interpolation</span></div>
</div>

<div class="hero-buttons" markdown>
[:material-play-circle: Interactive result](explorer.md){ .md-button .md-button--primary }
[:material-file-document: Paper (PDF)](assets/paper.pdf){ .md-button }
[:material-flask: Evidence & controls](robustness.md){ .md-button }
[:material-github: Code](https://github.com/mpodeley/jspace-qwen){ .md-button }
</div>

<div class="hero-gif" markdown>
[![The declension morph: the same 60 workspace states, organized by operand at the country token, reorganize by operator at the query token](figs/declension.gif)](explorer.md)
</div>

## The idea in 60 seconds

Latin marks a noun's *role* with a case ending: `ros-a / ros-am` — same stem, different job.
We find LLMs do something structurally similar with factual relations. Ask *"The currency of
Italy is"* and, mid-network, the state of the sentence is well described as

```
state ≈ μ + operand(Italy) + operator(currency-of) + small interaction
```

The **operator part is a real, causal object**: add `v(capital) − v(currency)` to the residual
stream and the model answers *Rome* instead of *euro* — for **every** ordered pair of the five
relations we test, in **all three models**. The direction **transfers**: built on half the
countries it flips the other half; built on one phrasing it flips paraphrased prompts unchanged.
And the operation is **not its output word**: *language-of* and *demonym-of* both produce
"Italian", yet are distinct directions — the model separates the grammatical role from the word
that realizes it, the way declension separates case from surface form (*syncretism*).

The factorization is **specific to relational retrieval**: run the identical pipeline on
arithmetic (+, ×, −) or comparison logic and the operator is entangled with its operands
(2–4× the interaction) and fails to generalize — consistent with "arithmetic as a bag of
heuristics". [Explore it interactively](explorer.md), [read the paper](assets/paper.pdf), or
[check every claim against its control](robustness.md).

!!! note "Methodological note — the honest null that started this"
    This project began as a replication of the J-space/global-workspace readout claim. Under
    matched controls (including a spectrum-matched random projection), the **J-lens readout did
    not outperform the logit lens** on any of our metrics. The operator/operand structure above
    is *causal* organization, not a privileged readable subspace — and the negative half is
    documented with the same rigor in the [archive](method.md) and [working log](findings.md).

---

**Reproducibility.** Everything runs on one AMD Strix Halo APU (no CUDA). Seeds, exact
checkpoints (`Qwen/Qwen3-1.7B`, `Qwen/Qwen3-8B`, `google/gemma-2-9b`), per-operand long-form
artifacts, and every figure's generator are in the repo — see [How to reproduce](reproduce.md).
MIT license · Matias Podeley · <mpodeley@gmail.com> ·
[How to cite](https://github.com/mpodeley/jspace-qwen#how-to-cite)
