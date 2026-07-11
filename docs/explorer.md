# Interactive explorer

An interactive, 3Blue1Brown-style view of the operator/operand ("declension") result on
Qwen3-1.7B — read live from the model, no re-run needed. Three scenes: the **declension**
morph (operand-organized → operator-organized along the sequence), the **injection** of an
operator direction (watch the answer flip), and the **syncretism** of `language`/`demonym`
with the pure desinence.

**How to drive it.** *Scene 1 (Declension):* drag the slider from "country token" to "query token"
and watch the same 60 dots reorganize — mixed by country on the left, five clean question-clusters on
the right; press ▶ to animate it. *Scene 2 (Injection):* pick a from→to pair, hit **inject**, and watch
the state slide into the target cluster while the answer margin flips; then tick **random control** and
see an equally large random arrow do nothing. *Scene 3 (Syncretism):* two questions that share their
answer word sit apart as directions; fire the **pure desinence** and watch a marker with the word
cancelled out still install the relation.

<iframe src="../interactive/declension.html" title="Declension explorer"
        style="width:100%;height:1180px;border:1px solid #ddd;border-radius:12px"
        loading="lazy"></iframe>

> Full-page: [open the explorer directly](interactive/declension.html). Deep-links:
> [declension](interactive/declension.html#0) · [injection](interactive/declension.html#1) ·
> [syncretism](interactive/declension.html#2).
