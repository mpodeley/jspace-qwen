# Findings — J-lens readout vs. causal routing on Qwen3

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

---

## The open interpretive question (deliberately unresolved)

The evidence splits cleanly into a negative half and a positive half:

- **Negative (well controlled):** the J-space is not more *readable* than the raw
  residual on Qwen3. Four comparisons agree; a random projection matches the
  fitted J-lens; the concept channel is a magnitude artifact.
- **Positive (well controlled):** the workspace is causally a *concept-routing
  node* — intervening on it reroutes the downstream concept (and the noun that
  realizes it), robustly, localized to the band, sharpening with scale.

Framings this could support, none yet chosen:

1. **Legibility ≠ causal importance.** The headline is the dissociation itself:
   a subspace can be causally load-bearing without being more linearly readable
   than the stream it sits in. Novel, honest, contrarian to the readout framing.
2. **The positive result alone.** Center on pointer-node routing; the nulls become
   a controls section.
3. **The negative result alone.** "The J-lens readout advantage does not replicate
   on Qwen3 under controls" — modest, defensive, solid.

Open threads that could change the picture before deciding:
- Everything here is 1.7B (causal also 8B). Does the readout null hold at 8B/32B,
  and does the causal reroute keep sharpening?
- `a/an` is a poor morphology probe in this model (determiner not look-ahead). A
  clean downstream-realization test would need a construction where the surface
  form provably depends on the dereferenced concept.
- The causal reroute is shown for concepts; whether *any* morphosyntactic feature
  rides the same node is untested (2.2 says determiner does not).

## Reproducing these numbers

```
python scripts/lens_eval.py 1.7b --set lemma-form      # 1.2
python scripts/syntax_probe.py 1.7b                     # 1.3
python scripts/syntax_probe.py 1.7b --lens out/lenses/1.7b-randproj.pt
python scripts/syntax_swap.py 1.7b                      # 2.2
python scripts/causal_swap.py 1.7b                      # 2.1 (existing)
```

Datasets: `data/morphosyntax/`. Control lenses: `out/lenses/*-randproj.pt`,
`*-permvocab.json` (`scripts/control_lens.py`).
