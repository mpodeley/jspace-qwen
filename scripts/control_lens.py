#!/usr/bin/env python
"""Control lenses: null models that answer "does the J-lens only look structured
because it is fitted?"

Neither the tuned lens (Belrose et al., 2023) nor the J-space paper (Gurnee,
Sofroniew, Lindsey et al., 2026) runs this control. The probing literature has
demanded it since Hewitt & Liang (EMNLP 2019, "control tasks"): report
selectivity = real - control, not raw accuracy.

Note the J-lens fit is *unsupervised* -- J_l = E[d h_final / d h_l], with no
targets -- so a literal shuffled-target refit does not map onto it. Two faithful
analogs, both CPU-only and seconds per lens:

  randproj   Keep the singular values of J_l, randomize the singular vectors:
             J_l = U S V^T  ->  R_l = U_rand S V_rand^T, with U_rand/V_rand drawn
             as random orthonormal bases. R_l has the *same spectrum* as J_l, so
             eff_dim (a participation ratio of singular values, metrics.py:103)
             is preserved exactly. It carries no fitted directional structure.

             read_var_frac is deliberately NOT preserved: it measures how much
             residual variance lies in J_l's top-25 *right singular vectors*, so
             it depends on where those directions point, not on the spectrum. A
             large gap read_var_frac(J) >> read_var_frac(R) is the evidence that
             the fitted read directions are aligned with the data. Preserving it
             would defeat the control.

  permvocab  Keep the real J_l; decode against a fixed random permutation of the
             unembedding rows. Structure survives, meaning does not.

             CAVEAT, and it is sharp: permuting W_U rows permutes the logit
             vector, and a linear probe on the *full* logit vector is invariant
             to that (a permutation is an invertible linear map, so the probe
             just relearns permuted weights -- identical accuracy). permvocab is
             therefore only a valid control for metrics that index the logit
             vector by *linguistic identity*: a restricted candidate-token axis
             set, rank-of-a-named-token, or a logit difference between two named
             tokens. That is exactly what syntax_probe.py and lens_eval.py do.
             Do not use it to control a full-vocabulary probe.

Usage:
    python scripts/control_lens.py 1.7b --kind randproj
    python scripts/control_lens.py 8b --int8 --kind randproj --seed 0

Writes out/lenses/<tag>-randproj.pt, consumable by every script's --lens flag.
The vocab permutation is written as out/lenses/<tag>-permvocab.json (a plain id
list) because it is a readout-side object, not a lens.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from _common import MODELS, resolve_tag
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent
LENSES = ROOT / "out" / "lenses"


def random_orthonormal(d: int, generator: torch.Generator) -> torch.Tensor:
    """A Haar-ish random orthonormal d x d matrix via QR of a Gaussian."""
    g = torch.randn(d, d, generator=generator, dtype=torch.float32)
    q, r = torch.linalg.qr(g)
    # Fix the sign convention so Q is not biased toward the identity's orthant.
    return q * torch.sign(torch.diagonal(r)).unsqueeze(0)


def randproj_lens(
    lens: JacobianLens, seed: int
) -> tuple[JacobianLens, dict[int, torch.Tensor]]:
    """Spectrum-matched, direction-randomized null lens.

    Returns the lens and the per-layer singular values used to build it, so the
    caller need not recompute any SVD to report eff_dim (each R_l has singular
    values exactly ``spectra[l]`` by construction: U, V are orthonormal)."""
    generator = torch.Generator().manual_seed(seed)
    out: dict[int, torch.Tensor] = {}
    spectra: dict[int, torch.Tensor] = {}
    for layer in lens.source_layers:
        J = lens.jacobians[layer].float()
        s = torch.linalg.svdvals(J)  # the only SVD per layer
        spectra[layer] = s
        d = J.shape[0]
        u = random_orthonormal(d, generator)
        v = random_orthonormal(d, generator)
        out[layer] = (u * s.unsqueeze(0)) @ v.t()
    control = JacobianLens(
        jacobians=out, n_prompts=lens.n_prompts, d_model=lens.d_model
    )
    return control, spectra


def vocab_permutation(vocab_size: int, seed: int) -> list[int]:
    """A fixed random permutation of vocabulary rows. Score token t as perm[t]."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(vocab_size, generator=generator).tolist()


def _eff_dim(s: torch.Tensor) -> float:
    return float(s.sum() ** 2 / (s.pow(2).sum() + 1e-9))


def _spectrum_report(
    control: JacobianLens, spectra: dict[int, torch.Tensor]
) -> None:
    """Report eff_dim from the known spectra (no SVD), and verify preservation on
    a single layer -- the construction guarantees it algebraically, so one check
    is a guard against a coding error, not a per-layer measurement."""
    layers = control.source_layers
    print("\n  eff_dim (participation ratio of singular values), from construction:")
    for layer in (layers[0], layers[len(layers) // 2], layers[-1]):
        print(f"    layer {layer:>3}: eff_dim = {_eff_dim(spectra[layer]):8.2f}")

    probe = layers[len(layers) // 2]
    s_actual = torch.linalg.svdvals(control.jacobians[probe].float())  # one SVD
    rel = float((s_actual - spectra[probe]).abs().max() / (spectra[probe].max() + 1e-9))
    print(f"    verify layer {probe}: max relative singular-value error = {rel:.2e}")
    if rel > 1e-3:
        raise SystemExit(f"randproj did not preserve the spectrum (err {rel:.2e})")


def main() -> None:
    # This is a pure-CPU utility that may run alongside a GPU fit; keep it from
    # grabbing every core and starving the fit's dataloader.
    torch.set_num_threads(min(4, torch.get_num_threads()))

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("model", help="key (1.7b/8b/32b) or HF id")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--kind", choices=["randproj", "permvocab", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vocab-size", type=int, default=151936, help="Qwen3 lm_head rows")
    ap.add_argument("--lens", default=None)
    args = ap.parse_args()

    tag = resolve_tag(args.model, int8=args.int8)
    lens_path = Path(args.lens) if args.lens else LENSES / f"{tag}.pt"
    if not lens_path.exists():
        raise SystemExit(f"no lens at {lens_path}; fit it first")

    lens = JacobianLens.load(str(lens_path))
    print(f"loaded {lens_path.name}: {lens!r}")

    if args.kind in ("randproj", "both"):
        control, spectra = randproj_lens(lens, args.seed)
        _spectrum_report(control, spectra)
        out = LENSES / f"{tag}-randproj.pt"
        control.save(str(out))
        print(f"  saved {out}")

    if args.kind in ("permvocab", "both"):
        perm = vocab_permutation(args.vocab_size, args.seed)
        out = LENSES / f"{tag}-permvocab.json"
        out.write_text(json.dumps({"seed": args.seed, "perm": perm}))
        n_fixed = sum(1 for i, p in enumerate(perm) if i == p)
        print(f"  saved {out}  (vocab={len(perm)}, fixed points={n_fixed})")


if __name__ == "__main__":
    main()
