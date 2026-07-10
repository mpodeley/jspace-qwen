#!/usr/bin/env python
"""Apply a fitted Jacobian lens and print the per-layer top-k readout.

Also prints the vanilla logit-lens baseline (use_jacobian=False) at the same
layers, so you can eyeball where the J-lens surfaces a concept earlier/cleaner.

    python scripts/apply_lens.py 1.7b --demo
    python scripts/apply_lens.py 1.7b --prompt "Fact: the capital of France is"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import MODELS, load_model
from jlens import JacobianLens

ROOT = Path(__file__).resolve().parent.parent

# The paper's canonical two-hop probe: "boot-shaped country" -> Italy -> euro,
# with the intermediate (Italy/euro) surfacing at mid layers before output.
DEMO_PROMPT = (
    "Fact: The capital of Japan is Tokyo.\n"
    "Fact: The currency used in the country shaped like a boot is"
)


def topk_words(tok, logits_row, k: int) -> list[str]:
    return [tok.decode([t]).strip() for t in logits_row.topk(k).indices.tolist()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="key (1.7b/8b/32b) or HF id")
    ap.add_argument("--lens", default=None, help="lens .pt (default out/lenses/<tag>.pt)")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--demo", action="store_true", help="use the boot-country probe")
    ap.add_argument("--position", type=int, default=-2)
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--every", type=int, default=3, help="show every Nth fitted layer")
    args = ap.parse_args()

    key = args.model
    tag = key if key in MODELS else key.split("/")[-1]
    lens_path = Path(args.lens) if args.lens else ROOT / "out" / "lenses" / f"{tag}.pt"
    prompt = DEMO_PROMPT if args.demo or args.prompt is None else args.prompt

    model = load_model(key)
    lens = JacobianLens.from_pretrained(str(lens_path))
    tok = model.tokenizer
    layers = lens.source_layers[:: args.every] + [model.n_layers - 1]

    j_logits, model_logits, ids = lens.apply(
        model, prompt, layers=lens.source_layers, positions=[args.position]
    )
    l_logits, _, _ = lens.apply(
        model, prompt, layers=lens.source_layers, positions=[args.position],
        use_jacobian=False,
    )

    print(f"\nprompt: {prompt!r}")
    print(f"position: {args.position}  (token {tok.decode([ids[0, args.position]])!r})\n")
    print(f"{'layer':>6} | {'J-lens (top-k)':<44} | logit-lens")
    print("-" * 96)
    for layer in layers:
        if layer not in j_logits:
            continue
        j = ", ".join(topk_words(tok, j_logits[layer][0], args.topk))
        l = ", ".join(topk_words(tok, l_logits[layer][0], args.topk))
        mark = "  <- model output" if layer == model.n_layers - 1 else ""
        print(f"{layer:>6} | {j:<44} | {l}{mark}")


if __name__ == "__main__":
    main()
