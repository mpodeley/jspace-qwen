"""Shared helpers for the jspace-qwen scripts: model loading, layer selection,
and a pretraining-like fitting corpus. Kept tiny and dependency-light so every
phase (fit / apply / metrics / ablation) imports the same conventions."""

from __future__ import annotations

import torch
import transformers

import jlens

# The three-model scale sweep (Qwen3 dense, Apache-2.0, shared architecture).
MODELS = {
    "1.7b": "Qwen/Qwen3-1.7B",
    "8b": "Qwen/Qwen3-8B",
    "32b": "Qwen/Qwen3-32B",
}


def load_model(
    name_or_key: str, *, dtype: torch.dtype = torch.bfloat16, compile: bool = False
) -> jlens.HFLensModel:
    """Load a Qwen3 (or any HF decoder) as an HFLensModel on the GPU in bf16."""
    name = MODELS.get(name_or_key, name_or_key)
    hf = transformers.AutoModelForCausalLM.from_pretrained(name, dtype=dtype).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(name)
    return jlens.from_hf(hf, tok, compile=compile)


def resolve_tag(key: str, *, int8: bool = False) -> str:
    """Filename tag for a model run: '8b', '8b-int8', or an HF basename."""
    base = key if key in MODELS else key.split("/")[-1]
    return f"{base}-int8" if int8 else base


def evenly_spaced_layers(n_layers: int, k: int = 25) -> list[int]:
    """~k evenly spaced source layers in ``[0, n_layers-2]`` (all strictly below
    the final/target layer). Mirrors the paper's "25 evenly spaced layers"."""
    top = n_layers - 2  # last valid source layer (target is n_layers-1)
    k = min(k, top + 1)
    xs = torch.linspace(0, top, k).round().long().tolist()
    return sorted(set(int(x) for x in xs))


def depth_percent(layer: int, n_layers: int) -> float:
    """Reindex a layer to the paper's 0-100 depth scale."""
    return 100.0 * layer / (n_layers - 1)


# Depth bands, in depth_percent units. Single source of truth: causal_swap,
# metrics, plots and syntax_swap all import this. Callers iterate BANDS, so do
# not add aliases to this dict -- an extra key becomes an extra output row.
#
# WARNING: these boundaries were chosen for the *semantic* workspace story.
# The morphosyntax work must not assume the motor boundary -- derive it from
# where form-accuracy rises relative to where lemma-accuracy saturates, then
# pin it here once. See docs/results-morphosyntax.md.
BANDS = {"early": (0, 33), "workspace": (38, 92), "late": (92, 100)}

# The band in which the concept pointer is dereferenced into a surface form.
# Provisionally the late band; syntax_probe.py is what licenses this choice.
MOTOR_BAND = "late"
WORKSPACE_BAND = "workspace"


def band_layers(source_layers, n_layers: int) -> dict[str, list[int]]:
    """Map each band to the fitted source layers whose depth falls inside it."""
    return {
        band: [l for l in source_layers if lo <= depth_percent(l, n_layers) <= hi]
        for band, (lo, hi) in BANDS.items()
    }


# --- BPE discipline -------------------------------------------------------
#
# Qwen3 uses byte-level BPE: " euro", "euro" and " Euro" are three distinct
# tokens. Any morphology contrast must therefore hold the leading space and the
# sentence position fixed, or a number effect is confounded with a spacing one.
#
# Measured on Qwen/Qwen3-1.7B (vocab 151643), NOT assumed -- the GPT-2 intuition
# that a first-token metric is blind to regular plurals is FALSE here. Qwen3 has
# whole-word plural tokens: " cats" is a single id (19423), distinct from " cat"
# (8251). Single-token coverage of regular plurals is frequency-dependent:
#
#     common nouns  ~96%   (cat/cats, book/books, tree/trees)
#     mid-frequency ~93%   (rocket/rockets, basket/baskets)
#     rare nouns     ~8%   (quokka -> [' qu','ok','ka'])
#
# So regular plurals ARE usable, and the contrast set need not be restricted to
# irregulars. The real bias is the other way round: filtering on single-token
# forms silently selects *frequent* lemmas. Report the coverage fraction and the
# frequency profile of what survives -- do not present coverage as neutral.
#
# Use single_leading_space_token() to filter, first_token_distinguishable() to
# assert a contrast is actually measurable at token 0.


def token_ids(tok, word: str) -> list[int]:
    """Token ids of ``word`` in mid-sentence position (with its leading space)."""
    return tok.encode(" " + word.strip(), add_special_tokens=False)


def first_token(tok, word: str) -> int:
    """First token id of ``word`` with a leading space."""
    return token_ids(tok, word)[0]


def single_leading_space_token(tok, word: str) -> int | None:
    """The token id if ``word`` is exactly one leading-space token, else None."""
    ids = token_ids(tok, word)
    return ids[0] if len(ids) == 1 else None


def first_token_distinguishable(tok, a: str, b: str) -> bool:
    """True iff ``a`` and ``b`` differ at token 0, i.e. a first-token metric can
    tell them apart. True for most common regular plurals on Qwen3 (whole-word
    plural tokens); False once a form fragments (' qu','ok','kas')."""
    return token_ids(tok, a)[0] != token_ids(tok, b)[0]


def get_corpus(n_prompts: int, *, min_chars: int = 600) -> list[str]:
    """WikiText-103 records (pretraining-like). Cached to data/ on first use so
    later runs (and the whole scale sweep) reuse the exact same prompt set and
    work offline."""
    import json
    from pathlib import Path

    cache = Path(__file__).resolve().parent.parent / "data" / "corpus.json"
    if cache.exists():
        prompts = json.loads(cache.read_text())
        if len(prompts) >= n_prompts:
            return prompts[:n_prompts]

    from jlens.examples import load_wikitext_prompts

    # Fetch a generous pool once (>= what any phase asks for) and cache it.
    pool = load_wikitext_prompts(max(n_prompts, 128), min_chars=min_chars)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(pool))
    return pool[:n_prompts]
