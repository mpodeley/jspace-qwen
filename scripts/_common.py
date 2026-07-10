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
