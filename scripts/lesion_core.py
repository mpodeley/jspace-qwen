#!/usr/bin/env python
"""Lesion machinery: read and silence individual attention heads and MLP neurons.

Everything else in this repo intervenes on the residual stream at BLOCK
granularity (`model.layers[l]`), which is the right unit for steering but the
wrong one for a lesion study. Neuropsychology's method -- localize a functional
network with a contrast, remove the tissue, measure whether the deficit is
SELECTIVE -- needs units you can actually remove: heads and neurons.

Two hook points, both inside the block, both read/write:

    layers[l].self_attn.o_proj   pre-hook input = [b, s, H*head_dim]
                                 -> the per-head outputs, concatenated
    layers[l].mlp.down_proj      pre-hook input = [b, s, d_mlp]
                                 -> act_fn(gate(x)) * up(x): *the* MLP neurons

Conventions that matter:

* A LESION removes the unit at EVERY token position of EVERY prompt (the tissue
  is gone), unlike the paper's position-restricted interventions, which are more
  like a momentary stimulation. This is what makes "general competence spared"
  a meaningful control.
* MEAN ablation is primary: the unit is pinned to its average activation over a
  reference corpus (WikiText), which keeps the rest of the network on
  distribution. ZERO ablation is off-distribution and systematically overstates
  importance; it is kept as a sensitivity check. RESAMPLE ablation (substitute
  another prompt's activation) is the third option.
* Qwen3 GOTCHA: head_dim != d_model // n_heads (Qwen3-32B: 5120/64 != 128). Read
  it from the config or from o_proj.in_features -- never divide.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class UnitDims:
    n_layers: int
    n_heads: int
    head_dim: int
    d_mlp: int

    @property
    def n_head_units(self) -> int:
        return self.n_layers * self.n_heads

    @property
    def n_neuron_units(self) -> int:
        return self.n_layers * self.d_mlp


def unit_dims(model) -> UnitDims:
    """Head/neuron geometry, read from the config (never derived by division)."""
    cfg = model._hf_model.config.get_text_config()
    attn = model.layers[0].self_attn
    head_dim = int(getattr(attn, "head_dim", 0) or cfg.head_dim)
    n_heads = int(cfg.num_attention_heads)
    assert attn.o_proj.in_features == n_heads * head_dim, (
        f"o_proj.in_features={attn.o_proj.in_features} != "
        f"n_heads*head_dim={n_heads}*{head_dim}")
    return UnitDims(n_layers=model.n_layers, n_heads=n_heads, head_dim=head_dim,
                    d_mlp=int(model.layers[0].mlp.down_proj.in_features))


# --- reading ------------------------------------------------------------------

class UnitRecorder:
    """Context manager recording per-head and per-neuron activations.

    After exit, `.heads[l]` is [seq, n_heads] (the L2 norm of each head's output
    slice -- a head's "activity", invariant to how o_proj mixes it) and
    `.neurons[l]` is [seq, d_mlp] (the raw neuron activations). Both float32 on
    CPU, one prompt at a time (HFLensModel.forward takes [1, seq])."""

    def __init__(self, model, layers=None, dims: UnitDims | None = None):
        self.m = model
        self.dims = dims or unit_dims(model)
        self.layers = list(range(model.n_layers)) if layers is None else list(layers)
        self.heads: dict[int, torch.Tensor] = {}
        self.neurons: dict[int, torch.Tensor] = {}
        self._handles = []

    def __enter__(self):
        d = self.dims
        for l in self.layers:
            def mk_head(l):
                def pre(_mod, args):
                    z = args[0]                                  # [b, s, H*hd]
                    z = z.view(*z.shape[:-1], d.n_heads, d.head_dim)
                    self.heads[l] = z[0].float().norm(dim=-1).cpu()   # [s, H]
                    return None                                  # read-only
                return pre

            def mk_neuron(l):
                def pre(_mod, args):
                    self.neurons[l] = args[0][0].float().cpu()   # [s, d_mlp]
                    return None
                return pre

            self._handles.append(
                self.m.layers[l].self_attn.o_proj.register_forward_pre_hook(mk_head(l)))
            self._handles.append(
                self.m.layers[l].mlp.down_proj.register_forward_pre_hook(mk_neuron(l)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


@torch.no_grad()
def record_units(model, prompt: str, pos: int | None = -1, dims: UnitDims | None = None):
    """Per-unit activations for one prompt. `pos=-1` reads the query token;
    `pos=None` averages over all token positions (the localizer convention when
    the function is not tied to one position, e.g. induction or text-vs-symbolic).

    Returns (heads, neurons): [n_layers, n_heads] and [n_layers, d_mlp]."""
    dims = dims or unit_dims(model)
    ids = model.encode(prompt, max_length=512)
    with UnitRecorder(model, dims=dims) as rec:
        model.forward(ids)
    def stack(d, width):
        out = torch.zeros(dims.n_layers, width)
        for l, a in d.items():
            out[l] = a[pos] if pos is not None else a.mean(0)
        return out
    return stack(rec.heads, dims.n_heads), stack(rec.neurons, dims.d_mlp)


@torch.no_grad()
def reference_means(model, prompts, dims: UnitDims | None = None):
    """Per-unit mean activation over a reference corpus, averaged over all token
    positions -- the substrate for mean-ablation. Heads: mean of the per-head
    output VECTOR (not its norm), since a lesion must write a vector back; so we
    return [n_layers, n_heads, head_dim] for heads and [n_layers, d_mlp] for
    neurons."""
    dims = dims or unit_dims(model)
    H = torch.zeros(dims.n_layers, dims.n_heads, dims.head_dim)
    N = torch.zeros(dims.n_layers, dims.d_mlp)
    n_tok = 0
    for p in prompts:
        ids = model.encode(p, max_length=512)
        acc_h, acc_n = {}, {}
        handles = []
        for l in range(model.n_layers):
            def mk_h(l):
                def pre(_m, args):
                    z = args[0]
                    z = z.view(*z.shape[:-1], dims.n_heads, dims.head_dim)
                    acc_h[l] = z[0].float().sum(0).cpu()          # [H, hd]
                    return None
                return pre

            def mk_n(l):
                def pre(_m, args):
                    acc_n[l] = args[0][0].float().sum(0).cpu()    # [d_mlp]
                    return None
                return pre
            handles.append(model.layers[l].self_attn.o_proj
                           .register_forward_pre_hook(mk_h(l)))
            handles.append(model.layers[l].mlp.down_proj
                           .register_forward_pre_hook(mk_n(l)))
        model.forward(ids)
        for h in handles:
            h.remove()
        for l in range(model.n_layers):
            H[l] += acc_h[l]
            N[l] += acc_n[l]
        n_tok += ids.shape[1]
    return H / n_tok, N / n_tok


# --- lesioning ----------------------------------------------------------------

class Lesion:
    """Context manager silencing a set of units at EVERY token position.

    `heads`: iterable of (layer, head); `neurons`: iterable of (layer, index).
    `mode`:
      mean     -- pin the unit to its reference-corpus mean (primary; keeps the
                  network on-distribution)
      zero     -- write zeros (off-distribution; overstates importance)
      resample -- write the unit's activation from a random other position of the
                  same forward pass (a cheap within-prompt resample ablation)

    `means` is the (H, N) pair from `reference_means`, required for mode="mean".
    An empty lesion is a no-op and must reproduce the clean forward bit-exactly
    (asserted in the battery's identity check)."""

    def __init__(self, model, heads=(), neurons=(), mode="mean", means=None,
                 dims: UnitDims | None = None, seed: int = 0):
        self.m = model
        self.dims = dims or unit_dims(model)
        self.mode = mode
        self.means = means
        self.seed = seed
        self.by_layer_heads: dict[int, list[int]] = {}
        self.by_layer_neurons: dict[int, list[int]] = {}
        for l, h in heads:
            self.by_layer_heads.setdefault(int(l), []).append(int(h))
        for l, i in neurons:
            self.by_layer_neurons.setdefault(int(l), []).append(int(i))
        if mode == "mean" and means is None and (self.by_layer_heads
                                                 or self.by_layer_neurons):
            raise ValueError("mode='mean' needs means=reference_means(...)")
        self._handles = []

    def __enter__(self):
        d = self.dims
        g = torch.Generator().manual_seed(self.seed)
        for l, hs in self.by_layer_heads.items():
            idx = torch.tensor(sorted(set(hs)))
            def mk(l, idx):
                def pre(_mod, args):
                    z = args[0].clone()                          # [b, s, H*hd]
                    z = z.view(*z.shape[:-1], d.n_heads, d.head_dim)
                    if self.mode == "zero":
                        z[..., idx, :] = 0.0
                    elif self.mode == "mean":
                        mv = self.means[0][l][idx].to(z.device, z.dtype)  # [k, hd]
                        z[..., idx, :] = mv
                    elif self.mode == "resample":
                        s = z.shape[-3]
                        perm = torch.randperm(s, generator=g).to(z.device)
                        z[..., idx, :] = z[..., perm, :, :][..., idx, :]
                    else:
                        raise ValueError(self.mode)
                    return (z.reshape(*z.shape[:-2], d.n_heads * d.head_dim),
                            *args[1:])
                return pre
            self._handles.append(self.m.layers[l].self_attn.o_proj
                                 .register_forward_pre_hook(mk(l, idx)))
        for l, ns in self.by_layer_neurons.items():
            idx = torch.tensor(sorted(set(ns)))
            def mkn(l, idx):
                def pre(_mod, args):
                    a = args[0].clone()                          # [b, s, d_mlp]
                    if self.mode == "zero":
                        a[..., idx] = 0.0
                    elif self.mode == "mean":
                        a[..., idx] = self.means[1][l][idx].to(a.device, a.dtype)
                    elif self.mode == "resample":
                        s = a.shape[-2]
                        perm = torch.randperm(s, generator=g).to(a.device)
                        a[..., idx] = a[..., perm, :][..., idx]
                    else:
                        raise ValueError(self.mode)
                    return (a, *args[1:])
                return pre
            self._handles.append(self.m.layers[l].mlp.down_proj
                                 .register_forward_pre_hook(mkn(l, idx)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False

    @property
    def size(self) -> dict:
        return {"heads": sum(len(v) for v in self.by_layer_heads.values()),
                "neurons": sum(len(v) for v in self.by_layer_neurons.values())}


# --- behavioral readouts under lesion -----------------------------------------

@torch.no_grad()
def perplexity(model, prompts, max_length=512) -> float:
    """Mean next-token NLL over a corpus (the general-competence control: a
    lesion that raises this is damage, not selective localization)."""
    import torch.nn.functional as F
    from jlens import ActivationRecorder
    tot, n = 0.0, 0
    final = model.n_layers - 1
    for p in prompts:
        ids = model.encode(p, max_length=max_length)
        with ActivationRecorder(model.layers, at=[final]) as rec:
            model.forward(ids)
            h = rec.activations[final][0]
        logits = model.unembed(h.float())
        lp = F.log_softmax(logits[:-1], -1)
        tgt = ids[0, 1:]
        tot += float(-lp.gather(-1, tgt[:, None]).sum())
        n += int(tgt.numel())
    return float(torch.tensor(tot / max(n, 1)).exp())
