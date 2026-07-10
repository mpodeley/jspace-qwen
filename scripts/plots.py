#!/usr/bin/env python
"""Generate the figures for the docs site from whatever results exist.

Runs safely with partial data (e.g. only 1.7B fitted) and again once the full
sweep lands. Writes PNGs to docs/figs/. Palette + mark conventions follow the
dataviz guidance: categorical color by model in fixed order, thin marks, direct
labels as secondary encoding, recessive grid, ink-colored text.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import pandas as pd

from _common import BANDS

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FIGS = ROOT / "docs" / "figs"
MET = ROOT / "results" / "metrics"
ABL = ROOT / "results" / "ablation"

# categorical color BY MODEL, fixed order (validated: CVD dE 47/41, contrast via labels)
COLOR = {"1.7b": "#2a78d6", "8b": "#1baf7a", "32b-int8": "#eda100", "8b-int8": "#199e70"}
SWEEP = ["1.7b", "8b", "32b-int8"]  # the scale sweep (bf16 where possible, int8 for 32B)
INK, MUTED, GRID, SURF = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})

METRICS = [
    ("jlens_top1", "J-lens next-token top-1 acc"),
    ("read_var_frac", "J-space read variance fraction"),
    ("eff_dim", "effective dim of $J_\\ell$"),
    ("kurtosis", "lens-logit excess kurtosis"),
    ("top1_autocorr", "top-1 autocorrelation"),
]


def load_metrics() -> dict[str, pd.DataFrame]:
    out = {}
    for tag in COLOR:
        p = MET / f"{tag}.parquet"
        if p.exists():
            out[tag] = pd.read_parquet(p)
    return out


def fig_scale_metrics(data):
    tags = [t for t in SWEEP if t in data]
    if not tags:
        return
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2), constrained_layout=True)
    for ax, (col, title) in zip(axes.flat, METRICS):
        for tag in tags:
            df = data[tag].sort_values("depth")
            ax.plot(df["depth"], df[col], color=COLOR[tag], lw=2, marker="o",
                    ms=3.5, label=tag)
            ax.annotate(tag, (df["depth"].iloc[-1], df[col].iloc[-1]),
                        color=COLOR[tag], fontsize=8, xytext=(3, 0),
                        textcoords="offset points", va="center")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("depth (0–100)")
        ax.axvspan(*BANDS["workspace"], color="#000000", alpha=0.04, lw=0)
    axes.flat[-1].axis("off")
    _ws_lo, _ws_hi = BANDS["workspace"]
    axes.flat[-1].text(0.0, 0.8, f"shaded = workspace band ({_ws_lo}–{_ws_hi}%)",
                       color=MUTED, fontsize=9)
    fig.suptitle("J-space metrics vs depth, across model scale", fontsize=13, x=0.01,
                 ha="left")
    fig.savefig(FIGS / "scale_metrics.png", bbox_inches="tight")
    plt.close(fig)


def fig_lenseval(data):
    rows = []
    for tag in COLOR:
        p = MET / f"{tag}_lenseval.json"
        if p.exists():
            rows.append((tag, json.loads(p.read_text())))
    tags = [t for t in SWEEP if t in dict(rows)]
    rows = [(t, d) for t, d in rows if t in tags]
    if not rows:
        return
    ks = [1, 5, 10]
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    import numpy as np
    x = np.arange(len(ks))
    w = 0.8 / max(len(rows), 1)
    for i, (tag, d) in enumerate(rows):
        j = [d[f"jlens_pass@{k}"] for k in ks]
        l = [d[f"logit_pass@{k}"] for k in ks]
        ax.bar(x + i * w, j, w * 0.9, color=COLOR[tag], label=f"{tag} J-lens")
        ax.bar(x + i * w, l, w * 0.9, facecolor="none", edgecolor=COLOR[tag],
               hatch="////", lw=0)
    ax.set_xticks(x + w * (len(rows) - 1) / 2, [f"pass@{k}" for k in ks])
    ax.set_ylabel("fraction of intermediates recovered")
    ax.set_title("Lens quality: J-lens (solid) vs logit-lens (hatched)", fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(rows))
    fig.savefig(FIGS / "lenseval.png", bbox_inches="tight")
    plt.close(fig)


def fig_causal():
    frames = []
    for tag in COLOR:
        p = ABL / f"{tag}_swap.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    frames = [f for f in frames if f["model"].iloc[0] in SWEEP]
    if not frames:
        return
    import numpy as np
    bands = ["early", "workspace", "late"]
    fig, ax = plt.subplots(figsize=(7.5, 4), constrained_layout=True)
    x = np.arange(len(bands))
    w = 0.8 / max(len(frames), 1)
    for i, df in enumerate(frames):
        tag = df["model"].iloc[0]
        d = df.set_index("band")
        swap = [d.loc[b, "swap_flip_rate"] if b in d.index else 0 for b in bands]
        ctrl = [d.loc[b, "control_flip_rate"] if b in d.index else 0 for b in bands]
        ax.bar(x + i * w, swap, w * 0.9, color=COLOR[tag], label=f"{tag} swap")
        ax.bar(x + i * w, ctrl, w * 0.9, facecolor="none", edgecolor=INK,
               hatch="xxxx", lw=0)
    ax.set_xticks(x + w * (len(frames) - 1) / 2, bands)
    ax.set_ylabel("answer flip-rate")
    ax.set_title("Causal swap by band: J-lens direction (solid) vs control (hatched)",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(frames))
    fig.savefig(FIGS / "causal.png", bbox_inches="tight")
    plt.close(fig)


def fig_quant_control(data):
    """8B bf16 vs int8: same color, solid vs dashed — quantization effect."""
    if "8b" not in data or "8b-int8" not in data:
        return
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), constrained_layout=True)
    for ax, (col, title) in zip(axes, METRICS[:3]):
        for tag, ls in (("8b", "-"), ("8b-int8", "--")):
            df = data[tag].sort_values("depth")
            ax.plot(df["depth"], df[col], color=COLOR["8b"], lw=2, ls=ls,
                    label="bf16" if ls == "-" else "int8")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("depth (0–100)")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Quantization control — 8B bf16 vs int8", fontsize=12, x=0.01,
                 ha="left")
    fig.savefig(FIGS / "quant_control.png", bbox_inches="tight")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    print("metrics available for:", sorted(data))
    fig_scale_metrics(data)
    fig_lenseval(data)
    fig_causal()
    fig_quant_control(data)
    print("wrote:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
