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

# categorical color BY OPERATOR, fixed order (validated: adjacent CVD dE 26.1, labels
# supply the sub-3:1 contrast relief). Used in the operator/operand geometry figures.
OP5 = ["#2a78d6", "#e0632f", "#1baf7a", "#9b5de5", "#eda100"]
# operand / operator / fusion for the variance split (blue vs orange = max CVD safety)
OPERAND_C, OPERATOR_C, FUSION_C = "#2a78d6", "#e0632f", MUTED

GEO = ROOT / "results" / "geometry"

# PAPER=1: print-targeted variants for the LaTeX builds -- white surface, serif
# text, figures drawn at (close to) their final physical size so type stays
# readable at \textwidth (~6.5in) instead of being shrunk from web size.
import os
PAPER = bool(os.environ.get("PAPER"))
if PAPER:
    FIGS = ROOT / "docs" / "figs" / "paper"
    SURF = "#ffffff"


def fs(w: float, h: float) -> tuple:
    """Figure size: web size, or normalized to ~7in width for print so type set at
    9-10.5pt renders near its nominal size once the PDF is included at
    \\textwidth (6.5in) — a 0.93 shrink, not the old 0.62."""
    return (7.0, h * 7.0 / w) if PAPER else (w, h)


def foot(fig, text: str, y: float = -0.04):
    """Figure footnote — web only. In print the LaTeX caption carries this text,
    and the tiny fig.text used to collide with the axes."""
    if not PAPER:
        fig.text(0.01, y, text, color=MUTED, fontsize=8.5, va="top")


def load_geo(tag: str, domain: str = "relations"):
    p = GEO / f"{tag}_{domain}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _savefig(fig, stem: str):
    """Write both a PNG (docs/site) and a vector PDF (paper) of a figure."""
    fig.savefig(FIGS / f"{stem}.png", bbox_inches="tight")
    fig.savefig(FIGS / f"{stem}.pdf", bbox_inches="tight")

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
    "font.size": 10, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130,
})
if PAPER:  # match the paper's Times body; hairline grid on white
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 9,
        "grid.color": "#e8e8e8", "figure.dpi": 200,
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


def fig_op_geometry():
    """Fig 1 -- where operand and operator live. Two PCA clouds of the 60 workspace
    vectors H[operand, operator], colored by operator: at the country token the cases
    intermix (operand-organized); at the query token they separate into case clusters
    (operator-organized). Faint webs link each country's 5 case-forms -- short at the
    country token, splayed at the query token: the concept is *declined*. Bottom: the
    variance split that quantifies the shift."""
    import numpy as np
    import matplotlib.patheffects as pe
    g, g8 = load_geo("1.7b"), load_geo("8b")
    if not g:
        return
    ops = g["operators"]
    opcol = {k: OP5[i % len(OP5)] for i, k in enumerate(ops)}

    fig = plt.figure(figsize=fs(11, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])
    for col, (cloud, title, sub) in enumerate([
        (g["cloud_country"], "At the country token",
         "colors intermixed — organized by operand (stem 59%)"),
        (g["cloud_query"], "At the query token",
         "colors cluster — organized by operator (case 86%)")]):
        ax = fig.add_subplot(gs[0, col])
        by_op = {}
        for pt in cloud:
            by_op.setdefault(pt["operand"], []).append(pt)
        for pts in by_op.values():  # web linking a country's 5 case-forms
            xs = [p["x"] for p in pts] + [pts[0]["x"]]
            ys = [p["y"] for p in pts] + [pts[0]["y"]]
            ax.plot(xs, ys, color=MUTED, lw=0.5, alpha=0.3, zorder=1)
        for pt in cloud:
            ax.scatter(pt["x"], pt["y"], color=opcol[pt["operator"]], s=42,
                       zorder=3, edgecolor=SURF, linewidth=0.7)
        if col == 1:  # direct operator labels at the query-token clusters
            for k, (x, y) in g["centroids_query"].items():
                ax.annotate(k, (x, y), color=opcol[k], fontsize=10, fontweight="bold",
                            ha="center", va="center", zorder=6, xytext=(0, 13),
                            textcoords="offset points",
                            path_effects=[pe.withStroke(linewidth=3.5, foreground=SURF)])
        ax.set_title(title, fontsize=12, loc="left")
        ax.text(0.0, -0.06, sub, transform=ax.transAxes, color=MUTED, fontsize=9.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal", adjustable="datalim")

    axb = fig.add_subplot(gs[1, :])
    groups, stems, cases, inter = [], [], [], []
    for tag, gg in (("1.7B", g), ("8B", g8)):
        if not gg:
            continue
        for pos in ("country", "query"):
            v = gg["variance"][pos]
            groups.append(f"{tag}\n{pos} token")
            stems.append(v["stem"]); cases.append(v["case"]); inter.append(v["interaction"])
    x = np.arange(len(groups))
    axb.bar(x, stems, 0.6, color=OPERAND_C, label="operand")
    axb.bar(x, cases, 0.6, bottom=stems, color=OPERATOR_C, label="operator")
    axb.bar(x, inter, 0.6, bottom=[s + c for s, c in zip(stems, cases)],
            color=FUSION_C, label="interaction (cell residual)")
    for i in range(len(groups)):
        axb.text(i, stems[i] / 2, f"{stems[i]:.0%}", ha="center", va="center",
                 color=SURF, fontsize=9, fontweight="bold")
        axb.text(i, stems[i] + cases[i] / 2, f"{cases[i]:.0%}", ha="center",
                 va="center", color=SURF, fontsize=9, fontweight="bold")
    axb.set_xticks(x, groups, fontsize=9.5)
    axb.set_ylim(0, 1)
    axb.set_ylabel("variance share")
    axb.set_title("Re-marked along the sequence: operand-dominant → operator-dominant  "
                  "(interaction ~9–13%, subspaces 41–82° apart)",
                  fontsize=10.5, loc="left")
    axb.legend(frameon=False, fontsize=9, ncol=3, loc="upper center")
    axb.grid(axis="x")
    if not PAPER:
        fig.suptitle("Where operand and operator live", fontsize=14, x=0.01, ha="left")
    _savefig(fig, "op_geometry")
    plt.close(fig)


def fig_op_swap():
    """Fig 2 -- what happens when you inject. All-pairs relational operator swap as a
    from×to heatmap: clean (baseline, 'from' wins, blue) vs after injecting
    v(to)−v(from) ('to' wins, red). 20/20 pairs flip sign; matched-norm random ~0."""
    import numpy as np
    g = load_geo("1.7b")
    if not g or not g["swap"]:
        return
    ops = g["operators"]
    n = len(ops)
    clean = np.full((n, n), np.nan)
    swap = np.full((n, n), np.nan)
    rand = []
    for i, a in enumerate(ops):
        for j, b in enumerate(ops):
            c = g["swap"].get(a, {}).get(b)
            if c:
                clean[i, j], swap[i, j] = c["clean"], c["swap"]
                rand.append(c["random"])
    vmax = float(np.nanmax([np.nanmax(np.abs(clean)), np.nanmax(np.abs(swap))]))
    flips = int(np.nansum(swap > 0))
    tot = int(np.sum(~np.isnan(swap)))

    fig, axes = plt.subplots(1, 2, figsize=fs(11, 4.8), constrained_layout=True)
    for ax, M, title in ((axes[0], clean, "Clean — “from” wins (baseline)"),
                         (axes[1], swap, "Inject  v(to) − v(from)  — “to” wins")):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(n), ops, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(n), ops, fontsize=9)
        ax.set_xlabel("to (target operator)")
        if ax is axes[0]:
            ax.set_ylabel("from (source operator)")
        for i in range(n):
            for j in range(n):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.0f}", ha="center", va="center",
                            fontsize=8.5, color=INK if abs(M[i, j]) < vmax * 0.55 else SURF)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(False)
    fig.colorbar(im, ax=axes, shrink=0.8, label="logit(to) − logit(from)")
    fig.suptitle(f"Operator injection flips the logit margin: {flips}/{tot} pairs flip sign "
                 f"(matched-norm random control mean {np.mean(rand):+.1f})",
                 fontsize=12.5, x=0.01, ha="left")
    _savefig(fig, "op_swap")
    plt.close(fig)


def fig_op_swap_dist():
    """Fig 2b -- the injection effect with its uncertainty. One row per ordered pair:
    per-operand swap values (dots), per-operand random-control values (gray x), the
    operand-bootstrap 95% CI on the swap mean, and the clean baseline mean (open
    circle). States the dependence structure honestly: 20 ordered pairs share 5
    operator directions, so operands are the independent unit within a pair; the
    family-level CI (operators as clusters) is quoted in the title."""
    import numpy as np
    p = ABL / "1.7b_relations_operator_swap_long.parquet"
    if not p.exists():
        return
    ldf = pd.read_parquet(p)
    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    import op_core
    ci = op_core.bootstrap_pair_ci(ldf, seed=0)
    fam = op_core.bootstrap_family_ci(ldf, seed=0)
    ci = ci.sort_values("swap_mean").reset_index(drop=True)  # ascending: best on top

    fig, ax = plt.subplots(figsize=fs(8.2, 7.6), constrained_layout=True)
    rng = np.random.default_rng(0)
    for i, r in ci.iterrows():
        sub = ldf[(ldf["from"] == r["from"]) & (ldf["to"] == r["to"])]
        jit = (rng.random(len(sub)) - 0.5) * 0.42
        ax.scatter(sub["random"], i + jit, s=14, marker="x", color=MUTED,
                   alpha=0.75, lw=1, zorder=2)
        ax.scatter(sub["swap"], i + jit, s=17, color=OPERATOR_C, alpha=0.85,
                   zorder=3, edgecolor=SURF, linewidth=0.4)
        ax.plot([r["swap_lo"], r["swap_hi"]], [i, i], color=INK, lw=2.4,
                zorder=4, solid_capstyle="round")
        ax.plot(r["swap_mean"], i, "o", color=INK, ms=4.5, zorder=5)
        ax.plot(sub["clean"].mean(), i, "o", mfc="none", mec=OPERAND_C, ms=6,
                mew=1.4, zorder=4)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(ci)),
                  [f"{r['from']} → {r['to']}" for _, r in ci.iterrows()], fontsize=8.5)
    ax.set_xlabel("logit(to) − logit(from) per operand")
    ax.set_title(
        f"Every ordered operator swap flips, with uncertainty shown\n"
        f"family level (operators as clusters): contrast {fam['contrast_mean']:+.1f} "
        f"[{fam['contrast_lo']:+.1f}, {fam['contrast_hi']:+.1f}] · flip fraction "
        f"{fam['flip_frac']:.0%} [{fam['flip_lo']:.0%}, {fam['flip_hi']:.0%}]",
        fontsize=10, loc="left")
    n_per = int(ldf.groupby(["from", "to"]).size().median())
    foot(fig,
             f"dots = swap per operand (orange, median {n_per}/pair; 4 on the syncretic "
             f"demonym–language pairs) · gray × = matched-norm random · "
             f"bar = operand-bootstrap 95% CI · ○ = clean baseline mean\n"
             f"20 ordered pairs share 5 operator directions — pairs are not independent; "
             f"operands are the unit within a pair, operators across the paradigm.",
             y=-0.015)
    _savefig(fig, "op_swap_dist")
    plt.close(fig)


def fig_op_dose():
    """Fig 2c -- dose-response and collateral cost of the intervention. (a) the
    operator-specific effect (swap minus matched-norm random) saturates at the
    default dose alpha=4; the random control's nonspecific shift is flat in alpha.
    (b) the band-wide intervention is not free: per-token KL on unrelated WikiText
    grows monotonically with dose. Honest cost accounting, one axis per panel."""
    p = ABL / "1.7b_relations_dose.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p).sort_values("alpha")
    signed = bool((df["alpha"] < 0).any())  # the +/- sweep: does the effect reverse?
    fig, axes = plt.subplots(1, 2, figsize=fs(9.2, 3.6), constrained_layout=True)

    ax = axes[0]
    spec = df["swap_shift"] - df["random_shift"]
    ax.plot(df["alpha"], spec, color=OPERATOR_C, lw=2, marker="o", ms=4,
            label="operator-specific (swap − random)")
    ax.plot(df["alpha"], df["random_shift"], color=MUTED, lw=2, marker="x", ms=5,
            label="nonspecific (matched-norm random)")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.axvline(4.0, color=MUTED, lw=0.8, ls=":")
    ax.text(4.15, ax.get_ylim()[0] + 1.5, "default α=4", color=MUTED, fontsize=8.5)
    ax.set_xlabel("intervention strength α")
    ax.set_ylabel("logit-margin shift")
    if signed:
        ax.axhline(0, color=MUTED, lw=1)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title("A signed axis: reversing α\nreverses the preference",
                     fontsize=10.5, loc="left")
    else:
        ax.set_title("Efficacy saturates at the default dose", fontsize=10.5, loc="left")

    ax2 = axes[1]
    ax2.plot(df["alpha"], df["kl_nats"], color=OPERAND_C, lw=2, marker="o", ms=4)
    ax2.axvline(4.0, color=MUTED, lw=0.8, ls=":")
    ax2.set_xlabel("intervention strength α")
    ax2.set_ylabel(r"KL(clean $\Vert$ intervened), nats/token")
    ax2.set_title("…but the band-wide edit is\nnot free off-task",
                  fontsize=10.5, loc="left")
    foot(fig,
         "Right: per-token KL on unrelated WikiText with the hook active at every "
         "position of the workspace band — the intervention is answer-surgical, "
         "not distribution-surgical.", y=-0.03)
    _savefig(fig, "op_dose")
    plt.close(fig)


def fig_op_syncretism():
    """Fig 4 -- operation ≠ realization. language & demonym both emit 'Italian' yet are
    distinct operator directions (their cosine is the least anti-aligned pair, but not
    1). The pure desinence, built where the two share an output word, still installs the
    relation causally."""
    import numpy as np
    from matplotlib.patches import Rectangle
    g = load_geo("1.7b")
    if not g:
        return
    labels = g["cos"]["labels"]
    M = np.array(g["cos"]["matrix"])
    n = len(labels)
    des = g["desinence"]

    fig, axes = plt.subplots(1, 2, figsize=fs(11, 4.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    Mm = M.copy()
    np.fill_diagonal(Mm, np.nan)
    vmax = float(np.nanmax(np.abs(Mm)))
    im = ax.imshow(Mm, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(n), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(n), labels, fontsize=9)
    for i in range(n):
        for j in range(n):
            if not np.isnan(Mm[i, j]):
                ax.text(j, i, f"{Mm[i, j]:+.2f}", ha="center", va="center", fontsize=8.5,
                        color=INK if abs(Mm[i, j]) < vmax * 0.6 else SURF)
    if des and des["pair"][0] in labels and des["pair"][1] in labels:
        a, b = labels.index(des["pair"][0]), labels.index(des["pair"][1])
        for (r, c) in ((a, b), (b, a)):
            ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False,
                                   edgecolor=OPERATOR_C, lw=2.5))
    ax.grid(False)
    ax.set_title("Five distinct directions — language & demonym\n(boxed) share "
                 "“Italian” yet differ (cos −0.26)", fontsize=10, loc="left")

    ax2 = axes[1]
    if des:
        vals = [des["clean"], des["desinence"]]
        bars = ax2.bar([0, 1], vals, 0.55, color=[MUTED, OPERATOR_C])
        ax2.axhline(0, color=MUTED, lw=1)
        ax2.set_xticks([0, 1], ["clean\n(currency prompt)", "+ exponent-free\nmarker"], fontsize=9.5)
        ax2.set_ylabel(f"logit({des['pair'][0]}) − logit({des['other']})")
        ax2.set_ylim(min(vals) - 2, max(vals) + 3)
        for bx, v in zip(bars, vals):
            ax2.text(bx.get_x() + bx.get_width() / 2, v + (0.5 if v >= 0 else -0.5),
                     f"{v:+.1f}", ha="center", va="bottom" if v >= 0 else "top",
                     fontsize=10.5, fontweight="bold", color=INK)
    ax2.set_title("The exponent-free marker still\ninstalls the relation",
                  fontsize=10, loc="left")
    if not PAPER:
        fig.suptitle("Operation ≠ realization: the case is separable from the word",
                     fontsize=12.5, x=0.01, ha="left")
    _savefig(fig, "op_syncretism")
    plt.close(fig)


def fig_op_patch(tag: str = "1.7b"):
    """The headline of the revision -- what does the donor activation carry that the
    operator direction does not? Left: greedy exact match (does the model SAY it?).
    Right: the logit margin (does it PREFER it?). The two panels are the paper's two
    levels of evidence, and the variants that move one without the other are the
    result."""
    import numpy as np
    p = ABL / f"{tag}_relations_patch_decomp.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    meta = s.pop("_meta")
    order = ["magnitude control", "interaction only", "operand only",
             "operator + wrong operand", "operator only",
             "operator + operand (held-out cell)", "operator + operand",
             "full (donor)"]
    order = [v for v in order if v in s]
    em = [s[v]["exact_match"] for v in order]
    dm = [s[v]["delta_margin"] for v in order]
    lo = [s[v]["delta_margin_lo"] for v in order]
    hi = [s[v]["delta_margin_hi"] for v in order]
    # the real donor is the reference bar; the two additive reconstructions that
    # match it are the result -- everything else is a control.
    HEAD = {"operator + operand", "operator + operand (held-out cell)"}
    color = [OPERATOR_C if v == "full (donor)" else
             OPERAND_C if v in HEAD else MUTED for v in order]

    fig, axes = plt.subplots(1, 2, figsize=fs(9.4, 4.0), constrained_layout=True)
    y = np.arange(len(order))

    ax = axes[0]
    ax.barh(y, em, 0.62, color=color)
    ax.axvline(meta["clean_exact_from"], color=INK, lw=1, ls="--")
    ax.text(meta["clean_exact_from"], len(order) - 0.35,
            f"  clean accuracy {meta['clean_exact_from']:.0%}",
            color=INK, fontsize=8.5, va="center")
    for i, v in enumerate(em):
        ax.text(v + 0.012, i, f"{v:.0%}", va="center", fontsize=9.5, color=INK)
    ax.set_yticks(y, order, fontsize=9.5)
    ax.set_xlim(0, max(max(em), meta["clean_exact_from"]) * 1.28)
    ax.set_xlabel("target contained in 3-token greedy continuation")
    ax.set_title("Behavioral sufficiency: does the model SAY it?",
                 fontsize=10.5, loc="left")
    ax.grid(axis="y", visible=False)

    ax2 = axes[1]
    ax2.barh(y, dm, 0.62, color=color)
    ax2.errorbar(dm, y, xerr=[np.array(dm) - np.array(lo), np.array(hi) - np.array(dm)],
                 fmt="none", ecolor=INK, elinewidth=1.4, capsize=2.5)
    ax2.axvline(0, color=MUTED, lw=1)
    ax2.set_yticks(y, ["" for _ in order])
    ax2.set_xlabel("Δ logit margin toward the target (95% CI)")
    ax2.set_title("Causal influence: does it PREFER it?", fontsize=10.5, loc="left")
    ax2.grid(axis="y", visible=False)

    if not PAPER:  # in print the LaTeX caption carries the headline
        fig.suptitle("A state composed from the factorization's parts generates the "
                     "answer — the interaction term is not needed",
                     fontsize=12.5, x=0.01, ha="left")
    foot(fig,
         f"Query-position patch at the workspace band, {tag}, "
         f"{meta['n_cells']} (pair, operand) cells. Every variant includes μ (the hook "
         f"REPLACES the residual, so a variant must be a plausible state).\n"
         f"“full (donor)” = μ + operand + operator + interaction is the real donor "
         f"activation by construction. “held-out cell” rebuilds the operand and "
         f"operator components WITHOUT ever seeing the target cell.\n"
         f"CIs: operator-level cluster bootstrap.",
         y=-0.035)
    _savefig(fig, "op_patch")
    plt.close(fig)


def fig_op_positions(tag: str = "1.7b"):
    """Is the margin effect a localized edit of the queried relation, or a global
    perturbation? Cross WHERE the vector lands with HOW WIDE the layer scope is,
    and show the metric the flips hide: the target's rank in the vocabulary."""
    import numpy as np
    p = ABL / f"{tag}_relations_positions.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    meta = s.pop("_meta")
    poss = ["all", "query", "operand", "wrong"]
    scopes = ["band", "single"]
    fig, axes = plt.subplots(1, 3, figsize=fs(11.0, 3.7), constrained_layout=True)
    x = np.arange(len(poss))
    w = 0.36
    C = {"band": OPERATOR_C, "single": OPERAND_C}

    for j, (key, label, title) in enumerate([
            ("delta_margin", "Δ logit margin", "Causal influence on the margin"),
            ("exact_match", "greedy exact match", "Behavioral sufficiency"),
            ("kl_offtask", "KL nats/token, unrelated text", "Off-task collateral")]):
        ax = axes[j]
        for i, sc in enumerate(scopes):
            vals = [s[f"{sc}/{pn}"].get(key) or 0.0 for pn in poss]
            b = ax.bar(x + (i - 0.5) * w, vals, w, color=C[sc], label=sc)
            if key == "delta_margin":
                lo = [s[f"{sc}/{pn}"]["delta_margin_lo"] for pn in poss]
                hi = [s[f"{sc}/{pn}"]["delta_margin_hi"] for pn in poss]
                ax.errorbar(x + (i - 0.5) * w, vals,
                            yerr=[np.array(vals) - np.array(lo),
                                  np.array(hi) - np.array(vals)],
                            fmt="none", ecolor=INK, elinewidth=1.2, capsize=2)
        ax.axhline(0, color=MUTED, lw=1)
        ax.set_xticks(x, poss, fontsize=9.5)
        ax.set_xlabel("position of the added vector")
        ax.set_ylabel(label)
        ax.set_title(title, fontsize=10.5, loc="left")
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.legend(frameon=False, fontsize=9, title="layer scope",
                      title_fontsize=9)
    fig.suptitle("Where the operator vector has to land — and what it does not buy",
                 fontsize=12.5, x=0.01, ha="left")
    foot(fig,
             f"{tag}, α={meta['alpha']}. “query” = the last prompt token (where the "
             f"direction is read); “operand” = the entity token; “wrong” = the "
             f"sentence-initial token (structurally irrelevant, but still upstream of "
             f"the answer). Bars are means over all ordered pairs × operands; CIs are "
             f"operator-level cluster bootstraps.",
             y=-0.05)
    _savefig(fig, "op_positions")
    plt.close(fig)


def fig_op_nulls(tag: str = "1.7b"):
    """Forest plot of the null battery, in two labeled groups. Semantic nulls (same
    statistics, destroyed content) go to zero -- the decisive result. Structural
    probes stay nonzero for mechanistically informative reasons: the margin metric
    decomposes into uninstall-source + install-category components, and residual
    additions persist across layers."""
    import numpy as np
    p = ABL / f"{tag}_relations_nulls.json"
    if not p.exists():
        return
    meta = json.loads(p.read_text())
    byname = {r["null"]: r for r in meta["nulls"]}
    # (row label, key, group) bottom-to-top; None = spacer
    layout = [
        ("wrong layer (additions persist)", "wrong layer (early band)", "B"),
        ("other relation (−source alone)", "other-relation direction", "B"),
        ("shuffled answers (category boost)", "shuffled answers", "B"),
        (None, None, None),
        ("random in operator subspace", "random in operator subspace", "A"),
        ("permuted labels (global)", "permuted labels (global)", "A"),
        ("permuted labels (per-operand)", "permuted labels (per-operand)", "A"),
        (None, None, None),
        ("real swap", "real swap", "real"),
    ]
    fig, ax = plt.subplots(figsize=fs(8.8, 4.6), constrained_layout=True)
    ylabels = []
    for i, (label, key, grp) in enumerate(layout):
        ylabels.append(label or "")
        if key is None:
            continue
        r = byname[key]
        c = OPERATOR_C if grp == "real" else (INK if grp == "A" else MUTED)
        ax.plot([r["contrast_lo"], r["contrast_hi"]], [i, i], color=c, lw=2.6,
                solid_capstyle="round")
        ax.plot(r["contrast"], i, "o", color=c, ms=6.5, mec=SURF, mew=0.8, zorder=3)
        ax.text(max(r["contrast_hi"], 0) + 0.7, i, f"{r['contrast']:+.1f}",
                va="center", fontsize=9.5,
                color=c, fontweight="bold" if grp == "real" else "normal")
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(len(layout)), ylabels, fontsize=10.5)
    ax.set_xlabel("contrast vs matched-norm random (logit units), 95% CI")
    ax.grid(axis="y", visible=False)
    ax.text(5.0, 7.0, "semantic nulls → 0: the content of the direction is everything",
            fontsize=9.5, color=INK, style="italic", va="center")
    ax.text(5.0, 3.0, "structural probes: what the margin metric itself is made of",
            fontsize=9.5, color=MUTED, style="italic", va="center")
    ax.set_title("Same statistics, destroyed semantics → no effect",
                 fontsize=12, loc="left")
    foot(fig,
             f"{tag}, α={meta['alpha']}, {meta['n_seeds']} redraws per stochastic null. "
             f"Permuted labels rebuild directions from the SAME residuals with operator "
             f"labels shuffled; the subspace null draws inside the span of the real "
             f"directions (right home, wrong content).\nStructural probes are nonzero "
             f"for known reasons: −case(source) alone contributes ≈half the margin "
             f"(uninstall); +case(target) boosts the whole answer category (operand "
             f"specificity lives in generation, §composition); and additions to the "
             f"residual stream persist downstream, so an early-layer injection is not "
             f"actually absent from the workspace.",
             y=-0.05)
    _savefig(fig, "op_nulls")
    plt.close(fig)


def fig_op_layer_sweep(tag: str = "1.7b"):
    """Where the operator is most READABLE vs where it is most CAUSAL. If those two
    peaks coincide, the band story is complete; if they diverge, readability and
    control are different properties of the same direction."""
    p = ABL / f"{tag}_relations_layersweep.parquet"
    if not p.exists():
        return
    df = pd.read_parquet(p).sort_values("depth")
    meta = json.loads((ABL / f"{tag}_relations_layersweep.json").read_text())
    fig, ax = plt.subplots(figsize=fs(8.6, 4.0), constrained_layout=True)
    lo, hi = BANDS["workspace"]
    ax.axvspan(lo, hi, color=GRID, alpha=0.55, lw=0)
    ax.text(lo + 1, 0.035, "workspace band", color=MUTED, fontsize=8.5,
            transform=ax.get_xaxis_transform())

    # LOO decodability is at ceiling at EVERY layer (the operator token is in the
    # prompt), so the informative representational curve is the ANOVA operator share.
    ax.plot(df["depth"], df["case_share"], color=OPERAND_C, lw=2, marker="o", ms=3.5)
    ax.set_ylabel("operator variance share (ANOVA)", color=OPERAND_C)
    ax.set_xlabel("depth (% of layers)")
    ax.set_ylim(0, 1.05)

    ax2 = ax.twinx()
    ax2.plot(df["depth"], df["contrast"], color=OPERATOR_C, lw=2, marker="s", ms=3.5)
    ax2.fill_between(df["depth"], df["contrast_lo"], df["contrast_hi"],
                     color=OPERATOR_C, alpha=0.14, lw=0)
    ax2.set_ylabel("single-layer causal contrast (logits)", color=OPERATOR_C)
    ax2.grid(False)

    pk, pc = meta["peak_case_share"], meta["peak_causal"]
    ax.annotate(f"factorization peaks\nL{pk['layer']} ({pk['depth']:.0f}%)",
                (pk["depth"], pk["value"]), color=OPERAND_C, fontsize=9,
                xytext=(4, 10), textcoords="offset points")
    ax2.annotate(f"causal leverage peaks\nL{pc['layer']} ({pc['depth']:.0f}%)",
                 (pc["depth"], pc["value"]), color=OPERATOR_C, fontsize=9,
                 xytext=(-10, 8), textcoords="offset points", ha="right")
    ax.set_title("Where the operator is represented is not where it acts",
                 fontsize=12, loc="left")
    dec = df["decodability"]
    foot(fig,
             f"{tag}. Blue: operator variance share of the two-way ANOVA at that layer "
             f"(query position). Orange: all-pairs swap contrast when the direction is "
             f"built AND injected at that single layer (shaded = 95% CI).\n"
             f"Held-out-operand decodability is {dec.min():.0%}–{dec.max():.0%} at "
             f"every layer — the operator token is present in the prompt, so mere "
             f"readability locates nothing; these two curves do.",
             y=-0.04)
    _savefig(fig, "op_layer_sweep")
    plt.close(fig)


def fig_method():
    """The method in one schematic: the state grid h_{l,t} (one residual vector per
    layer per token), the two reading positions, the averaging that produces the
    per-layer operator direction, and the three ways the same object is applied.
    Requested by a reader: the compressed H[operand,operator] notation hides that
    every measurement fixes a layer AND a position."""
    import numpy as np
    tokens = ["The", "currency", "of", "Italy", "is"]
    n_l = 6
    fig, axes = plt.subplots(1, 2, figsize=fs(10.4, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # --- left: the grid ------------------------------------------------------
    ax = axes[0]
    ax.set_xlim(-0.7, len(tokens) - 0.3)
    ax.set_ylim(-3.1, n_l + 0.9)
    ax.axis("off")
    for t in range(len(tokens)):
        col = OPERAND_C if t == 3 else OPERATOR_C if t == 4 else MUTED
        ax.annotate("", (t, n_l - 0.7), (t, 0.3),
                    arrowprops=dict(arrowstyle="-|>", color=GRID, lw=3))
        for l in range(n_l):
            ax.plot(t, l, "o", color=col, ms=7 if t >= 3 else 5,
                    alpha=1.0 if t >= 3 else 0.55, zorder=3)
        ax.text(t, -0.75, tokens[t], ha="center", fontsize=10.5,
                color=col, fontweight="bold" if t >= 3 else "normal")
    ax.text(3, -1.7, "entity token\n(operand read here)", ha="center", fontsize=9,
            color=OPERAND_C)
    ax.text(4.3, -2.5, "query token\n(answer read here)", ha="center", fontsize=9,
            color=OPERATOR_C)
    for l in range(n_l):
        ax.text(-0.55, l, f"$\\ell={l}$" if l < n_l - 1 else "…", va="center",
                fontsize=9, color=MUTED)
    # band shading
    ax.axhspan(2, 4.5, xmin=0.04, xmax=0.99, color=GRID, alpha=0.5, lw=0)
    ax.text(0.35, 2.18, "workspace band: directions built here", fontsize=9,
            color=MUTED)
    ax.plot(4, 3, "o", ms=13, mfc="none", mec=INK, mew=1.6, zorder=4)
    ax.annotate("$h_{\\ell,t}$ — one vector per layer per position",
                xy=(4, 3), xytext=(1.6, 5.55), textcoords="data",
                fontsize=9, color=INK, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK, lw=1,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_title("The state is a grid: layers × positions", fontsize=11, loc="left")

    # --- right: averaging and the three applications -------------------------
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("One object, applied three ways", fontsize=11, loc="left")
    ax.text(0.2, 9.3, "at each band layer $\\ell$, average the query-token states:",
            fontsize=9.5, color=INK)
    ax.text(0.6, 8.15, r"$v_\ell(\mathrm{op}) = \mathrm{mean}_o\,[\,h_{\ell,-1}(o,"
                       r"\mathrm{op}) - \mathrm{mean}_k\, h_{\ell,-1}(o,k)\,]$",
            fontsize=10.5, color=INK)
    ax.text(0.2, 6.9, r"$\equiv$ the ANOVA operator main effect $b_\ell(\mathrm{op})$"
                      " (verified: dev. $10^{-8}$)",
            fontsize=9.5, color=OPERAND_C)
    for y, (name, desc, c) in enumerate([
        ("steer", r"add $\alpha\,(v_\ell(B)-v_\ell(A))$ at the query token"
                  " → margin flips; generates at calibrated $\\alpha$", OPERATOR_C),
        ("compose", r"replace $h_{\ell,-1}$ with $\mu_\ell + a_\ell(o) + b_\ell(k)$"
                    " → generates at the clean ceiling", OPERAND_C),
        ("decode", "classify / ANOVA the same states → factorization,"
                   " layer profiles", MUTED),
    ]):
        yy = 5.4 - y * 1.7
        ax.text(0.4, yy, name, fontsize=10, fontweight="bold", color=c)
        ax.text(2.6, yy, desc, fontsize=9.2, color=INK, va="center", wrap=True)
    foot(fig,
             "Prompt → token × layer grid of residual states → per-layer operator "
             "direction by averaging at the query token → applied by addition "
             "(steering), by replacement (composition), or read out (ANOVA).",
             y=-0.03)
    _savefig(fig, "op_method")
    plt.close(fig)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    data = load_metrics()
    print("metrics available for:", sorted(data))
    fig_scale_metrics(data)
    fig_lenseval(data)
    fig_causal()
    fig_quant_control(data)
    fig_op_geometry()
    fig_op_swap()
    fig_op_swap_dist()
    fig_op_dose()
    fig_op_syncretism()
    fig_op_patch()
    fig_op_positions()
    fig_op_nulls()
    fig_op_layer_sweep()
    print("wrote:", sorted(p.name for p in FIGS.glob("*.png")))


if __name__ == "__main__":
    main()
