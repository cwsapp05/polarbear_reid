"""Figures. Fixed series colours (validated for colour-vision deficiency on the light surface):
MegaDescriptor = blue, BearID = orange, fused = near-black ink (separated from both by lightness).
Neutral grey is reserved for chance lines / negative pairs."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = {"megadescriptor": "#2a78d6", "bearid": "#eb6834", "fused": "#1f1f1d"}
LABELS = {"megadescriptor": "MegaDescriptor (body)", "bearid": "BearID (face)", "fused": "Fused (body + face)"}
SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#1f1f1d", "#5f5e58", "#e6e5e0", "#9a9890"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def face_width_histogram(widths, thresholds, primary, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    if len(widths):
        bins = np.arange(0, max(np.nanmax(widths), max(thresholds)) + 4, 2)
        ax.hist(widths, bins=bins, color=COLORS["megadescriptor"], edgecolor=SURFACE, linewidth=1)
    for t in thresholds:
        if t <= 0:
            continue
        ax.axvline(t, color=INK if t == primary else NEUTRAL, lw=1.5 if t == primary else 1, ls="--")
        ax.annotate(f"{t}px" + (" (primary)" if t == primary else ""), (t, 1), xycoords=("data", "axes fraction"),
                    xytext=(3, -12), textcoords="offset points", fontsize=8, color=INK2)
    ax.set(xlabel="Face box width (original pixels)", ylabel="Frames",
           title="Face size of frames with a valid BearID face")
    _save(fig, path)


def roc_curves(curves: dict, title: str, path):
    """curves: {model: (fpr, tpr, auc, (lo, hi))}"""
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--", label="Chance (0.5)")
    for m, (fpr, tpr, auc, ci) in curves.items():
        ax.plot(fpr, tpr, color=COLORS[m], lw=2, label=f"{LABELS[m]}  AUROC {auc:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
    ax.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="False positive rate (different bears)",
           ylabel="True positive rate (same bear)", title=title)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def similarity_hists(data: dict, title: str, path):
    """data: {model: (pos_scores, neg_scores, metric_name)} - one panel per model."""
    fig, axes = plt.subplots(1, len(data), figsize=(4.4 * len(data), 3.4), squeeze=False)
    for ax, (m, (pos, neg, metric)) in zip(axes[0], data.items()):
        lo, hi = np.percentile(np.concatenate([pos, neg]), [0.5, 99.5])
        bins = np.linspace(lo, hi, 50)
        ax.hist(neg, bins=bins, density=True, color=NEUTRAL, alpha=0.55, label="Different bears")
        ax.hist(pos, bins=bins, density=True, histtype="step", color=COLORS[m], lw=2, label="Same bear")
        ax.set(title=LABELS[m], xlabel=metric, ylabel="Density")
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    _save(fig, path)


def auc_by_threshold(rows: list[dict], path):
    """rows: [{'subset_w': int, 'model': str, 'auc': float, 'lo': float, 'hi': float}]"""
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ws = sorted({r["subset_w"] for r in rows})
    x = {w: i for i, w in enumerate(ws)}
    present = [m for m in LABELS if any(r["model"] == m for r in rows)]
    for k, m in enumerate(present):
        rr = [r for r in rows if r["model"] == m]
        xs = np.array([x[r["subset_w"]] for r in rr]) + (k - (len(present) - 1) / 2) * 0.18
        ys = np.array([r["auc"] for r in rr])
        err = np.array([[r["auc"] - r["lo"] for r in rr], [r["hi"] - r["auc"] for r in rr]])
        ax.errorbar(xs, ys, yerr=err, fmt="o", ms=8, color=COLORS[m], ecolor=COLORS[m], elinewidth=2,
                    capsize=0, label=LABELS[m])
    ax.set_xticks(range(len(ws)), [f">= {w}px" if w > 0 else "any face" for w in ws])
    ax.set(xlabel="Face-visible subset (minimum face width)", ylabel="AUROC (95% bootstrap CI)",
           title="Verification AUROC on face-visible frames")
    ax.legend(fontsize=8, ncol=len(present), loc="upper center", bbox_to_anchor=(0.5, -0.2))
    _save(fig, path)
