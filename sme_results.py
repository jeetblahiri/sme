"""
sme_results.py  ─  Results Saving & Figure Generation for the SME Framework
=============================================================================
Saves everything produced by each experiment to disk:

  results/
  ├── tables/
  │   ├── e1_motor_imagery_results.csv
  │   ├── e1_motor_imagery_results.tex
  │   ├── e1_ablation.csv
  │   └── … (one set per experiment)
  └── figures/
      ├── e1_ablation_curve.pdf
      ├── e1_confusion_matrix.pdf
      ├── e1_perclass_f1.pdf
      ├── e1_results_bar.pdf
      ├── e2_ablation_curve.pdf
      ├── e2_perclass_f1.pdf
      ├── e2_confusion_matrix.pdf
      ├── e3_ablation_curve.pdf
      ├── e3_mp_deviation.pdf
      ├── e3_confusion_matrix.pdf
      ├── e4_ablation_curve.pdf
      ├── e4_mp_deviation.pdf
      ├── e4_kurtosis_heatmap.pdf
      ├── e4_confusion_matrix.pdf
      ├── e5_ablation_curve.pdf
      ├── e5_feature_importance.pdf
      └── e5_confusion_matrix.pdf

Call  save_experiment(exp_id, results_dict, ablation_dict, **extras)
from the run() function of each experiment.

All figures use a clean IEEE-style theme (no seaborn dependency —
only matplotlib + numpy).
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

import numpy as np

# ── output directories ─────────────────────────────────────────────────
_ROOT    = Path(__file__).parent
TABLES   = _ROOT / "results" / "tables"
FIGURES  = _ROOT / "results" / "figures"

for _d in (TABLES, FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# ── matplotlib setup ───────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — always works
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap

# IEEE-style rcParams
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         9,
    "axes.titlesize":    10,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "legend.fontsize":   8,
    "figure.dpi":        150,
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
    "lines.linewidth":   1.6,
    "lines.markersize":  5,
    "errorbar.capsize":  3,
})

# Colour palette (colour-blind safe)
_COLORS = ["#2196F3", "#F44336", "#4CAF50", "#FF9800",
           "#9C27B0", "#00BCD4", "#795548", "#607D8B"]


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _savefig(fig: plt.Figure, path: Path, close: bool = True) -> None:
    fig.savefig(str(path))
    if close:
        plt.close(fig)
    print(f"    saved → {path.relative_to(_ROOT)}")


def _bootstrap_ci(scores: np.ndarray, n: int = 2000,
                  seed: int = 42) -> tuple[float, float]:
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(scores, len(scores), replace=True).mean()
             for _ in range(n)]
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


# ══════════════════════════════════════════════════════════════════════════════
# 1.  TABLE SAVERS
# ══════════════════════════════════════════════════════════════════════════════

def save_results_csv(results: dict, path: Path) -> None:
    """Save method-level results to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Method", "Mean_Acc", "Std_Acc",
                    "Mean_F1", "Std_F1", "CI_lo", "CI_hi"])
        for name, r in results.items():
            lo, hi = _bootstrap_ci(r["fold_acc"])
            w.writerow([name,
                        f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}",
                        f"{r['mean_f1']:.4f}",  f"{r['std_f1']:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}"])
    print(f"    saved → {path.relative_to(_ROOT)}")


def save_results_latex(results: dict, path: Path, caption: str = "") -> None:
    """Save a LaTeX booktabs table."""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Method & Accuracy (\%) & Macro-F1 \\ \midrule",
    ]
    for name, r in results.items():
        a = rf"{r['mean_acc']*100:.1f}$\pm${r['std_acc']*100:.1f}"
        f = rf"{r['mean_f1']:.3f}$\pm${r['std_f1']:.3f}"
        lines.append(rf"{name} & {a} & {f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines))
    print(f"    saved → {path.relative_to(_ROOT)}")


def save_ablation_csv(ablation: dict, path: Path) -> None:
    """Save p-ablation results to CSV."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "Mean_Acc", "Std_Acc",
                    "Mean_F1", "Std_F1", "CI_lo", "CI_hi"])
        for p, r in sorted(ablation.items()):
            lo, hi = _bootstrap_ci(r["fold_acc"])
            w.writerow([p,
                        f"{r['mean_acc']:.4f}", f"{r['std_acc']:.4f}",
                        f"{r['mean_f1']:.4f}",  f"{r['std_f1']:.4f}",
                        f"{lo:.4f}", f"{hi:.4f}"])
    print(f"    saved → {path.relative_to(_ROOT)}")


def save_confusion_matrix_csv(cm: np.ndarray, class_names: list[str],
                               path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, row in enumerate(cm):
            w.writerow([class_names[i]] + row.tolist())
    print(f"    saved → {path.relative_to(_ROOT)}")


# ══════════════════════════════════════════════════════════════════════════════
# 2.  FIGURE: ABLATION CURVE
# ══════════════════════════════════════════════════════════════════════════════

def fig_ablation_curve(ablation: dict,
                       pocr_result: dict | None = None,
                       metric: str = "acc",
                       title: str = "",
                       path: Path | None = None,
                       extra_baselines: dict | None = None) -> plt.Figure:
    """
    Line plot of accuracy (or F1) vs moment order p, with 95% CI ribbon.
    Horizontal dashed lines for POCR and any extra baselines.
    """
    ps   = sorted(ablation.keys())
    vals = [ablation[p][f"mean_{metric}"] for p in ps]
    stds = [ablation[p][f"std_{metric}"]  for p in ps]
    cis  = [_bootstrap_ci(ablation[p]["fold_acc"]) for p in ps]
    lo   = [c[0] for c in cis]
    hi   = [c[1] for c in cis]

    ylabel = "Accuracy" if metric == "acc" else "Macro-F1"

    fig, ax = plt.subplots(figsize=(4.5, 3.0))

    # SME curve with CI ribbon
    ax.plot(ps, vals, "o-", color=_COLORS[0], label="SME (diagonal)", zorder=3)
    ax.fill_between(ps, lo, hi, alpha=0.18, color=_COLORS[0])

    # POCR horizontal reference
    if pocr_result is not None:
        pv = pocr_result[f"mean_{metric}"]
        ax.axhline(pv, ls="--", lw=1.4, color=_COLORS[1],
                   label=f"POCR/CRA ({pv:.3f})", zorder=2)

    # Extra baselines
    if extra_baselines:
        for i, (name, res) in enumerate(extra_baselines.items(), 2):
            bv = res[f"mean_{metric}"]
            ax.axhline(bv, ls=":", lw=1.2, color=_COLORS[i % len(_COLORS)],
                       label=f"{name} ({bv:.3f})", zorder=2)

    ax.set_xlabel("Moment order  $p$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.set_xticks(ps)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3.  FIGURE: RESULTS BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def fig_results_bar(results: dict,
                    metric: str = "acc",
                    title: str = "",
                    path: Path | None = None) -> plt.Figure:
    """Horizontal bar chart of mean accuracy ± std for all methods."""
    names = list(results.keys())
    vals  = [results[n][f"mean_{metric}"] for n in names]
    errs  = [results[n][f"std_{metric}"]  for n in names]
    cols  = [_COLORS[i % len(_COLORS)] for i in range(len(names))]

    ylabel = "Accuracy" if metric == "acc" else "Macro-F1"

    fig, ax = plt.subplots(figsize=(5.5, 0.55 * len(names) + 1.2))
    y = np.arange(len(names))
    bars = ax.barh(y, vals, xerr=errs, color=cols, alpha=0.85,
                   error_kw=dict(elinewidth=1.0, capsize=3), height=0.55)

    # Annotate bars with value
    for bar, v in zip(bars, vals):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}", va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel(ylabel)
    ax.set_title(title, pad=6)
    ax.set_xlim(0, min(1.0, max(vals) * 1.18))
    ax.invert_yaxis()
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4.  FIGURE: CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════════════════

def fig_confusion_matrix(cm: np.ndarray,
                         class_names: list[str],
                         title: str = "",
                         path: Path | None = None,
                         normalise: bool = True) -> plt.Figure:
    """Annotated confusion matrix heatmap."""
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot  = cm.astype(float) / row_sums
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm.astype(float)
        fmt, vmax = "d", cm.max()

    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(3.5, 0.65 * n + 1.0),
                                    max(3.0, 0.65 * n + 0.8)))

    cmap = LinearSegmentedColormap.from_list(
        "sme_cm", ["#ffffff", "#1565C0"], N=256)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap,
                   vmin=0, vmax=vmax, aspect="equal")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    thresh = cm_plot.max() / 2.0
    for i in range(n):
        for j in range(n):
            v = cm_plot[i, j]
            txt = (f"{v:{fmt}}" if fmt != "d"
                   else str(int(v)))
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if v > thresh else "black")

    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title, pad=8)
    ax.grid(False)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5.  FIGURE: PER-CLASS F1 BAR CHART
# ══════════════════════════════════════════════════════════════════════════════

def fig_perclass_f1(results_dict: dict,
                    class_names: list[str],
                    title: str = "",
                    path: Path | None = None,
                    highlight: list[int] | None = None) -> plt.Figure:
    """
    Grouped bar chart of per-class F1 for multiple methods.
    `highlight` is a list of class indices to mark with a star (key boundaries).
    """
    from sklearn.metrics import f1_score as _f1

    methods = list(results_dict.keys())
    n_cls   = len(class_names)
    n_meth  = len(methods)

    f1_matrix = np.zeros((n_meth, n_cls))
    for mi, (mname, res) in enumerate(results_dict.items()):
        for ci in range(n_cls):
            f1_matrix[mi, ci] = _f1(res["y_true"], res["y_pred"],
                                    labels=[ci], average="macro",
                                    zero_division=0)

    x     = np.arange(n_cls)
    w     = 0.8 / n_meth
    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n_cls + 1.5), 3.4))

    for mi, mname in enumerate(methods):
        offset = (mi - n_meth / 2 + 0.5) * w
        bars   = ax.bar(x + offset, f1_matrix[mi], w * 0.9,
                        label=mname, color=_COLORS[mi % len(_COLORS)],
                        alpha=0.85)

    # Star on highlighted classes
    if highlight:
        ymax = f1_matrix.max()
        for ci in highlight:
            ax.text(ci, ymax + 0.02, "★", ha="center",
                    fontsize=10, color="#E53935")

    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=30, ha="right")
    ax.set_ylabel("Per-class F1")
    ax.set_ylim(0, min(1.08, f1_matrix.max() + 0.12))
    ax.set_title(title, pad=6)
    ax.legend(loc="upper right", framealpha=0.9, ncol=min(n_meth, 3))
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 6.  FIGURE: MP DEVIATION  (E3, E4)
# ══════════════════════════════════════════════════════════════════════════════

def fig_mp_deviation(delta_matrix: np.ndarray,
                     class_names: list[str],
                     p_max: int,
                     title: str = "",
                     path: Path | None = None) -> plt.Figure:
    """
    Grouped bar chart of mean Δ_k per class and per moment order k.
    delta_matrix : (n_classes, p_max)
    """
    n_cls, p_m = delta_matrix.shape
    ks    = list(range(1, p_m + 1))
    x     = np.arange(n_cls)
    w     = 0.8 / p_m

    fig, ax = plt.subplots(figsize=(max(5.0, 0.9 * n_cls + 1.5), 3.4))

    for ki in range(p_m):
        offset = (ki - p_m / 2 + 0.5) * w
        ax.bar(x + offset, delta_matrix[:, ki], w * 0.9,
               label=f"$k={ks[ki]}$",
               color=_COLORS[ki % len(_COLORS)], alpha=0.85)

    ax.axhline(0, color="black", lw=0.8, ls="-")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel(r"$\Delta_k = \mathrm{Tr}(C^k)/N - \mathbb{E}_{\mathrm{MP}}[\lambda^k]$")
    ax.set_title(title, pad=6)
    ax.legend(loc="upper left", framealpha=0.9, ncol=p_m)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 7.  FIGURE: KURTOSIS HEATMAP  (E4)
# ══════════════════════════════════════════════════════════════════════════════

def fig_kurtosis_heatmap(kurt_matrix: np.ndarray,
                         class_names: list[str],
                         p_values: list[int],
                         title: str = "",
                         path: Path | None = None) -> plt.Figure:
    """
    Heatmap of trajectory kurtosis of SME components.
    kurt_matrix : (n_classes, len(p_values))
    """
    fig, ax = plt.subplots(figsize=(0.7 * len(p_values) + 1.5,
                                    0.55 * len(class_names) + 1.0))

    cmap = LinearSegmentedColormap.from_list(
        "kurt", ["#E3F2FD", "#0D47A1"], N=256)
    im = ax.imshow(kurt_matrix, cmap=cmap, aspect="auto")
    plt.colorbar(im, ax=ax, label="Excess kurtosis")

    for i in range(kurt_matrix.shape[0]):
        for j in range(kurt_matrix.shape[1]):
            ax.text(j, i, f"{kurt_matrix[i,j]:.1f}",
                    ha="center", va="center", fontsize=8,
                    color="white" if kurt_matrix[i,j] > kurt_matrix.max()*0.6
                    else "black")

    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"$p={p}$" for p in p_values])
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Moment order"); ax.set_ylabel("Fault class")
    ax.set_title(title, pad=6)
    ax.grid(False)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 8.  FIGURE: FEATURE IMPORTANCE HEATMAP  (E5)
# ══════════════════════════════════════════════════════════════════════════════

def fig_feature_importance(diag_imp: np.ndarray,
                           rsum_imp: np.ndarray,
                           ch_names: list[str],
                           title: str = "",
                           path: Path | None = None) -> plt.Figure:
    """
    Two-panel heatmap: diagonal importance and row-sum importance,
    indexed by (moment order p, channel j).
    diag_imp : (p_max, N_ch)
    rsum_imp : (p_max, N_ch)
    """
    p_max = diag_imp.shape[0]
    ylabs = [f"$p={k+1}$" for k in range(p_max)]
    cmap  = LinearSegmentedColormap.from_list(
        "imp", ["#F3E5F5", "#4A148C"], N=256)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 0.55 * p_max + 1.4),
                              sharey=True)
    for ax, mat, ttl in zip(axes,
                            [diag_imp, rsum_imp],
                            ["Diagonal projection", "Row-sum projection"]):
        vmax = max(diag_imp.max(), rsum_imp.max())
        im = ax.imshow(mat, cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
        for i in range(p_max):
            for j in range(len(ch_names)):
                ax.text(j, i, f"{mat[i,j]:.3f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if mat[i,j] > vmax*0.6 else "black")
        ax.set_xticks(range(len(ch_names)))
        ax.set_xticklabels(ch_names)
        ax.set_yticks(range(p_max))
        ax.set_yticklabels(ylabs)
        ax.set_title(ttl)
        ax.set_xlabel("Channel")
        ax.grid(False)
        plt.colorbar(im, ax=ax, fraction=0.046)

    axes[0].set_ylabel("Moment order")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


def fig_importance_by_order(total_per_p: np.ndarray,
                             p_values: list[int],
                             title: str = "",
                             path: Path | None = None) -> plt.Figure:
    """Bar chart of total GBT importance per moment order."""
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    cols = [_COLORS[i % len(_COLORS)] for i in range(len(p_values))]
    ax.bar(range(len(p_values)), total_per_p, color=cols, alpha=0.85,
           width=0.6)
    for i, v in enumerate(total_per_p):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(p_values)))
    ax.set_xticklabels([f"$p={p}$" for p in p_values])
    ax.set_ylabel("Relative feature importance")
    ax.set_ylim(0, total_per_p.max() * 1.2)
    ax.set_title(title, pad=6)
    fig.tight_layout()
    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 9.  FIGURE: DUAL ABLATION CURVE  (E3 — diag vs vech)
# ══════════════════════════════════════════════════════════════════════════════

def fig_dual_ablation(abl_a: dict, abl_b: dict,
                      label_a: str = "SME-diagonal",
                      label_b: str = "SME-vech",
                      pocr_result: dict | None = None,
                      metric: str = "acc",
                      title: str = "",
                      path: Path | None = None) -> plt.Figure:
    """Overlay two ablation curves on one axis (E3 diag vs vech)."""
    fig, ax = plt.subplots(figsize=(4.5, 3.0))
    ylabel = "Accuracy" if metric == "acc" else "Macro-F1"

    for abl, label, col, mk in [(abl_a, label_a, _COLORS[0], "o"),
                                  (abl_b, label_b, _COLORS[2], "s")]:
        ps   = sorted(abl.keys())
        vals = [abl[p][f"mean_{metric}"] for p in ps]
        cis  = [_bootstrap_ci(abl[p]["fold_acc"]) for p in ps]
        lo   = [c[0] for c in cis]
        hi   = [c[1] for c in cis]
        ax.plot(ps, vals, f"{mk}-", color=col, label=label, zorder=3)
        ax.fill_between(ps, lo, hi, alpha=0.15, color=col)

    if pocr_result is not None:
        pv = pocr_result[f"mean_{metric}"]
        ax.axhline(pv, ls="--", lw=1.4, color=_COLORS[1],
                   label=f"POCR/CRA ({pv:.3f})")

    ax.set_xlabel("Moment order  $p$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=6)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SIGNIFICANCE HEATMAP
# ══════════════════════════════════════════════════════════════════════════════

def fig_significance_heatmap(results: dict,
                              title: str = "",
                              path: Path | None = None) -> plt.Figure:
    """
    Heatmap of pairwise Wilcoxon p-values.
    Green cell = significant (p < 0.05), white = not significant.
    """
    from scipy import stats

    methods = list(results.keys())
    n = len(methods)
    pmat = np.ones((n, n))

    for i, ma in enumerate(methods):
        for j, mb in enumerate(methods):
            if i == j: continue
            a = results[ma]["fold_acc"]
            b = results[mb]["fold_acc"]
            nn = min(len(a), len(b))
            if nn < 5 or np.allclose(a[:nn], b[:nn]):
                pmat[i, j] = 1.0
                continue
            try:
                _, pval = stats.wilcoxon(a[:nn], b[:nn])
                pmat[i, j] = pval
            except Exception:
                pmat[i, j] = 1.0

    cmap = LinearSegmentedColormap.from_list(
        "sig", ["#1B5E20", "#A5D6A7", "#FFFFFF"], N=256)

    sz = max(3.5, 0.7 * n + 1.0)
    fig, ax = plt.subplots(figsize=(sz, sz * 0.9))
    im = ax.imshow(pmat, cmap=cmap, vmin=0, vmax=0.2, aspect="equal")
    plt.colorbar(im, ax=ax, label="p-value")

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
            else:
                sig = pmat[i, j] < 0.05
                txt = f"{pmat[i,j]:.3f}" + ("*" if sig else "")
                ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                        color="white" if pmat[i, j] < 0.03 else "black")

    ax.set_xticks(range(n))
    ax.set_xticklabels([m[:14] for m in methods], rotation=35, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels([m[:14] for m in methods])
    ax.set_title(title + "\n(* p < 0.05, Wilcoxon signed-rank)", pad=6)
    ax.grid(False)
    fig.tight_layout()

    if path:
        _savefig(fig, path, close=False)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# 11.  MASTER SAVE FUNCTION  (called from each experiment's run())
# ══════════════════════════════════════════════════════════════════════════════

def save_experiment(
    exp_id:       str,                  # "e1", "e2", …
    results:      dict,                 # {method_name: result_dict}
    ablation:     dict,                 # {p: result_dict}          (primary)
    class_names:  list[str],
    primary_metric: str = "acc",        # "acc" or "f1"
    caption:      str = "",
    # Optional extras passed as keyword args:
    ablation_b:   dict | None = None,   # second ablation (E3 vech)
    label_a:      str = "SME-diagonal",
    label_b:      str = "SME-vech",
    mp_delta:     np.ndarray | None = None,   # (n_cls, p_max) — E3/E4
    kurt_matrix:  np.ndarray | None = None,   # (n_cls, p_max) — E4
    diag_imp:     np.ndarray | None = None,   # (p_max, N_ch)  — E5
    rsum_imp:     np.ndarray | None = None,   # (p_max, N_ch)  — E5
    imp_total_p:  np.ndarray | None = None,   # (p_max,)       — E5
    ch_names:     list[str] | None = None,    # E5 channel names
    perclass_methods: dict | None = None,     # subset of results for F1 chart
    highlight_cls: list[int] | None = None,   # class indices to star in F1 chart
    best_result_key: str | None = None,       # key in results for CM
) -> None:
    """
    Save all tables and figures for one experiment.
    Call this at the end of each experiment's run() function.
    """
    eid = exp_id.lower()
    print(f"\n  ── Saving results for {exp_id.upper()} ──")

    # ── Find POCR result for reference lines ──────────────────────────
    pocr_res = next((v for k, v in results.items() if "POCR" in k), None)

    # ── Best SME key for confusion matrix ─────────────────────────────
    if best_result_key is None:
        sme_keys = [k for k in results if "SME" in k]
        best_result_key = (max(sme_keys, key=lambda k: results[k]["mean_acc"])
                           if sme_keys else list(results.keys())[0])
    best_res = results[best_result_key]

    # ── Extra baselines for ablation plot (non-POCR, non-SME) ─────────
    extra_bl = {k: v for k, v in results.items()
                if "POCR" not in k and "SME" not in k}
    extra_bl = dict(list(extra_bl.items())[:2])   # cap at 2 for readability

    # ═══════ TABLES ════════════════════════════════════════════════════
    save_results_csv(
        results, TABLES / f"{eid}_results.csv")
    save_results_latex(
        results, TABLES / f"{eid}_results.tex",
        caption=caption or f"Results for Experiment {exp_id.upper()}")
    save_ablation_csv(
        ablation, TABLES / f"{eid}_ablation.csv")
    if ablation_b:
        save_ablation_csv(
            ablation_b, TABLES / f"{eid}_ablation_vech.csv")
    if best_res.get("cm") is not None:
        save_confusion_matrix_csv(
            best_res["cm"], class_names, TABLES / f"{eid}_confusion_matrix.csv")

    # ═══════ FIGURES ═══════════════════════════════════════════════════

    # 1. Ablation curve
    if ablation_b:
        fig_dual_ablation(
            ablation, ablation_b,
            label_a=label_a, label_b=label_b,
            pocr_result=pocr_res, metric=primary_metric,
            title=f"{exp_id.upper()}: SME Ablation over Moment Order $p$",
            path=FIGURES / f"{eid}_ablation_curve.pdf")
        plt.close("all")
    else:
        fig_ablation_curve(
            ablation, pocr_result=pocr_res,
            metric=primary_metric,
            title=f"{exp_id.upper()}: SME Ablation over Moment Order $p$",
            extra_baselines=extra_bl if extra_bl else None,
            path=FIGURES / f"{eid}_ablation_curve.pdf")
        plt.close("all")

    # 2. Results bar chart
    fig_results_bar(
        results, metric=primary_metric,
        title=f"{exp_id.upper()}: Method Comparison",
        path=FIGURES / f"{eid}_results_bar.pdf")
    plt.close("all")

    # 3. Confusion matrix
    if best_res.get("cm") is not None:
        fig_confusion_matrix(
            best_res["cm"], class_names,
            title=f"{exp_id.upper()}: Confusion Matrix — {best_result_key}",
            path=FIGURES / f"{eid}_confusion_matrix.pdf")
        plt.close("all")

    # 4. Per-class F1
    pc_methods = perclass_methods or results
    if len(pc_methods) >= 2:
        fig_perclass_f1(
            pc_methods, class_names,
            title=f"{exp_id.upper()}: Per-Class F1",
            highlight=highlight_cls,
            path=FIGURES / f"{eid}_perclass_f1.pdf")
        plt.close("all")

    # 5. Significance heatmap
    if len(results) >= 2:
        fig_significance_heatmap(
            results,
            title=f"{exp_id.upper()}: Pairwise Wilcoxon",
            path=FIGURES / f"{eid}_significance_heatmap.pdf")
        plt.close("all")

    # 6. MP deviation  (E3, E4)
    if mp_delta is not None:
        p_max = mp_delta.shape[1]
        fig_mp_deviation(
            mp_delta, class_names, p_max,
            title=f"{exp_id.upper()}: Marchenko–Pastur Deviation $\\Delta_k$",
            path=FIGURES / f"{eid}_mp_deviation.pdf")
        plt.close("all")

    # 7. Kurtosis heatmap  (E4)
    if kurt_matrix is not None:
        p_vals = list(range(1, kurt_matrix.shape[1] + 1))
        fig_kurtosis_heatmap(
            kurt_matrix, class_names, p_vals,
            title=f"{exp_id.upper()}: Trajectory Kurtosis of SME-diagonal",
            path=FIGURES / f"{eid}_kurtosis_heatmap.pdf")
        plt.close("all")

    # 8. Feature importance  (E5)
    if diag_imp is not None and rsum_imp is not None:
        fig_feature_importance(
            diag_imp, rsum_imp,
            ch_names=ch_names or [f"ch{i}" for i in range(diag_imp.shape[1])],
            title=f"{exp_id.upper()}: GBT Feature Importance by Order and Channel",
            path=FIGURES / f"{eid}_feature_importance.pdf")
        plt.close("all")

    if imp_total_p is not None:
        p_vals = list(range(1, len(imp_total_p) + 1))
        fig_importance_by_order(
            imp_total_p, p_vals,
            title=f"{exp_id.upper()}: Total Importance per Moment Order",
            path=FIGURES / f"{eid}_importance_by_order.pdf")
        plt.close("all")

    print(f"  ✓ {exp_id.upper()} — all results saved to results/")
