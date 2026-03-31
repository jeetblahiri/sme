"""
sme_core.py  ─  Spectral Moment Embedding: Core Library
========================================================
Implements the complete SME feature extraction pipeline, the POCR/CRA
baseline (Panwar et al., IEEE TNSRE 2019), Marchenko–Pastur deviation
analysis, trajectory statistics, and all evaluation utilities used
across the five experiments.

All public functions are fully documented and unit-tested in
run_all_experiments.py::test_sme_core().

Authors: SME framework (2025)
"""

from __future__ import annotations

import warnings
from math import comb

import numpy as np
from scipy import stats
from scipy.linalg import eigh as sp_eigh
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score)
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  COVARIANCE
# ══════════════════════════════════════════════════════════════════════════════

def compute_cov(X: np.ndarray, reg: float = 1e-6) -> np.ndarray:
    """
    Regularised sample covariance of an (T × N) data matrix X.

    C = (X_c^T X_c) / (T-1) + reg·(tr(C)/N)·I

    The adaptive regularisation scales with signal energy so it is
    meaningful across datasets with very different amplitude ranges.
    """
    X_c = X - X.mean(axis=0, keepdims=True)
    T   = max(X_c.shape[0] - 1, 1)
    C   = (X_c.T @ X_c) / T
    N   = C.shape[0]
    scale = max(np.trace(C) / N, 1e-10)
    return C + reg * scale * np.eye(N)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  MATRIX POWER ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def _matrix_powers(C: np.ndarray, p_max: int) -> list[np.ndarray]:
    """
    Return [C, C², …, C^{p_max}] via repeated multiplication.
    O(N² · p_max) — no eigendecomposition.
    """
    out = []
    Ck  = C.copy()
    for k in range(p_max):
        if k > 0:
            Ck = Ck @ C
        out.append(Ck.copy())
    return out


# ══════════════════════════════════════════════════════════════════════════════
# 3.  SME FEATURE PROJECTIONS
# ══════════════════════════════════════════════════════════════════════════════

def sme_diagonal(C: np.ndarray, p_max: int = 4) -> np.ndarray:
    """
    Diagonal projection of moment matrices.

    sme_diag[k·N + j] = (C^{k+1})_{jj}
                       = Σ_path  product of covariances along all (k+1)-step
                         paths in the covariance graph that return to channel j.

    Shape: (p_max · N,)
    """
    N  = C.shape[0]
    out = np.empty(p_max * N, dtype=np.float64)
    for k, Ck in enumerate(_matrix_powers(C, p_max)):
        out[k * N : (k + 1) * N] = np.diag(Ck)
    return out


def sme_rowsum(C: np.ndarray, p_max: int = 4) -> np.ndarray:
    """
    Row-sum projection: P_row(C^k) = C^k · 1_N.

    Captures the total outgoing coupling from each channel at order k.
    Shape: (p_max · N,)
    """
    N    = C.shape[0]
    ones = np.ones(N, dtype=np.float64)
    out  = np.empty(p_max * N, dtype=np.float64)
    for k, Ck in enumerate(_matrix_powers(C, p_max)):
        out[k * N : (k + 1) * N] = Ck @ ones
    return out


def sme_vech(C: np.ndarray, p_max: int = 2) -> np.ndarray:
    """
    Upper-triangle (vech) projection — lossless for symmetric C^k.
    Shape: (p_max · N(N+1)/2,)
    """
    idx  = np.triu_indices(C.shape[0])
    return np.concatenate([Ck[idx] for Ck in _matrix_powers(C, p_max)])


def sme_trace(C: np.ndarray, p_max: int = 6) -> np.ndarray:
    """
    Normalised trace moments: Tr(C^k)/N for k = 1,…,p_max.
    These are the power sums of the empirical spectral distribution (ESD).
    Shape: (p_max,)
    """
    N = C.shape[0]
    return np.array([np.trace(Ck) / N
                     for Ck in _matrix_powers(C, p_max)], dtype=np.float64)


def sme_combined(C: np.ndarray, p_max: int = 4) -> np.ndarray:
    """Diagonal ∥ row-sum — used when N is small (E3, E4)."""
    return np.concatenate([sme_diagonal(C, p_max), sme_rowsum(C, p_max)])


# ══════════════════════════════════════════════════════════════════════════════
# 4.  POCR / CRA BASELINE   (Panwar et al. 2019, IEEE TNSRE)
# ══════════════════════════════════════════════════════════════════════════════

def compute_pocr(C: np.ndarray):
    """
    Positive-Orthant Characteristic Response vector.

    POCR_j = Σ_i  λ_i |v_{ij}|     (sign-invariant; all components ≥ 0)

    Theorem 3 (SME framework): POCR is a distorted, non-injective
    approximation to sme_diagonal(C, p_max=1) = diag(C).

    Returns
    -------
    pocr      : (N,)  cartesian vector in the positive orthant
    theta_deg : (N-1,) angular coordinates in degrees, each ∈ [0°, 90°]
    """
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, 0.0)

    # POCR_j = Σ_i λ_i |v_{ij}|
    pocr = (eigvals[np.newaxis, :] * np.abs(eigvecs)).sum(axis=1)

    N = pocr.shape[0]
    r = np.linalg.norm(pocr)
    if r < 1e-12:
        return pocr, np.zeros(N - 1)

    theta_deg = np.zeros(N - 1)
    for k in range(N - 1):
        tail = np.linalg.norm(pocr[k:])
        theta_deg[k] = (np.degrees(np.arccos(np.clip(pocr[k] / tail, -1.0, 1.0)))
                        if tail > 1e-12 else 0.0)
    return pocr, theta_deg


def bhattacharyya_distance_gaussian(data: np.ndarray,
                                    lo: float = -90.0,
                                    hi: float = 180.0,
                                    step: float = 0.1) -> float:
    """
    BD between the empirical distribution of `data` and an ideal Gaussian
    with the same mean and std (Equation 4, Panwar et al. 2019).
    """
    bins    = np.arange(lo, hi + step, step)
    centers = 0.5 * (bins[:-1] + bins[1:])

    counts, _ = np.histogram(data, bins=bins)
    if counts.sum() == 0:
        return 0.0
    p = counts / counts.sum()

    mu, sigma = float(np.mean(data)), max(float(np.std(data)), 1e-9)
    q = stats.norm.pdf(centers, mu, sigma) * step
    q_sum = q.sum()
    if q_sum < 1e-10:
        return 0.0
    q /= q_sum

    bc = float(np.sqrt(np.maximum(p, 0) * np.maximum(q, 0)).sum())
    return float(-np.log(max(bc, 1e-300)))


def pocr_features(C: np.ndarray) -> np.ndarray:
    """
    Full POCR feature vector for one covariance matrix:
      [θ_1, …, θ_{N-1}, BD(θ_{N-1})]
    Shape: (N,)
    """
    _, thetas = compute_pocr(C)
    bd = bhattacharyya_distance_gaussian(thetas)
    return np.append(thetas, bd).astype(np.float64)


def pocr_features_from_list(cov_list: list[np.ndarray]) -> np.ndarray:
    """
    POCR over a time-series of covariance matrices.
    Returns mean and std of each angular coordinate + BD of θ_{N-1}.
    """
    all_theta = np.array([compute_pocr(C)[1] for C in cov_list])  # (T, N-1)
    bd        = bhattacharyya_distance_gaussian(all_theta[:, -1])
    return np.concatenate([all_theta.mean(axis=0),
                           all_theta.std(axis=0),
                           [bd]]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  TRAJECTORY STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def trajectory_stats(feat_traj: np.ndarray,
                     include_bd: bool = True) -> np.ndarray:
    """
    Distributional statistics of a feature trajectory (T windows × D features).

    Returns [mean, var, skewness, kurtosis (,BD)] concatenated per feature.
    Shape: (5·D,) with BD or (4·D,) without.
    """
    if feat_traj.ndim == 1:
        feat_traj = feat_traj.reshape(-1, 1)
    T, D = feat_traj.shape

    if T < 4:
        n = 5 * D if include_bd else 4 * D
        return np.zeros(n, dtype=np.float64)

    mu = feat_traj.mean(axis=0)
    va = feat_traj.var(axis=0, ddof=min(1, T - 1))
    sk = stats.skew(feat_traj, axis=0)
    ku = stats.kurtosis(feat_traj, axis=0, fisher=True)

    parts = [mu, va, sk, ku]
    if include_bd:
        bd = np.array([bhattacharyya_distance_gaussian(feat_traj[:, d])
                       for d in range(D)])
        parts.append(bd)
    return np.concatenate(parts).astype(np.float64)


def sliding_covs(X: np.ndarray, win: int, hop: int,
                 reg: float = 1e-6) -> list[np.ndarray]:
    """Compute covariance matrices from sliding windows of (T × N) matrix X."""
    T = X.shape[0]
    return [compute_cov(X[s : s + win], reg=reg)
            for s in range(0, T - win + 1, hop)]


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MARCHENKO–PASTUR DEVIATION
# ══════════════════════════════════════════════════════════════════════════════

def _mp_moment(k: int, gamma: float) -> float:
    """
    k-th moment of the Marchenko–Pastur(γ) distribution via Narayana numbers.

    m_k = (1/k) Σ_{j=0}^{k-1}  C(k, j) · C(k, j+1) · γ^j

    Verification:
      k=1: m_1 = (1/1)·C(1,0)·C(1,1)·1        = 1           ✓
      k=2: m_2 = (1/2)·[C(2,0)·C(2,1) + C(2,1)·C(2,2)·γ]
               = (1/2)·[2 + 2γ] = 1 + γ         ✓
      k=3: m_3 = 1 + 3γ + γ²                     ✓

    Bug that was here: used comb(k-1, j) instead of comb(k, j+1).
    comb(k-1, j) gives the wrong Narayana coefficient, producing
    m_2 = (1+2γ)/2 ≈ 0.5 instead of 1+γ ≈ 1, so every Δ_k was
    computed against a wildly wrong baseline.
    """
    s = 0.0
    for j in range(k):
        s += comb(k, j) * comb(k, j + 1) * (gamma ** j)
    return s / k


def mp_deviation(C: np.ndarray, T: int, p_max: int = 6) -> np.ndarray:
    """
    Δ_k = Tr(C^k)/N − E_MP[λ^k]  for k = 1,…,p_max.

    Under the Marchenko–Pastur null (pure noise, N/T = γ < 1), Δ_k ≈ 0.
    Structured signal makes Δ_k significantly positive.
    """
    N     = C.shape[0]
    gamma = min(N / T, 0.99)
    mp    = np.array([_mp_moment(k, gamma) for k in range(1, p_max + 1)])
    return np.array([np.trace(Ck) / N - mp[k]
                     for k, Ck in enumerate(_matrix_powers(C, p_max))])


# ══════════════════════════════════════════════════════════════════════════════
# 7.  EVALUATION UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def make_pipeline(clf_cls, **kw) -> Pipeline:
    """StandardScaler → classifier Pipeline."""
    return Pipeline([("sc", StandardScaler()), ("clf", clf_cls(**kw))])


def cross_validate(clf, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray | None = None,
                   strategy: str = "skf",
                   n_splits: int = 10,
                   seed: int = 42) -> dict:
    """
    Run cross-validated evaluation and return per-fold and aggregate metrics.

    strategy : "skf"  → StratifiedKFold(n_splits)
               "loso" → LeaveOneGroupOut  (requires groups)
    """
    if strategy == "loso" and groups is not None:
        cv   = LeaveOneGroupOut()
        args = (X, y, groups)
    else:
        cv   = StratifiedKFold(n_splits=n_splits, shuffle=True,
                               random_state=seed)
        args = (X, y)

    fold_acc, fold_f1 = [], []
    y_true_all, y_pred_all = [], []

    for tr, te in cv.split(*args):
        clf.fit(X[tr], y[tr])
        yp = clf.predict(X[te])
        fold_acc.append(accuracy_score(y[te], yp))
        fold_f1.append(f1_score(y[te], yp, average="macro", zero_division=0))
        y_true_all.extend(y[te].tolist())
        y_pred_all.extend(yp.tolist())

    ya, yp = np.array(y_true_all), np.array(y_pred_all)
    return dict(
        fold_acc=np.array(fold_acc),
        fold_f1=np.array(fold_f1),
        mean_acc=float(np.mean(fold_acc)),
        std_acc=float(np.std(fold_acc)),
        mean_f1=float(np.mean(fold_f1)),
        std_f1=float(np.std(fold_f1)),
        y_true=ya, y_pred=yp,
        report=classification_report(ya, yp, zero_division=0),
        cm=confusion_matrix(ya, yp),
    )


def wilcoxon_paired(a: np.ndarray, b: np.ndarray,
                    name_a: str = "A", name_b: str = "B") -> dict:
    """Two-sided Wilcoxon signed-rank test on paired fold accuracies."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n < 5 or np.allclose(a, b):
        return dict(stat=np.nan, pval=np.nan, sig=False,
                    delta=float(a.mean() - b.mean()))
    try:
        stat, pval = stats.wilcoxon(a, b, alternative="two-sided")
    except Exception:
        stat, pval = np.nan, np.nan
    return dict(stat=float(stat), pval=float(pval),
                sig=pval < 0.05, delta=float(a.mean() - b.mean()))


def bootstrap_ci(scores: np.ndarray, n_boot: int = 2000,
                 alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """95 % bootstrap CI for the mean of fold-level accuracy scores."""
    rng   = np.random.default_rng(seed)
    boots = [rng.choice(scores, len(scores), replace=True).mean()
             for _ in range(n_boot)]
    return (float(np.percentile(boots, 100 * alpha / 2)),
            float(np.percentile(boots, 100 * (1 - alpha / 2))))


# ══════════════════════════════════════════════════════════════════════════════
# 8.  ABLATION RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def ablation_over_p(feat_fn,           # callable: p (int) → X (n_samples, d)
                    y: np.ndarray,
                    groups: np.ndarray | None,
                    p_values: list[int],
                    clf_factory,       # callable: () → sklearn estimator
                    strategy: str = "skf",
                    n_splits: int = 10,
                    label: str = "") -> dict[int, dict]:
    """
    Run cross-validation for each moment order p in p_values.
    Returns {p: result_dict}.
    """
    results = {}
    for p in p_values:
        tag = f"  [{label} p={p}]" if label else f"  [p={p}]"
        print(f"{tag} extracting features …", end=" ", flush=True)
        X = feat_fn(p).astype(np.float32)
        print(f"({X.shape[1]}d) evaluating …", end=" ", flush=True)
        res = cross_validate(clf_factory(), X, y,
                             groups=groups, strategy=strategy,
                             n_splits=n_splits)
        results[p] = res
        lo, hi = bootstrap_ci(res["fold_acc"])
        print(f"acc={res['mean_acc']:.3f}±{res['std_acc']:.3f} "
              f"F1={res['mean_f1']:.3f} CI=[{lo:.3f},{hi:.3f}]")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 9.  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

_W = 72

def hline(c="═"): print(c * _W)


def print_results_table(results: dict, title: str = "") -> None:
    hline()
    if title:
        print(f"  {title}")
        hline("─")
    print(f"  {'Method':<34} {'Acc':>14}   {'Macro-F1':>14}")
    hline("─")
    for name, r in results.items():
        a = f"{r['mean_acc']:.3f}±{r['std_acc']:.3f}"
        f = f"{r['mean_f1']:.3f}±{r['std_f1']:.3f}"
        print(f"  {str(name):<34} {a:>14}   {f:>14}")
    hline()


def print_ablation_table(abl: dict[int, dict], title: str = "") -> None:
    hline()
    if title:
        print(f"  {title}")
        hline("─")
    print(f"  {'p':>4}  {'Acc':>16}   {'Macro-F1':>16}   {'95% CI (Acc)':>18}")
    hline("─")
    for p, r in sorted(abl.items()):
        lo, hi = bootstrap_ci(r["fold_acc"])
        a = f"{r['mean_acc']:.3f}±{r['std_acc']:.3f}"
        f = f"{r['mean_f1']:.3f}±{r['std_f1']:.3f}"
        print(f"  {p:>4}  {a:>16}   {f:>16}   [{lo:.3f}, {hi:.3f}]")
    hline()


def print_significance_table(results: dict, title: str = "") -> None:
    """All-pairs Wilcoxon signed-rank p-value matrix."""
    methods = list(results.keys())
    n = len(methods)
    hline()
    if title:
        print(f"  {title}")
    print(f"  Pairwise Wilcoxon signed-rank (fold accuracy, two-sided):")
    hline("─")
    col_w = 14
    header = "  " + f"{'':20}" + "".join(f"{m[:col_w]:>{col_w}}" for m in methods)
    print(header)
    for ma in methods:
        row = f"  {ma[:20]:<20}"
        for mb in methods:
            if ma == mb:
                row += f"{'—':>{col_w}}"
            else:
                a = results[ma]["fold_acc"]
                b = results[mb]["fold_acc"]
                res = wilcoxon_paired(a, b)
                s = f"{res['pval']:.3f}{'*' if res['sig'] else ' '}"
                row += f"{s:>{col_w}}"
        print(row)
    hline()
    print("  (* p < 0.05)")


def print_perclass_f1(result: dict, class_names: list[str],
                      result_b: dict | None = None,
                      name_a: str = "SME",
                      name_b: str = "POCR") -> None:
    """Per-class F1 breakdown with optional comparison to a second method."""
    hline("─")
    if result_b:
        print(f"  {'Class':<14} {name_a:>10}  {name_b:>10}   Δ")
    else:
        print(f"  {'Class':<14} {name_a:>10}")
    hline("─")
    for i, cn in enumerate(class_names):
        f1a = f1_score(result["y_true"], result["y_pred"],
                       labels=[i], average="macro", zero_division=0)
        if result_b:
            f1b = f1_score(result_b["y_true"], result_b["y_pred"],
                           labels=[i], average="macro", zero_division=0)
            delta = f1a - f1b
            arrow = "↑" if delta > 0.01 else ("↓" if delta < -0.01 else "≈")
            print(f"  {cn:<14} {f1a:>10.3f}  {f1b:>10.3f}   {delta:+.3f} {arrow}")
        else:
            print(f"  {cn:<14} {f1a:>10.3f}")
    hline("─")


def latex_table(results: dict, caption: str = "") -> str:
    """Produce a LaTeX table string from results dict."""
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        rf"\caption{{{caption}}}",
        r"\begin{tabular}{lcc}", r"\toprule",
        r"Method & Accuracy (\%) & Macro-F1 \\ \midrule",
    ]
    for name, r in results.items():
        a = f"{r['mean_acc']*100:.1f}$\\pm${r['std_acc']*100:.1f}"
        f = f"{r['mean_f1']:.3f}$\\pm${r['std_f1']:.3f}"
        lines.append(f"{name} & {a} & {f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
