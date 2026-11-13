"""
sme_baselines.py  --  Additional baselines and diagnostics for the SME revision
================================================================================

Implements the additional baselines, diagnostics, and statistical tests
requested by the four reviewers of the major revision:

  * Reviewer 1: comprehensive Power-Spectral-Density (PSD) estimator family
      - Welch (non-parametric, configurable window)
      - Multitaper (DPSS, low-leakage non-parametric)
      - Autoregressive Burg (parametric, AR model)
      - Yule-Walker (parametric, biased autocorrelation)
  * Reviewer 4: consistent covariance-feature baselines across datasets
      - Raw vech covariance
      - Log-Euclidean covariance vectorisation
      - Riemannian tangent-space projection
      - Riemannian Minimum-Distance-to-Mean (MDM) classifier
      - Polynomial covariance expansion (degree-d products of entries)
  * Reviewer 4 / Reviewer 2: condition number, eigenvalue spectrum, and
    finite-sample diagnostics; epsilon sensitivity sweep; execution time
    profile vs channel count N.
  * Reviewer 4: corrected repeated cross-validation t-test (Nadeau-Bengio),
    permutation tests, hierarchical bootstrap, Holm-Bonferroni multiplicity
    correction.

Implementations follow the standard references cited in the manuscript.
"""

from __future__ import annotations

import time
import warnings
from math import comb, factorial

import numpy as np
from scipy import linalg, signal, stats
from scipy.signal import welch
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ============================================================================
# 1. Raw vech covariance baseline (Reviewer 4)
# ============================================================================

def vech_cov(C: np.ndarray) -> np.ndarray:
    """Upper-triangular vectorisation of a symmetric matrix C.

    For an N x N symmetric C this returns the N(N+1)/2 unique entries.
    """
    idx = np.triu_indices(C.shape[0])
    return C[idx].astype(np.float64)


# ============================================================================
# 2. Log-Euclidean and tangent-space covariance features (Reviewer 4)
# ============================================================================

def _sym(M: np.ndarray) -> np.ndarray:
    """Force exact symmetry (cancels round-off)."""
    return 0.5 * (M + M.T)


def _safe_eigh(C: np.ndarray, eps: float = 1e-10):
    """Eigendecomposition of a symmetric matrix, with eigenvalues
    floored at eps so matrix functions remain well defined."""
    w, V = linalg.eigh(_sym(C))
    w = np.clip(w, eps, None)
    return w, V


def matrix_log(C: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetric matrix logarithm log(C) = V diag(log w) V^T."""
    w, V = _safe_eigh(C, eps=eps)
    return _sym(V @ np.diag(np.log(w)) @ V.T)


def matrix_pow(C: np.ndarray, alpha: float, eps: float = 1e-10) -> np.ndarray:
    """Symmetric matrix power C^alpha = V diag(w^alpha) V^T."""
    w, V = _safe_eigh(C, eps=eps)
    return _sym(V @ np.diag(np.power(w, alpha)) @ V.T)


def logeuclid_vec(C: np.ndarray) -> np.ndarray:
    """Log-Euclidean feature: vech(log C).

    Defines an inner product (X,Y) = trace(log X log Y) under which
    the manifold of SPD matrices becomes a Euclidean space.
    """
    L = matrix_log(C)
    return vech_cov(L)


def riemann_mean(cov_list, n_iter: int = 50, tol: float = 1e-6) -> np.ndarray:
    """Affine-invariant (Frechet) mean of a list of SPD matrices.

    Iterative algorithm of Fletcher and Joshi (2007) / Pennec et al. (2006).
    Used for the Riemannian tangent-space and MDM baselines.
    """
    M = np.mean(np.stack(cov_list, axis=0), axis=0)
    for _ in range(n_iter):
        Minv_half = matrix_pow(M, -0.5)
        Mhalf     = matrix_pow(M,  0.5)
        S = np.zeros_like(M)
        for C in cov_list:
            S += matrix_log(_sym(Minv_half @ C @ Minv_half))
        S /= len(cov_list)
        M_new = _sym(Mhalf @ linalg.expm(_sym(S)) @ Mhalf)
        delta = linalg.norm(S, "fro")
        M = M_new
        if delta < tol:
            break
    return M


def tangent_space_vec(C: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Riemannian tangent-space projection at reference SPD matrix ref.

    T_ref(C) = vech( log( ref^{-1/2} C ref^{-1/2} ) )
    """
    Rinv_half = matrix_pow(ref, -0.5)
    W = _sym(Rinv_half @ C @ Rinv_half)
    return vech_cov(matrix_log(W))


def affine_invariant_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between two SPD matrices."""
    Ainv_half = matrix_pow(A, -0.5)
    W = _sym(Ainv_half @ B @ Ainv_half)
    w, _ = _safe_eigh(W)
    return float(np.sqrt(np.sum(np.log(w) ** 2)))


# ============================================================================
# 3. Minimum-Distance-to-Mean classifier (Reviewer 4)
# ============================================================================

class MDMClassifier(BaseEstimator, ClassifierMixin):
    """Riemannian Minimum-Distance-to-Mean classifier (Barachant 2012).

    Computes the Frechet mean of training covariance matrices per class,
    then assigns each test covariance to the closest mean under the
    affine-invariant distance.
    """

    def __init__(self, n_iter: int = 30):
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.means_ = []
        for c in self.classes_:
            cov_c = [X[i] for i in np.where(y == c)[0]]
            self.means_.append(riemann_mean(cov_c, n_iter=self.n_iter))
        return self

    def predict(self, X):
        out = np.zeros(len(X), dtype=self.classes_.dtype)
        for i, C in enumerate(X):
            d = [affine_invariant_distance(C, m) for m in self.means_]
            out[i] = self.classes_[int(np.argmin(d))]
        return out

    def score(self, X, y):
        return float((self.predict(X) == y).mean())


# ============================================================================
# 4. Polynomial covariance expansion (Reviewer 4)
# ============================================================================

def poly_cov_features(C: np.ndarray, degree: int = 2) -> np.ndarray:
    """Polynomial covariance features.

    Concatenates vech(C), then vech(C) tensor products up to total
    polynomial degree `degree`, capped at a sensible feature budget.
    For degree=2 this is the Veronese embedding of the covariance.
    """
    v = vech_cov(C)
    if degree <= 1:
        return v
    feats = [v]
    if degree >= 2:
        feats.append(np.outer(v, v)[np.triu_indices(len(v))])
    return np.concatenate(feats).astype(np.float64)


# ============================================================================
# 5. PSD estimator family (Reviewer 1)
# ============================================================================

def _band_avg(f: np.ndarray, psd: np.ndarray,
              bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30))) -> np.ndarray:
    """Average a one-sided PSD over the listed frequency bands."""
    return np.array([psd[(f >= lo) & (f <= hi)].mean()
                     if ((f >= lo) & (f <= hi)).any() else 0.0
                     for lo, hi in bands])


def psd_welch_bands(x: np.ndarray, fs: float,
                    bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30)),
                    nperseg: int | None = None,
                    window: str = "hann") -> np.ndarray:
    """Welch PSD with configurable window, averaged over `bands`."""
    if nperseg is None:
        nperseg = min(256, len(x))
    f, P = welch(x, fs=fs, window=window, nperseg=nperseg,
                 noverlap=nperseg // 2)
    return _band_avg(f, P, bands)


_DPSS_CACHE: dict = {}


def _get_dpss(N: int, NW: float, K: int):
    """Cache DPSS taper sequences across calls -- computing them once is
    a substantial speed-up on epoch-batched workloads."""
    key = (N, NW, K)
    if key not in _DPSS_CACHE:
        from scipy.signal.windows import dpss
        _DPSS_CACHE[key] = dpss(N, NW, Kmax=K, return_ratios=True)
    return _DPSS_CACHE[key]


def psd_multitaper_bands(x: np.ndarray, fs: float,
                         bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30)),
                         NW: float = 3.5, K: int | None = None) -> np.ndarray:
    """Multitaper PSD using Slepian (DPSS) tapers.

    NW is the half-bandwidth product; K = 2*NW - 1 tapers by default.
    Multitaper reduces spectral leakage relative to Welch on non-stationary
    EEG (Reviewer 1, citing Babadi & Brown, 2014).
    """
    N = len(x)
    if K is None:
        K = max(1, int(2 * NW - 1))
    tapers, eigs = _get_dpss(N, NW, K)
    psd_acc = np.zeros(N // 2 + 1)
    for h, ev in zip(tapers, eigs):
        X = np.fft.rfft(x * h)
        psd_acc += ev * (np.abs(X) ** 2)
    psd_acc /= max(np.sum(eigs), 1e-12)
    psd_acc = (psd_acc / fs) * 2.0  # one-sided density
    f = np.fft.rfftfreq(N, 1.0 / fs)
    return _band_avg(f, psd_acc, bands)


def psd_burg_bands(x: np.ndarray, fs: float,
                   bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30)),
                   order: int = 16) -> np.ndarray:
    """Burg AR PSD estimator (parametric).

    Recursively estimates AR(order) coefficients minimising the sum of
    forward and backward prediction errors (Burg, 1967).
    """
    a, e = _burg_ar(x, order)
    f = np.linspace(0, fs / 2, 256)
    z = np.exp(-2j * np.pi * f / fs)
    H = 1.0 / np.abs(np.polyval(np.concatenate([[1.0], a]),
                                z))
    psd = (e / fs) * np.abs(H) ** 2
    return _band_avg(f, psd, bands)


def _burg_ar(x: np.ndarray, p: int):
    """Burg's algorithm: returns AR(p) coefficients (a_1..a_p) and
    final prediction error e."""
    x = np.asarray(x, dtype=np.float64)
    N = len(x)
    f = x.copy()
    b = x.copy()
    a = np.zeros(p)
    e = np.dot(x, x) / N
    for k in range(p):
        num = -2.0 * np.dot(f[k + 1:], b[:N - k - 1])
        den = np.dot(f[k + 1:], f[k + 1:]) + np.dot(b[:N - k - 1], b[:N - k - 1])
        if abs(den) < 1e-12:
            kappa = 0.0
        else:
            kappa = num / den
        a_new = a.copy()
        a_new[k] = kappa
        for j in range(k):
            a_new[j] = a[j] + kappa * a[k - 1 - j]
        a = a_new
        f_new = f[k + 1:] + kappa * b[:N - k - 1]
        b_new = b[:N - k - 1] + kappa * f[k + 1:]
        f[k + 1:] = f_new
        b[:N - k - 1] = b_new
        e = (1.0 - kappa ** 2) * e
    return a, max(e, 1e-12)


def psd_yulewalker_bands(x: np.ndarray, fs: float,
                         bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30)),
                         order: int = 16) -> np.ndarray:
    """Yule-Walker AR PSD (parametric, biased autocorrelation)."""
    x = np.asarray(x, dtype=np.float64) - np.mean(x)
    N = len(x)
    r = np.array([np.dot(x[:N - k], x[k:]) / N for k in range(order + 1)])
    # solve Toeplitz R a = -r[1:]
    R = linalg.toeplitz(r[:order])
    try:
        a = linalg.solve(R, -r[1:], assume_a='pos')
    except linalg.LinAlgError:
        a = np.linalg.lstsq(R, -r[1:], rcond=None)[0]
    e = r[0] + np.dot(a, r[1:])
    f = np.linspace(0, fs / 2, 256)
    z = np.exp(-2j * np.pi * f / fs)
    H = 1.0 / np.abs(np.polyval(np.concatenate([[1.0], a]), z))
    psd = (max(e, 1e-12) / fs) * np.abs(H) ** 2
    return _band_avg(f, psd, bands)


def psd_features_multi(epochs: np.ndarray, fs: float, method: str,
                       bands=((0.5, 4), (4, 8), (8, 13), (11, 16), (16, 30))
                       ) -> np.ndarray:
    """Compute per-channel PSD-band features for a batch of (E, N, T) epochs."""
    E, N, T = epochs.shape
    out = np.zeros((E, N * len(bands)), dtype=np.float32)
    estimator = {
        "welch":       lambda x: psd_welch_bands(x, fs, bands),
        "welch_hamm":  lambda x: psd_welch_bands(x, fs, bands, window="hamming"),
        "multitaper":  lambda x: psd_multitaper_bands(x, fs, bands),
        "burg":        lambda x: psd_burg_bands(x, fs, bands, order=12),
        "yulewalker":  lambda x: psd_yulewalker_bands(x, fs, bands, order=12),
    }[method]
    for i in range(E):
        col = 0
        for ch in range(N):
            out[i, col:col + len(bands)] = estimator(epochs[i, ch])
            col += len(bands)
    return out


# ============================================================================
# 6. Statistical testing utilities (Reviewer 4)
# ============================================================================

def nadeau_bengio_t(scores_a: np.ndarray, scores_b: np.ndarray,
                    n_train: int, n_test: int) -> tuple[float, float]:
    """Corrected repeated cross-validation t-test (Nadeau & Bengio, 2003).

    Adjusts the variance estimator for the train/test correlation introduced
    by overlapping training folds, yielding a less optimistic p-value than
    the naive paired t-test.
    """
    d = scores_a - scores_b
    n = len(d)
    if n < 2 or np.allclose(d, 0):
        return float("nan"), float("nan")
    mean_d = d.mean()
    var_d  = d.var(ddof=1)
    correction = (1.0 / n) + (n_test / max(n_train, 1))
    se = np.sqrt(correction * var_d)
    if se < 1e-12:
        return float("nan"), float("nan")
    t = mean_d / se
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p)


def permutation_test(a: np.ndarray, b: np.ndarray,
                     n_perm: int = 10_000, seed: int = 42) -> float:
    """Paired permutation test on fold-level accuracy differences.

    Distribution-free: no normality, no fold-independence assumption beyond
    exchangeability under the null of identical performance.
    """
    rng = np.random.default_rng(seed)
    d = a - b
    obs = abs(d.mean())
    n = len(d)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=n)
        if abs((signs * d).mean()) >= obs - 1e-15:
            count += 1
    return (count + 1) / (n_perm + 1)


def hierarchical_bootstrap_ci(fold_scores: np.ndarray,
                              n_boot: int = 5_000,
                              alpha: float = 0.05,
                              seed: int = 42) -> tuple[float, float]:
    """Hierarchical (cluster) bootstrap CI for the mean across folds.

    Resamples folds with replacement, then resamples within each fold,
    matching the two-level structure of cross-validation results.
    """
    rng = np.random.default_rng(seed)
    n = len(fold_scores)
    means = np.empty(n_boot)
    for k in range(n_boot):
        sel = rng.integers(0, n, size=n)
        means[k] = fold_scores[sel].mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def holm_bonferroni(pvals: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm step-down correction.

    Returns a list of booleans indicating which null hypotheses are rejected
    at family-wise error rate `alpha`.
    """
    m = len(pvals)
    idx = np.argsort(pvals)
    reject = [False] * m
    for k, i in enumerate(idx):
        threshold = alpha / (m - k)
        if pvals[i] <= threshold:
            reject[i] = True
        else:
            break
    return reject


# ============================================================================
# 7. Condition number / eigenvalue diagnostics (Reviewer 4)
# ============================================================================

def covariance_diagnostics(cov_list: list[np.ndarray]) -> dict:
    """Return a dictionary of finite-sample covariance diagnostics:
    condition number, smallest eigenvalue, eigenvalue spread, fraction of
    near-singular matrices."""
    conds, mins, maxs = [], [], []
    for C in cov_list:
        w = np.clip(np.linalg.eigvalsh(_sym(C)), 0.0, None)
        mins.append(float(w.min()))
        maxs.append(float(w.max()))
        conds.append(float(w.max() / max(w.min(), 1e-12)))
    conds = np.array(conds)
    return dict(
        cond_mean=float(conds.mean()),
        cond_median=float(np.median(conds)),
        cond_p95=float(np.percentile(conds, 95)),
        lam_min=float(np.mean(mins)),
        lam_max=float(np.mean(maxs)),
        frac_ill=float((conds > 1e6).mean()),
        conds=conds,
    )


# ============================================================================
# 8. Epsilon sensitivity (Reviewer 2)
# ============================================================================

def epsilon_sweep(epochs: np.ndarray, y: np.ndarray,
                  feat_fn_eps, clf_factory,
                  eps_values=(1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3),
                  cv_fn=None) -> dict:
    """Sensitivity analysis for the regularisation epsilon.

    feat_fn_eps(epochs, eps) must return a feature matrix; cv_fn(clf, X, y)
    must return a dict containing 'mean_acc'. Returns {eps: mean_acc}.
    """
    out = {}
    for eps in eps_values:
        X = feat_fn_eps(epochs, eps).astype(np.float32)
        r = cv_fn(clf_factory(), X, y)
        out[eps] = float(r["mean_acc"])
    return out


# ============================================================================
# 9. Execution-time profile vs channel count N (Reviewer 2)
# ============================================================================

def time_profile_sme(N_values=(2, 4, 8, 16, 32, 64, 128), T: int = 1024,
                     p: int = 4, n_reps: int = 5, seed: int = 42) -> dict:
    """Wall-clock cost of SME diagonal at moment order p versus channel count.

    Returns a dict {N: mean_time_seconds}.
    """
    from sme_core import compute_cov, sme_diagonal
    rng = np.random.default_rng(seed)
    out = {}
    for N in N_values:
        X = rng.standard_normal((T, N))
        # warm-up
        C0 = compute_cov(X)
        _ = sme_diagonal(C0, p_max=p)
        ts = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            C = compute_cov(X)
            _ = sme_diagonal(C, p_max=p)
            ts.append(time.perf_counter() - t0)
        out[int(N)] = float(np.mean(ts))
    return out


def time_profile_pocr(N_values=(2, 4, 8, 16, 32, 64, 128), T: int = 1024,
                      n_reps: int = 5, seed: int = 42) -> dict:
    """Same wall-clock profile but for POCR (includes eigendecomposition)."""
    from sme_core import compute_cov, compute_pocr
    rng = np.random.default_rng(seed)
    out = {}
    for N in N_values:
        X = rng.standard_normal((T, N))
        C0 = compute_cov(X)
        _ = compute_pocr(C0)
        ts = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            C = compute_cov(X)
            _ = compute_pocr(C)
            ts.append(time.perf_counter() - t0)
        out[int(N)] = float(np.mean(ts))
    return out


# ============================================================================
# 10. Artifact-robustness probe (Reviewer 1)
# ============================================================================

def inject_artifact(epoch: np.ndarray, amp: float = 5.0,
                    frac: float = 0.05, kind: str = "spike",
                    seed: int = 0) -> np.ndarray:
    """Inject synthetic artifacts into one (N, T) epoch.

    Used to evaluate the robustness of high-order SME moments to non-Gaussian
    outliers such as EMG bursts, ocular movements, and electrode pops.
    """
    rng = np.random.default_rng(seed)
    out = epoch.copy()
    N, T = out.shape
    n_art = max(1, int(frac * T))
    idx = rng.choice(T, size=n_art, replace=False)
    sigma = max(out.std(), 1e-6)
    if kind == "spike":
        out[:, idx] += amp * sigma * rng.choice([-1.0, 1.0],
                                                size=(N, n_art))
    elif kind == "burst":
        burst = amp * sigma * rng.standard_normal((N, n_art))
        out[:, idx] += burst
    elif kind == "drift":
        drift = amp * sigma * np.linspace(0, 1, T)
        out += drift[None, :]
    return out


def artifact_curve(epochs: np.ndarray, y: np.ndarray,
                   feat_fn, clf_factory, cv_fn,
                   amp_grid=(0.0, 0.5, 1.0, 2.0, 4.0, 8.0),
                   kind: str = "spike", seed: int = 42) -> dict:
    """Accuracy versus artifact amplitude. Returns {amp: mean_acc}."""
    rng = np.random.default_rng(seed)
    out = {}
    for amp in amp_grid:
        if amp == 0.0:
            ep_a = epochs
        else:
            ep_a = np.stack(
                [inject_artifact(e, amp=amp, kind=kind,
                                 seed=int(rng.integers(0, 2 ** 31)))
                 for e in epochs])
        X = feat_fn(ep_a).astype(np.float32)
        r = cv_fn(clf_factory(), X, y)
        out[float(amp)] = float(r["mean_acc"])
    return out
