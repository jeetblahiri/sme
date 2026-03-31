"""
exp5_har.py  ─  Experiment E5: Inertial Activity Recognition (UCI HAR)
=======================================================================
Dataset : Smartphone-Based Recognition of Human Activities, v2.1
          archive.ics.uci.edu (id=341)  — CC BY 4.0, no login required
          uci_har/RawData/acc_expXX_userYY.txt  (3-axis accelerometer)
          uci_har/RawData/gyro_expXX_userYY.txt (3-axis gyroscope)
          uci_har/RawData/labels.txt  [exp_id, user_id, act_id, start, end]

Signal  : 6-channel IMU (acc_x/y/z + gyro_x/y/z), 50 Hz
Window  : 128 samples (2.56 s), 50% overlap  → 6×6 covariance

Task    : 6-class basic activity recognition
          0=Walking  1=Walk-Up  2=Walk-Down  3=Sitting  4=Standing  5=Lying
          (Activity IDs 1-6; IDs 7-12 are postural transitions → excluded)

N = 6  →  6×6 covariance per window; POCR gives 5 angular coordinates.
SME     : sme_diagonal(p=1..5) ‖ sme_rowsum(p=1..5) = 60 features.
CV      : LOSO-CV (30 folds, one subject held out each time).

Key prediction: GBT feature importances should show
  p=1 dominates for Sit/Stand/Lying (gravity-driven variance),
  p=2 dominates for Walking variants (inter-axis energy exchange).

Baselines:
  · Per-channel statistics (mean, std, skew, kurtosis) = 24 features
  · POCR/CRA: 5 angles + BD(θ₅) = 6 features
  · UCI provided 561-dim engineered features (upper-bound, train/test split)

Ablation  : p ∈ {1, 2, 3, 4, 5}
Statistics: Wilcoxon LOSO-CV fold accuracy, confusion matrix, per-class F1.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, sme_rowsum, pocr_features,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table, print_significance_table,
    print_perclass_f1, wilcoxon_paired, bootstrap_ci, latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR   = _HERE   # scripts and data are in the same directory
HAR_DIR    = BASE_DIR / "uci_har"
RAW_DIR    = HAR_DIR / "RawData"
TRAIN_DIR  = HAR_DIR / "Train"
TEST_DIR   = HAR_DIR / "Test"

# ── constants ──────────────────────────────────────────────────────────
_SFREQ    = 50
_WIN_LEN  = 128    # 2.56 s
_WIN_HOP  = 64     # 50% overlap

# Activity labels (1-indexed in dataset → 0-indexed internally)
_ACT_NAMES = ["Walking", "Walk-Up", "Walk-Down",
               "Sitting", "Standing", "Lying"]
# Activities 1-6 are basic; 7-12 are transitions (excluded)
_BASIC_ACTS = set(range(1, 7))

# Feature-name helpers for importance analysis
_CH_NAMES = ["Ax", "Ay", "Az", "Gx", "Gy", "Gz"]


# ── loader ─────────────────────────────────────────────────────────────

def _load_txt(path: Path) -> np.ndarray:
    """Load whitespace-delimited text file → float32 array."""
    return np.loadtxt(str(path), dtype=np.float32)


def load_uci_har(data_dir: Path = RAW_DIR,
                 win: int = _WIN_LEN,
                 hop: int = _WIN_HOP):
    """
    Build sliding-window dataset from raw IMU signals.

    Returns
    -------
    windows  : (n, win, 6) float32
    labels   : (n,) int32   — 0-indexed activity
    subjects : (n,) int32   — user ID 1-30
    """
    labels_path = data_dir / "labels.txt"
    if not labels_path.exists():
        raise FileNotFoundError(f"labels.txt not found in {data_dir}")

    label_data = np.loadtxt(str(labels_path), dtype=int)
    # cols: [exp_id, user_id, act_id, start_sample (1-indexed), end_sample]

    windows, labels_out, subjects_out = [], [], []

    # Cache raw signals per (exp, user) to avoid reloading
    _cache: dict[tuple, np.ndarray] = {}

    for row in label_data:
        exp_id, user_id, act_id, start, end = row
        if act_id not in _BASIC_ACTS:
            continue               # skip postural transitions

        key = (exp_id, user_id)
        if key not in _cache:
            af = data_dir / f"acc_exp{exp_id:02d}_user{user_id:02d}.txt"
            gf = data_dir / f"gyro_exp{exp_id:02d}_user{user_id:02d}.txt"
            if not af.exists() or not gf.exists():
                continue
            try:
                acc  = _load_txt(af)   # (T, 3)
                gyro = _load_txt(gf)   # (T, 3)
            except Exception:
                continue
            _cache[key] = np.hstack([acc, gyro])   # (T, 6)

        sig = _cache[key]
        # Convert from 1-indexed sample range to 0-indexed slice
        seg = sig[start - 1 : end, :]
        n   = seg.shape[0]

        for s in range(0, n - win + 1, hop):
            windows.append(seg[s : s + win])
            labels_out.append(act_id - 1)      # 0-indexed
            subjects_out.append(user_id)

    if not windows:
        raise RuntimeError("No windows created — check labels.txt format and raw signal files.")

    return (np.array(windows,     dtype=np.float32),
            np.array(labels_out,  dtype=np.int32),
            np.array(subjects_out,dtype=np.int32))


# ── feature extractors ─────────────────────────────────────────────────

def feat_sme_combined(windows: np.ndarray, p: int) -> np.ndarray:
    """
    SME diagonal ‖ SME row-sum per 128-sample window.
    Returns (n, 2·6·p) = (n, 12p).
    """
    return np.array([
        np.concatenate([
            sme_diagonal(compute_cov(w), p_max=p),
            sme_rowsum(  compute_cov(w), p_max=p),
        ])
        for w in windows
    ], dtype=np.float32)


def feat_pocr(windows: np.ndarray) -> np.ndarray:
    """POCR for 6×6: 5 angles + BD(θ₅) = 6 features."""
    return np.array([pocr_features(compute_cov(w))
                     for w in windows], dtype=np.float32)


def feat_stats(windows: np.ndarray) -> np.ndarray:
    """Per-channel mean, std, skewness, excess-kurtosis → 6×4 = 24 features."""
    out = np.zeros((len(windows), 24), dtype=np.float32)
    for i, w in enumerate(windows):
        for ch in range(6):
            x = w[:, ch]
            out[i, ch*4 : (ch+1)*4] = [
                x.mean(), x.std(),
                float(sp_stats.skew(x)),
                float(sp_stats.kurtosis(x, fisher=True)),
            ]
    return out


def feat_uci_engineered() -> tuple[np.ndarray, np.ndarray,
                                   np.ndarray, np.ndarray]:
    """
    Load the pre-computed 561-dimensional feature vectors provided by UCI.
    Used as an upper-bound baseline (fixed train/test split, not CV).
    """
    Xtr = _load_txt(TRAIN_DIR / "X_train.txt")
    ytr = np.loadtxt(str(TRAIN_DIR / "y_train.txt"), dtype=int) - 1
    Xte = _load_txt(TEST_DIR  / "X_test.txt")
    yte = np.loadtxt(str(TEST_DIR  / "y_test.txt"),  dtype=int) - 1
    return Xtr, ytr, Xte, yte


# ── feature importance analysis ────────────────────────────────────────

def feature_importance_analysis(windows: np.ndarray, labels: np.ndarray,
                                 p_max: int = 5) -> None:
    """
    Train a GBT on the full dataset and decompose importances by (order p, axis j).
    Key prediction: p=1 dominates for Sit/Stand/Lying; p=2 for Walking variants.
    """
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    from sklearn.preprocessing import StandardScaler

    print("\n  ── Feature Importance Analysis (GBT on full dataset) ──")
    X   = feat_sme_combined(windows, p_max).astype(np.float64)
    X_s = StandardScaler().fit_transform(X)
    gbt = GBC(n_estimators=300, max_depth=5, learning_rate=0.1,
              random_state=42, subsample=0.8)
    gbt.fit(X_s, labels)

    N   = 6
    imp = gbt.feature_importances_   # (12p,)
    # First 6p = diag; next 6p = rowsum
    diag_imp  = imp[: N * p_max].reshape(p_max, N)
    rsum_imp  = imp[N * p_max :].reshape(p_max, N)
    total_per_p = (diag_imp.sum(axis=1) + rsum_imp.sum(axis=1))
    total_per_p /= total_per_p.sum()

    hline("─")
    print("  SME feature importances by moment order p  (diagonal projection):")
    print(f"  {'p':<4} " + "  ".join(f"{c:>6}" for c in _CH_NAMES))
    hline("─")
    for k in range(p_max):
        row = "  ".join(f"{v:>6.4f}" for v in diag_imp[k])
        print(f"  p={k+1:<3} {row}")
    hline("─")
    print("  Total importance per order p (diagonal + row-sum):")
    for k, v in enumerate(total_per_p):
        bar = "█" * int(v * 50)
        print(f"    p={k+1}: {v:.4f}  {bar}")

    # Per-activity dominant order
    print("\n  Per-activity dominant feature order:")
    from sklearn.metrics import accuracy_score
    for act, an in enumerate(_ACTIVITY_NAMES_E5):
        mask    = labels == act
        if mask.sum() < 10: continue
        scores  = []
        for k in range(p_max):
            # Use just that order's diag features
            Xk = X_s[:, k*N : (k+1)*N]
            from sklearn.svm import SVC
            from sklearn.model_selection import cross_val_score
            s = cross_val_score(SVC(C=5, kernel="rbf", gamma="scale"),
                                Xk, (labels == act).astype(int),
                                cv=3, scoring="f1").mean()
            scores.append(s)
        dom_p = np.argmax(scores) + 1
        print(f"    {an:<12}: dominant order = p={dom_p}  "
              f"(F1 per order: {[f'{s:.3f}' for s in scores]})")


_ACTIVITY_NAMES_E5 = _ACT_NAMES   # alias


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(p_values: list[int] | None = None) -> dict:

    if p_values is None:
        p_values = [1, 2, 3, 4, 5]

    hline()
    print("  EXPERIMENT E5 ─ Inertial Activity Recognition (UCI HAR, v2.1)")
    hline()

    if not RAW_DIR.exists():
        print(f"  ERROR: {RAW_DIR} not found."); return {}

    print("  Loading raw IMU signals and building windows …")
    windows, labels, subjects = load_uci_har(RAW_DIR)

    print(f"  Windows   : {len(labels)}")
    print(f"  Subjects  : {len(np.unique(subjects))}")
    for i, an in enumerate(_ACT_NAMES):
        print(f"    {an}: {(labels==i).sum()}")

    # ── Classifier ─────────────────────────────────────────────────────
    from sklearn.ensemble import GradientBoostingClassifier as GBC
    def clf():
        return make_pipeline(GBC, n_estimators=200, max_depth=5,
                             learning_rate=0.1, random_state=42,
                             subsample=0.8)

    # ── SME combined ablation (LOSO-CV) ─────────────────────────────────
    print("\n  ── SME Combined (diag + row-sum) Ablation — LOSO-CV ──")
    abl = ablation_over_p(
        feat_fn=lambda p: feat_sme_combined(windows, p),
        y=labels, groups=subjects, p_values=p_values,
        clf_factory=clf, strategy="loso",
        label="SME-comb",
    )
    print_ablation_table(abl, "E5 SME-combined ablation (LOSO-CV, 30 folds)")

    best_p = max(abl, key=lambda p: abl[p]["mean_acc"])
    print(f"\n  ★ Best order p={best_p}  "
          f"acc={abl[best_p]['mean_acc']:.3f}  "
          f"F1={abl[best_p]['mean_f1']:.3f}")

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    results = {f"SME-combined p={best_p}": abl[best_p]}

    print("  [POCR/CRA] ", end="", flush=True)
    X_pocr = feat_pocr(windows)
    r_pocr = cross_validate(clf(), X_pocr, labels,
                             groups=subjects, strategy="loso")
    results["POCR/CRA"] = r_pocr
    print(f"acc={r_pocr['mean_acc']:.3f}  F1={r_pocr['mean_f1']:.3f}")

    print("  [Per-ch statistics] ", end="", flush=True)
    X_stat = feat_stats(windows)
    r_stat = cross_validate(clf(), X_stat, labels,
                             groups=subjects, strategy="loso")
    results["Per-ch stats (24d)"] = r_stat
    print(f"acc={r_stat['mean_acc']:.3f}  F1={r_stat['mean_f1']:.3f}")

    # UCI engineered features — fixed split (not LOSO), reported as upper bound
    if (TRAIN_DIR / "X_train.txt").exists():
        print("  [UCI 561-dim engineered] ", end="", flush=True)
        try:
            from sklearn.svm import SVC
            from sklearn.preprocessing import StandardScaler
            from sklearn.metrics import accuracy_score, f1_score as _f1
            Xtr, ytr, Xte, yte = feat_uci_engineered()
            sc = StandardScaler()
            Xtr_s = sc.fit_transform(Xtr)
            Xte_s = sc.transform(Xte)
            svm   = SVC(C=10, kernel="rbf", gamma="scale")
            svm.fit(Xtr_s, ytr)
            yp_uci = svm.predict(Xte_s)
            acc_uci = accuracy_score(yte, yp_uci)
            f1_uci  = _f1(yte, yp_uci, average="macro")
            results["UCI 561-feat (upper bound)"] = dict(
                mean_acc=acc_uci, std_acc=0.0,
                mean_f1=f1_uci,  std_f1=0.0,
                fold_acc=np.array([acc_uci]),
                fold_f1=np.array([f1_uci]),
                y_true=yte, y_pred=yp_uci,
            )
            print(f"acc={acc_uci:.3f}  F1={f1_uci:.3f}  (fixed split)")
        except Exception as e:
            print(f"(failed: {e})")

    # ── Summary ─────────────────────────────────────────────────────────
    print_results_table(results, "E5 Final Results  (LOSO-CV, 30 folds)")
    print_significance_table(results, "E5 Pairwise Wilcoxon (LOSO folds)")

    print(f"\n  Per-class F1 — SME-combined(p={best_p}) vs POCR/CRA:")
    print_perclass_f1(abl[best_p], _ACT_NAMES, r_pocr,
                      name_a=f"SME p={best_p}", name_b="POCR/CRA")

    # ── Confusion matrix ────────────────────────────────────────────────
    cm = abl[best_p]["cm"]
    print(f"\n  Confusion matrix — SME-combined(p={best_p})  [row=true, col=pred]:")
    print("  " + "  ".join(f"{n[:4]:>9}" for n in _ACT_NAMES))
    for i, row in enumerate(cm):
        print(f"  {_ACT_NAMES[i][:4]:<9}" + "  ".join(f"{v:>9}" for v in row))

    # ── Feature importance ──────────────────────────────────────────────
    feature_importance_analysis(windows, labels, p_max=max(p_values))

    print("\n  LaTeX table:")
    print(latex_table(results, "E5: Inertial Activity Recognition (UCI HAR v2.1)"))

    # ── Compute GBT feature importances for saving ───────────────────────
    diag_imp_mat = rsum_imp_mat = total_imp = None
    try:
        from sklearn.ensemble import GradientBoostingClassifier as _GBC
        from sklearn.preprocessing import StandardScaler as _SS
        _p_max = max(p_values)
        _N     = 6
        _X     = feat_sme_combined(windows, _p_max).astype(float)
        _Xs    = _SS().fit_transform(_X)
        _gbt   = _GBC(n_estimators=300, max_depth=5, learning_rate=0.1,
                      random_state=42, subsample=0.8)
        _gbt.fit(_Xs, labels)
        _imp   = _gbt.feature_importances_
        diag_imp_mat = _imp[: _N * _p_max].reshape(_p_max, _N)
        rsum_imp_mat = _imp[_N * _p_max :].reshape(_p_max, _N)
        total_imp    = (diag_imp_mat.sum(axis=1) + rsum_imp_mat.sum(axis=1))
        total_imp   /= total_imp.sum()
    except Exception as e:
        print(f"  (feature importance computation failed: {e})")

    # ── Save results & figures ───────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id="e5",
            results=results,
            ablation=abl,
            class_names=_ACT_NAMES,
            primary_metric="acc",
            caption="E5: Inertial Activity Recognition (UCI HAR v2.1). "
                    "LOSO-CV (30 folds). Gradient Boosted Trees classifier.",
            best_result_key=f"SME-combined p={best_p}",
            perclass_methods={
                f"SME p={best_p}":      abl[best_p],
                "POCR/CRA":             results["POCR/CRA"],
                "Per-ch stats (24d)":   results["Per-ch stats (24d)"],
            },
            diag_imp=diag_imp_mat,
            rsum_imp=rsum_imp_mat,
            imp_total_p=total_imp,
            ch_names=_CH_NAMES,
            highlight_cls=[0, 1, 2],   # walking variants
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation": abl, "results": results}


if __name__ == "__main__":
    run()
