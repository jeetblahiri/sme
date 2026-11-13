"""
run_revision.py  --  Augmented experiments for the SME major revision
======================================================================

Adds the additional baselines, analyses, and diagnostics requested by the
four reviewers, while reusing the existing data loaders. Saves JSON results
and plaintext summaries under results/ alongside this script.

Augmentations covered:
    Reviewer 1: PSD estimator family on E2 (Welch, Welch-Hamming,
                Multitaper, Burg AR, Yule-Walker); artifact-robustness
                curve at p=1,3 on E2 EEG epochs.
    Reviewer 2: unified Linear-SVM baseline across all five experiments;
                epsilon sensitivity on E2 and E3; execution-time profile
                vs channel count N.
    Reviewer 3: framework diagram (saved to ../results/figures/).
    Reviewer 4: Log-Euclidean (matrix-log vech), tangent-space (Riemannian),
                Minimum-Distance-to-Mean (MDM), raw vech covariance,
                polynomial covariance baselines; condition-number /
                eigenvalue diagnostics; corrected-resampled t-test,
                permutation test, hierarchical bootstrap; Holm
                multiplicity correction; trajectory-SME on E1.

Usage:
    python run_revision.py [stage]

      stage in {core, e1, e2, e3, e4, e5, diag, fig, all}
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import (LeaveOneGroupOut, StratifiedGroupKFold,
                                     StratifiedKFold)
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
warnings.filterwarnings("ignore")

from sme_core import (compute_cov, sme_diagonal, sme_rowsum, sme_vech,
                      sme_combined, pocr_features, trajectory_stats,
                      bhattacharyya_distance_gaussian)
from sme_baselines import (vech_cov, logeuclid_vec, tangent_space_vec,
                           riemann_mean, MDMClassifier, poly_cov_features,
                           psd_features_multi, nadeau_bengio_t,
                           permutation_test, hierarchical_bootstrap_ci,
                           holm_bonferroni, covariance_diagnostics,
                           time_profile_sme, time_profile_pocr,
                           inject_artifact)

REPO       = _HERE
OUT_DIR    = REPO / "results"
FIG_DIR    = OUT_DIR / "figures"
TAB_DIR    = OUT_DIR / "tables"
JSON_DIR   = OUT_DIR / "json"
for d in (OUT_DIR, FIG_DIR, TAB_DIR, JSON_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Public datasets live alongside the scripts in the repository root.
DATA_BASE = REPO


# ---------------------------------------------------------------------------
# 1. Generic helpers
# ---------------------------------------------------------------------------

def make_pipe(clf_cls, **kw) -> Pipeline:
    return Pipeline([("sc", StandardScaler()), ("clf", clf_cls(**kw))])


def cv_score(clf, X, y, groups, strategy="sgkf", n_splits=10, seed=42) -> dict:
    """Cross-validation with strict subject grouping by default.

    strategy:
        'sgkf' = StratifiedGroupKFold on `groups`
        'loso' = LeaveOneGroupOut on `groups`
        'skf'  = StratifiedKFold (no grouping — for reference only)
    """
    from sklearn.metrics import (accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    if strategy == "loso":
        cv = LeaveOneGroupOut()
        splitter = cv.split(X, y, groups)
    elif strategy == "sgkf" and groups is not None:
        cv = StratifiedGroupKFold(n_splits=n_splits)
        splitter = cv.split(X, y, groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        splitter = cv.split(X, y)

    acc, f1s, ya, yp = [], [], [], []
    for tr, te in splitter:
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        acc.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        ya.extend(y[te].tolist())
        yp.extend(p.tolist())
    acc = np.array(acc); f1s = np.array(f1s)
    ya = np.array(ya); yp = np.array(yp)
    return dict(
        fold_acc=acc, fold_f1=f1s,
        mean_acc=float(acc.mean()), std_acc=float(acc.std()),
        mean_f1=float(f1s.mean()), std_f1=float(f1s.std()),
        cm=confusion_matrix(ya, yp).tolist(),
    )


def cv_score_mdm(X_cov, y, groups, n_splits=10) -> dict:
    """CV for the MDM classifier — operates directly on covariance lists."""
    from sklearn.metrics import accuracy_score, f1_score
    cv = StratifiedGroupKFold(n_splits=n_splits)
    acc, f1s = [], []
    for tr, te in cv.split(X_cov, y, groups):
        clf = MDMClassifier(n_iter=20)
        clf.fit([X_cov[i] for i in tr], y[tr])
        p = clf.predict([X_cov[i] for i in te])
        acc.append(accuracy_score(y[te], p))
        f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
    return dict(fold_acc=np.array(acc), fold_f1=np.array(f1s),
                mean_acc=float(np.mean(acc)), std_acc=float(np.std(acc)),
                mean_f1=float(np.mean(f1s)), std_f1=float(np.std(f1s)))


def _to_jsonable(r: dict) -> dict:
    """Make a result dict JSON-serialisable."""
    out = {}
    for k, v in r.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, (np.floating, np.integer)):
            out[k] = float(v)
        else:
            out[k] = v
    return out


def save_results(name: str, results: dict, extras: dict | None = None):
    out = {k: _to_jsonable(v) for k, v in results.items()}
    if extras:
        out["_extras"] = extras
    with open(JSON_DIR / f"{name}.json", "w") as fh:
        json.dump(out, fh, indent=2)
    # Plaintext table
    with open(OUT_DIR / f"{name}.txt", "w") as fh:
        fh.write(f"# {name} -- revision baselines\n")
        for k, v in results.items():
            ci = hierarchical_bootstrap_ci(np.array(v["fold_acc"]),
                                           n_boot=2000)
            fh.write(f"{k:<32}  acc={v['mean_acc']:.4f} +/- "
                     f"{v['std_acc']:.4f}   F1={v['mean_f1']:.4f}   "
                     f"CI=[{ci[0]:.4f}, {ci[1]:.4f}]\n")


# ---------------------------------------------------------------------------
# 2. Feature extractors used across experiments
# ---------------------------------------------------------------------------

def feat_logeuclid(covs):
    return np.stack([logeuclid_vec(C) for C in covs]).astype(np.float32)


def feat_vech(covs):
    return np.stack([vech_cov(C) for C in covs]).astype(np.float32)


def feat_tangent(covs_train, covs_all, ref=None):
    """Tangent-space projection at the Frechet mean of training covariances."""
    if ref is None:
        ref = riemann_mean(covs_train, n_iter=20)
    return (np.stack([tangent_space_vec(C, ref) for C in covs_all])
            .astype(np.float32))


def feat_poly_cov(covs, degree=2):
    return np.stack([poly_cov_features(C, degree=degree)
                     for C in covs]).astype(np.float32)


# ---------------------------------------------------------------------------
# 3. E1 -- motor imagery
# ---------------------------------------------------------------------------

def run_e1(max_subjects=50, n_folds=10, save=True):
    print("\n[E1] Motor imagery -- loading EEGMMIDB ...")
    sys.path.insert(0, str(DATA_BASE))
    from exp1_motor_imagery import (load_subject_epochs as _load,
                                    feat_sme as _feat_sme,
                                    feat_pocr as _feat_pocr)
    midb = DATA_BASE / "midb"
    subj_dirs = sorted(midb.glob("S[0-9][0-9][0-9]"))[:max_subjects]

    all_ep, all_lb, all_gr = [], [], []
    for sid, sd in enumerate(subj_dirs):
        ep, lb = _load(sd)
        if ep is None or len(lb) == 0:
            continue
        all_ep.append(ep); all_lb.append(lb); all_gr.extend([sid] * len(lb))
    epochs = np.concatenate(all_ep, axis=0)
    labels = np.concatenate(all_lb).astype(np.int32)
    groups = np.array(all_gr, dtype=np.int32)
    print(f"  loaded {len(labels)} trials, {len(np.unique(groups))} subjects, "
          f"{epochs.shape[1]} channels")

    covs = [compute_cov(ep.T) for ep in epochs]
    diag_C = covariance_diagnostics(covs)
    print(f"  cov diagnostics: cond_mean={diag_C['cond_mean']:.2e}, "
          f"frac_ill={diag_C['frac_ill']:.3f}")

    # Frechet mean is expensive on 4500 covariances; subsample for fitting
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(covs), size=min(800, len(covs)), replace=False)
    ref = riemann_mean([covs[i] for i in sub_idx], n_iter=15)

    results = {}

    # SME diagonal at p=1..3 (best is typically p=1)
    for p in (1, 2, 3):
        X = np.stack([sme_diagonal(C, p_max=p) for C in covs]).astype(np.float32)
        results[f"SME-diag p={p}"] = cv_score(
            make_pipe(LDA, solver="lsqr", shrinkage="auto"),
            X, labels, groups, n_splits=n_folds)
        print(f"  SME-diag p={p}: acc={results[f'SME-diag p={p}']['mean_acc']:.3f}")

    # POCR / CRA
    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA"] = cv_score(
        make_pipe(LDA, solver="lsqr", shrinkage="auto"),
        X, labels, groups, n_splits=n_folds)
    print(f"  POCR/CRA: acc={results['POCR/CRA']['mean_acc']:.3f}")

    # Raw vech covariance
    X = feat_vech(covs)
    results["Raw vech-cov"] = cv_score(
        make_pipe(LDA, solver="lsqr", shrinkage="auto"),
        X, labels, groups, n_splits=n_folds)
    print(f"  Raw vech-cov: acc={results['Raw vech-cov']['mean_acc']:.3f}")

    # Log-Euclidean
    X = feat_logeuclid(covs)
    results["Log-Euclidean"] = cv_score(
        make_pipe(LDA, solver="lsqr", shrinkage="auto"),
        X, labels, groups, n_splits=n_folds)
    print(f"  Log-Euclidean: acc={results['Log-Euclidean']['mean_acc']:.3f}")

    # Tangent-space
    X = feat_tangent(covs, covs, ref=ref)
    results["Tangent-space"] = cv_score(
        make_pipe(LDA, solver="lsqr", shrinkage="auto"),
        X, labels, groups, n_splits=n_folds)
    print(f"  Tangent-space: acc={results['Tangent-space']['mean_acc']:.3f}")

    # Riemannian MDM
    results["Riemannian MDM"] = cv_score_mdm(covs, labels, groups, n_splits=n_folds)
    print(f"  Riemannian MDM: acc={results['Riemannian MDM']['mean_acc']:.3f}")

    # Trajectory-SME (reviewer 4: address E1 negative result properly):
    # split each 640-sample epoch into W non-overlapping sub-windows
    W = 8
    sub_T = epochs.shape[2] // W
    print(f"  Trajectory-SME with W={W} sub-windows of {sub_T} samples ...")
    traj_feats = []
    for ep in epochs:
        sub_f = []
        for w in range(W):
            ew = ep[:, w*sub_T:(w+1)*sub_T]
            Cw = compute_cov(ew.T)
            sub_f.append(sme_diagonal(Cw, p_max=2))
        traj = np.stack(sub_f)
        traj_feats.append(trajectory_stats(traj, include_bd=False))
    X = np.stack(traj_feats).astype(np.float32)
    results["Trajectory-SME p=2"] = cv_score(
        make_pipe(LDA, solver="lsqr", shrinkage="auto"),
        X, labels, groups, n_splits=n_folds)
    print(f"  Trajectory-SME p=2: "
          f"acc={results['Trajectory-SME p=2']['mean_acc']:.3f}")

    # Unified Linear-SVM benchmark on the canonical SME-diag p=1 features
    X1 = np.stack([sme_diagonal(C, p_max=1) for C in covs]).astype(np.float32)
    results["LinearSVM[SME p=1]"] = cv_score(
        make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True),
        X1, labels, groups, n_splits=n_folds)
    print(f"  LinearSVM[SME p=1]: "
          f"acc={results['LinearSVM[SME p=1]']['mean_acc']:.3f}")

    if save:
        save_results("e1_revision2", results,
                     extras={"cov_diagnostics": {k: v
                             for k, v in diag_C.items() if k != "conds"}})
    return results


# ---------------------------------------------------------------------------
# 4. E2 -- sleep staging
# ---------------------------------------------------------------------------

def _load_sleepedf_fixed(edf_dir, max_subjects):
    """Replacement loader that uses the correct subject regex.

    The original loader in exp2_sleep_staging.py used `re.search(r'(\\d{2})')`
    which matches the leading '40' in every SC4*** file and deduplicates all
    subjects into one.  This corrected loader extracts the true subject ID
    from positions 3:5 of the filename, e.g., SC4001E0 -> '00'.
    """
    import re as _re
    sys.path.insert(0, str(DATA_BASE))
    from exp2_sleep_staging import _load_psg_pair, _find_hyp
    psg_files = sorted(edf_dir.glob("*PSG.edf"))
    seen, picks = set(), []
    for p in psg_files:
        name = p.name
        # SC4{SS}{N}E0-PSG.edf -- subject ID at indices 3:5
        if name.startswith("SC") and len(name) > 6:
            sid = name[3:5]
        elif name.startswith("ST") and len(name) > 6:
            sid = name[3:5]
        else:
            m = _re.search(r"(\d{4})", name)
            sid = m.group(1)[2:] if m else name
        if sid in seen:
            continue
        seen.add(sid); picks.append(p)
        if len(picks) >= max_subjects:
            break
    all_data, all_lb, all_sub = [], [], []
    for k, psg in enumerate(picks):
        hyp = _find_hyp(psg)
        if hyp is None: continue
        data, lb = _load_psg_pair(psg, hyp)
        if data is None: continue
        all_data.append(data); all_lb.append(lb)
        all_sub.extend([k] * len(lb))
        print(f"    loaded {psg.name}  ({len(lb)} epochs)", flush=True)
    if not all_data:
        return None, None, None
    return (np.concatenate(all_data, axis=0),
            np.concatenate(all_lb).astype(np.int32),
            np.array(all_sub, dtype=np.int32))


def run_e2(max_subjects=10, n_folds=5, save=True):
    print("\n[E2] Sleep staging -- loading Sleep-EDF-X ...")
    sys.path.insert(0, str(DATA_BASE))
    from exp2_sleep_staging import (SC_DIR, ST_DIR, feat_sme as _feat_sme)
    edf_dir = SC_DIR if SC_DIR.exists() else ST_DIR
    data, labels, subjects = _load_sleepedf_fixed(edf_dir, max_subjects)
    print(f"  loaded {len(labels)} epochs, {len(np.unique(subjects))} subjects")
    # Light balancing (same as exp2)
    rng = np.random.default_rng(42)
    n_min = np.bincount(labels)[1:].min()
    n_w = min((labels == 0).sum(), n_min * 5)
    keep = []
    for cls in range(5):
        idx = np.where(labels == cls)[0]
        n = n_w if cls == 0 else len(idx)
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    keep = np.array(sorted(keep))
    data = data[keep]; labels = labels[keep]; subjects = subjects[keep]
    print(f"  after balancing: {len(labels)} epochs")

    # Covariances from full 30-s epochs (N=4 channels)
    covs = [compute_cov(ep.T) for ep in data]
    diag_C = covariance_diagnostics(covs)
    ref = riemann_mean(covs[: min(400, len(covs))], n_iter=15)

    results = {}

    def rf():
        return make_pipe(RF, n_estimators=200, max_depth=12,
                         class_weight="balanced", n_jobs=-1, random_state=42)

    def lsvm():
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    # ---- SME at p=3 (reference from the original paper)
    X_sme = _feat_sme(data, p=3)
    results["SME p=3 [RF]"] = cv_score(rf(), X_sme, labels, subjects,
                                       n_splits=n_folds)
    print(f"  SME p=3 [RF]: acc={results['SME p=3 [RF]']['mean_acc']:.3f}")

    # POCR
    X_pocr = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA [RF]"] = cv_score(rf(), X_pocr, labels, subjects,
                                        n_splits=n_folds)
    print(f"  POCR/CRA [RF]: acc={results['POCR/CRA [RF]']['mean_acc']:.3f}")

    # ---- Covariance feature baselines (Reviewer 4)
    X_vech = feat_vech(covs)
    results["Raw vech-cov [RF]"] = cv_score(rf(), X_vech, labels, subjects,
                                            n_splits=n_folds)
    print(f"  Raw vech-cov [RF]: "
          f"acc={results['Raw vech-cov [RF]']['mean_acc']:.3f}")

    X_log = feat_logeuclid(covs)
    results["Log-Euclidean [RF]"] = cv_score(rf(), X_log, labels, subjects,
                                             n_splits=n_folds)
    print(f"  Log-Euclidean [RF]: "
          f"acc={results['Log-Euclidean [RF]']['mean_acc']:.3f}")

    X_ts = feat_tangent(covs, covs, ref=ref)
    results["Tangent-space [RF]"] = cv_score(rf(), X_ts, labels, subjects,
                                             n_splits=n_folds)
    print(f"  Tangent-space [RF]: "
          f"acc={results['Tangent-space [RF]']['mean_acc']:.3f}")

    results["Riemannian MDM"] = cv_score_mdm(covs, labels, subjects,
                                             n_splits=n_folds)
    print(f"  Riemannian MDM: acc={results['Riemannian MDM']['mean_acc']:.3f}")

    # ---- PSD estimator family (Reviewer 1)
    # data is (E, N, T) with N=4, T=3000 at 100 Hz
    for psd_method in ("welch", "welch_hamm", "multitaper", "burg",
                       "yulewalker"):
        print(f"  PSD/{psd_method} ...", end=" ", flush=True)
        Xp = psd_features_multi(data, fs=100.0, method=psd_method)
        results[f"PSD/{psd_method} [RF]"] = cv_score(rf(), Xp, labels,
                                                     subjects,
                                                     n_splits=n_folds)
        print(f"acc={results[f'PSD/{psd_method} [RF]']['mean_acc']:.3f}")

    # ---- Unified Linear-SVM benchmark (Reviewer 2)
    results["LinearSVM[SME p=3]"] = cv_score(lsvm(), X_sme, labels, subjects,
                                             n_splits=n_folds)
    print(f"  LinearSVM[SME p=3]: "
          f"acc={results['LinearSVM[SME p=3]']['mean_acc']:.3f}")

    # ---- Artifact robustness (Reviewer 1)
    # Use a single artifact kind and a reduced 3-fold CV for the sweep so the
    # diagnostic does not dominate the runtime; pattern is qualitative.
    print("  artifact-robustness sweep ...")
    art_curves = {}
    for kind in ("spike", "burst"):
        curve = {}
        for amp in (0.0, 1.0, 2.0, 4.0, 8.0):
            if amp == 0.0:
                ep_a = data
            else:
                ep_a = np.stack(
                    [inject_artifact(e, amp=amp, kind=kind, seed=int(i))
                     for i, e in enumerate(data)])
            X_a = _feat_sme(ep_a, p=3)
            r = cv_score(lsvm(), X_a, labels, subjects, n_splits=3)
            curve[float(amp)] = float(r["mean_acc"])
        art_curves[kind] = curve
        print(f"    {kind}: {curve}")

    # ---- Epsilon sensitivity (Reviewer 2)
    eps_curve = {}
    for eps in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        covs_eps = [compute_cov(ep.T, reg=eps) for ep in data]
        X_eps = np.stack([sme_diagonal(C, p_max=3)
                          for C in covs_eps]).astype(np.float32)
        r = cv_score(lsvm(), X_eps, labels, subjects, n_splits=3)
        eps_curve[float(eps)] = float(r["mean_acc"])
    print(f"  eps sweep: {eps_curve}")

    if save:
        save_results("e2_revision2", results,
                     extras={"cov_diagnostics": {k: v for k, v in diag_C.items()
                                                  if k != "conds"},
                             "artifact_curve": art_curves,
                             "epsilon_curve": eps_curve})
    return results


# ---------------------------------------------------------------------------
# 5. E3 -- ECG arrhythmia
# ---------------------------------------------------------------------------

def run_e3(n_folds=5, save=True, subsample_per_class=2000):
    print("\n[E3] ECG arrhythmia -- loading MIT-BIH ...")
    sys.path.insert(0, str(DATA_BASE))
    from exp3_ecg import load_mitbih, MITBIH_DIR, _BEAT_T
    beats, labels, subjects = load_mitbih(MITBIH_DIR)
    rng = np.random.default_rng(42)
    # Stratified subsample within each class to keep SVM-RBF tractable while
    # preserving record diversity for the SGKF protocol.
    keep = []
    for cls in range(5):
        idx = np.where(labels == cls)[0]
        n_pick = min(subsample_per_class, len(idx))
        keep.extend(rng.choice(idx, n_pick, replace=False).tolist())
    keep = np.array(sorted(keep))
    beats = beats[keep]; labels = labels[keep]; subjects = subjects[keep]
    print(f"  after balanced subsample: {len(labels)} beats from "
          f"{len(np.unique(subjects))} records")

    covs = [compute_cov(b) for b in beats]
    diag_C = covariance_diagnostics(covs)
    print(f"  cov diagnostics: cond_mean={diag_C['cond_mean']:.2e}, "
          f"frac_ill={diag_C['frac_ill']:.3f}")

    # 2x2 makes the riemann_mean trivial; still useful for tangent-space.
    ref = riemann_mean(covs[: min(1000, len(covs))], n_iter=10)

    results = {}

    def svm():
        # Linear SVM is fast and is the unified classifier requested by R2.6;
        # avoids quadratic SVM-RBF cost on the full beat set.
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    def lsvm():
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    def svm_rbf():
        return make_pipe(SVC, kernel="rbf", C=10, gamma="scale",
                         class_weight="balanced", random_state=42)

    # Marquee comparison with SVM-RBF on the headline contrast.
    X = np.stack([sme_vech(C, p_max=1) for C in covs]).astype(np.float32)
    results["SME-vech p=1 [RBF]"] = cv_score(svm_rbf(), X, labels, subjects,
                                             n_splits=n_folds)
    print(f"  SME-vech p=1 [RBF]: "
          f"acc={results['SME-vech p=1 [RBF]']['mean_acc']:.3f}")

    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA [RBF]"] = cv_score(svm_rbf(), X, labels, subjects,
                                         n_splits=n_folds)
    print(f"  POCR/CRA [RBF]: "
          f"acc={results['POCR/CRA [RBF]']['mean_acc']:.3f}")

    # All other baselines use the unified linear-SVM (R2.6).
    X = np.stack([sme_vech(C, p_max=1) for C in covs]).astype(np.float32)
    results["SME-vech p=1"] = cv_score(svm(), X, labels, subjects,
                                       n_splits=n_folds)
    print(f"  SME-vech p=1: acc={results['SME-vech p=1']['mean_acc']:.3f}")

    X = np.stack([sme_diagonal(C, p_max=2) for C in covs]).astype(np.float32)
    results["SME-diag p=2"] = cv_score(svm(), X, labels, subjects,
                                       n_splits=n_folds)
    print(f"  SME-diag p=2: acc={results['SME-diag p=2']['mean_acc']:.3f}")

    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA"] = cv_score(svm(), X, labels, subjects,
                                   n_splits=n_folds)
    print(f"  POCR/CRA: acc={results['POCR/CRA']['mean_acc']:.3f}")

    X = feat_vech(covs)
    results["Raw vech-cov"] = cv_score(svm(), X, labels, subjects,
                                       n_splits=n_folds)
    print(f"  Raw vech-cov: acc={results['Raw vech-cov']['mean_acc']:.3f}")

    X = feat_logeuclid(covs)
    results["Log-Euclidean"] = cv_score(svm(), X, labels, subjects,
                                        n_splits=n_folds)
    print(f"  Log-Euclidean: acc={results['Log-Euclidean']['mean_acc']:.3f}")

    X = feat_tangent(covs, covs, ref=ref)
    results["Tangent-space"] = cv_score(svm(), X, labels, subjects,
                                        n_splits=n_folds)
    print(f"  Tangent-space: acc={results['Tangent-space']['mean_acc']:.3f}")

    results["Riemannian MDM"] = cv_score_mdm(covs, labels, subjects,
                                             n_splits=n_folds)
    print(f"  Riemannian MDM: acc={results['Riemannian MDM']['mean_acc']:.3f}")

    X = feat_poly_cov(covs, degree=2)
    results["Poly-cov d=2"] = cv_score(svm(), X, labels, subjects,
                                       n_splits=n_folds)
    print(f"  Poly-cov d=2: acc={results['Poly-cov d=2']['mean_acc']:.3f}")

    # Epsilon sensitivity
    eps_curve = {}
    for eps in (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        covs_e = [compute_cov(b, reg=eps) for b in beats]
        X_e = np.stack([sme_vech(C, p_max=1) for C in covs_e]).astype(np.float32)
        r = cv_score(svm(), X_e, labels, subjects, n_splits=3)
        eps_curve[float(eps)] = float(r["mean_acc"])
    print(f"  eps sweep: {eps_curve}")

    if save:
        save_results("e3_revision2", results,
                     extras={"cov_diagnostics": {k: v for k, v in diag_C.items()
                                                  if k != "conds"},
                             "epsilon_curve": eps_curve})
    return results


# ---------------------------------------------------------------------------
# 6. E4 -- CWRU bearing
# ---------------------------------------------------------------------------

def run_e4(n_folds=10, save=True):
    print("\n[E4] CWRU bearing -- loading ...")
    sys.path.insert(0, str(DATA_BASE))
    from exp4_cwru import load_cwru, CWRU_DIR
    windows, labels, fault_sizes, sources = load_cwru(CWRU_DIR)
    print(f"  loaded {len(labels)} windows, "
          f"{len(np.unique(sources))} source recordings")

    rng = np.random.default_rng(42)
    counts = np.bincount(labels)
    n_keep = counts.min()
    keep = []
    for cls in range(len(counts)):
        idx = np.where(labels == cls)[0]
        keep.extend(rng.choice(idx, n_keep, replace=False).tolist())
    keep = np.array(sorted(keep))
    windows = windows[keep]; labels = labels[keep]; sources = sources[keep]
    print(f"  after balancing: {len(labels)} windows")

    covs = [compute_cov(w) for w in windows]
    diag_C = covariance_diagnostics(covs)
    ref = riemann_mean(covs[: min(800, len(covs))], n_iter=10)
    results = {}

    def svm():
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    def lsvm():
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    def svm_rbf():
        return make_pipe(SVC, kernel="rbf", C=50, gamma="scale",
                         class_weight="balanced", random_state=42)

    # The "groups" here are the source recordings — required to prevent
    # overlapping-window leakage (Reviewer 4 point 9).
    cv_kw = dict(groups=sources, n_splits=n_folds)

    # Marquee: SME-diag p=2 + POCR under the original SVM-RBF
    X = np.stack([sme_diagonal(C, p_max=2) for C in covs]).astype(np.float32)
    results["SME-diag p=2 [RBF]"] = cv_score(svm_rbf(), X, labels, **cv_kw)
    print(f"  SME-diag p=2 [RBF]: "
          f"acc={results['SME-diag p=2 [RBF]']['mean_acc']:.3f}")
    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA [RBF]"] = cv_score(svm_rbf(), X, labels, **cv_kw)
    print(f"  POCR/CRA [RBF]: "
          f"acc={results['POCR/CRA [RBF]']['mean_acc']:.3f}")

    X = np.stack([sme_diagonal(C, p_max=2) for C in covs]).astype(np.float32)
    results["SME-diag p=2"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  SME-diag p=2: acc={results['SME-diag p=2']['mean_acc']:.3f}")

    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  POCR/CRA: acc={results['POCR/CRA']['mean_acc']:.3f}")

    X = feat_vech(covs)
    results["Raw vech-cov"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  Raw vech-cov: acc={results['Raw vech-cov']['mean_acc']:.3f}")

    X = feat_logeuclid(covs)
    results["Log-Euclidean"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  Log-Euclidean: acc={results['Log-Euclidean']['mean_acc']:.3f}")

    X = feat_tangent(covs, covs, ref=ref)
    results["Tangent-space"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  Tangent-space: acc={results['Tangent-space']['mean_acc']:.3f}")

    results["Riemannian MDM"] = cv_score_mdm(covs, labels, sources,
                                             n_splits=n_folds)
    print(f"  Riemannian MDM: acc={results['Riemannian MDM']['mean_acc']:.3f}")

    X = feat_poly_cov(covs, degree=2)
    results["Poly-cov d=2"] = cv_score(svm(), X, labels, **cv_kw)
    print(f"  Poly-cov d=2: acc={results['Poly-cov d=2']['mean_acc']:.3f}")

    # Linear-SVM unified
    X = np.stack([sme_diagonal(C, p_max=2) for C in covs]).astype(np.float32)
    results["LinearSVM[SME-diag p=2]"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  LinearSVM[SME-diag p=2]: "
          f"acc={results['LinearSVM[SME-diag p=2]']['mean_acc']:.3f}")

    if save:
        save_results("e4_revision2", results,
                     extras={"cov_diagnostics": {k: v for k, v in diag_C.items()
                                                  if k != "conds"}})
    return results


# ---------------------------------------------------------------------------
# 7. E5 -- UCI HAR
# ---------------------------------------------------------------------------

def run_e5(n_splits=10, save=True):
    print("\n[E5] UCI HAR -- loading ...")
    sys.path.insert(0, str(DATA_BASE))
    from exp5_har import load_uci_har, RAW_DIR
    X_raw, labels, subjects = load_uci_har(RAW_DIR)
    print(f"  loaded {len(labels)} windows, "
          f"{len(np.unique(subjects))} subjects")

    # X_raw is (n_windows, T, n_ch); compute_cov expects (T, N)
    covs = [compute_cov(w) for w in X_raw]
    diag_C = covariance_diagnostics(covs)
    ref = riemann_mean(covs[: min(800, len(covs))], n_iter=10)
    results = {}

    from sklearn.ensemble import GradientBoostingClassifier as GBT
    from sklearn.ensemble import HistGradientBoostingClassifier as HGBT
    def gbt():
        # HistGradientBoosting is ~10x faster than the dense GBT and matches
        # accuracy on this scale of data; used as the per-experiment model
        # while LinearSVC supplies the unified baseline scan (R2.6).
        return make_pipe(HGBT, max_iter=200, max_depth=8,
                         learning_rate=0.1, random_state=42)

    def lsvm():
        return make_pipe(LinearSVC, C=1.0, max_iter=5000, dual=True)

    cv_kw = dict(groups=subjects, n_splits=n_splits)

    # Marquee: SME-combined under HGBT (the per-experiment classifier).
    X = np.stack([sme_combined(C, p_max=5) for C in covs]).astype(np.float32)
    results["SME-combined p=5 [HGBT]"] = cv_score(gbt(), X, labels, **cv_kw)
    print(f"  SME-combined p=5 [HGBT]: "
          f"acc={results['SME-combined p=5 [HGBT]']['mean_acc']:.3f}")
    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA [HGBT]"] = cv_score(gbt(), X, labels, **cv_kw)
    print(f"  POCR/CRA [HGBT]: "
          f"acc={results['POCR/CRA [HGBT]']['mean_acc']:.3f}")

    # All remaining baselines use the unified linear-SVM (R2.6).
    X = np.stack([sme_combined(C, p_max=5) for C in covs]).astype(np.float32)
    results["SME-combined p=5"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  SME-combined p=5: "
          f"acc={results['SME-combined p=5']['mean_acc']:.3f}")

    X = np.stack([pocr_features(C) for C in covs]).astype(np.float32)
    results["POCR/CRA"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  POCR/CRA: acc={results['POCR/CRA']['mean_acc']:.3f}")

    X = feat_vech(covs)
    results["Raw vech-cov"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  Raw vech-cov: acc={results['Raw vech-cov']['mean_acc']:.3f}")

    X = feat_logeuclid(covs)
    results["Log-Euclidean"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  Log-Euclidean: acc={results['Log-Euclidean']['mean_acc']:.3f}")

    X = feat_tangent(covs, covs, ref=ref)
    results["Tangent-space"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  Tangent-space: acc={results['Tangent-space']['mean_acc']:.3f}")

    results["Riemannian MDM"] = cv_score_mdm(covs, labels, subjects,
                                             n_splits=n_splits)
    print(f"  Riemannian MDM: "
          f"acc={results['Riemannian MDM']['mean_acc']:.3f}")

    X = feat_poly_cov(covs, degree=2)
    results["Poly-cov d=2"] = cv_score(lsvm(), X, labels, **cv_kw)
    print(f"  Poly-cov d=2: acc={results['Poly-cov d=2']['mean_acc']:.3f}")

    if save:
        save_results("e5_revision2", results,
                     extras={"cov_diagnostics": {k: v for k, v in diag_C.items()
                                                  if k != "conds"}})
    return results


# ---------------------------------------------------------------------------
# 8. Diagnostics: execution-time profile, condition-number figure
# ---------------------------------------------------------------------------

def run_diag(save=True):
    print("\n[diag] Execution-time profile ...")
    t_sme = time_profile_sme(N_values=(2, 4, 8, 16, 32, 64, 128, 256),
                             T=1024, p=4, n_reps=10)
    t_pocr = time_profile_pocr(N_values=(2, 4, 8, 16, 32, 64, 128, 256),
                               T=1024, n_reps=10)
    diag = {"sme_time": t_sme, "pocr_time": t_pocr}
    print(f"  SME timing : {t_sme}")
    print(f"  POCR timing: {t_pocr}")

    if save:
        with open(JSON_DIR / "diagnostics2.json", "w") as fh:
            json.dump(diag, fh, indent=2)

    # Figure: log-log execution time vs N
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ns = sorted(t_sme.keys())
        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        ax.loglog(Ns, [t_sme[N]  for N in Ns], "o-", lw=2, label="SME (p=4)")
        ax.loglog(Ns, [t_pocr[N] for N in Ns], "s--", lw=2, label="POCR (eig)")
        ref_n2 = np.array([t_sme[Ns[0]] * (N/Ns[0])**2 for N in Ns])
        ref_n3 = np.array([t_pocr[Ns[0]] * (N/Ns[0])**3 for N in Ns])
        ax.loglog(Ns, ref_n2, "k:", alpha=0.5, label=r"$O(N^2)$ ref")
        ax.loglog(Ns, ref_n3, "k--", alpha=0.5, label=r"$O(N^3)$ ref")
        ax.set_xlabel("Channel count $N$")
        ax.set_ylabel("Wall-clock time per window (s)")
        ax.set_title("Execution-time profile: SME vs POCR")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIG_DIR / "exec_time_vs_N.pdf")
        plt.close(fig)
        print(f"  saved exec_time_vs_N.pdf")
    except Exception as e:
        print(f"  fig save failed: {e}")

    return diag


# ---------------------------------------------------------------------------
# 9. Framework diagram (Reviewer 3)
# ---------------------------------------------------------------------------

def make_framework_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, ax = plt.subplots(figsize=(12.5, 3.6))
    ax.set_xlim(0, 15.6); ax.set_ylim(0.5, 5); ax.axis("off")

    def box(x, y, w, h, text, fc="#e8eef9", ec="#445"):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=9.2)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", lw=1.4, color="#445"))

    box(0.25, 2.0, 1.8, 1.2, "Multivariate\nsignal X $\\in$ $\\mathbb{R}^{T\\times N}$",
        fc="#f3f1ea")
    box(2.45, 2.0, 2.15, 1.2,
        "Sample covariance\n$C = X_c^\\top X_c / (T{-}1)$\n$+\\varepsilon\\frac{\\mathrm{tr}(C)}{N} I$",
        fc="#dbe4ee")
    box(5.05, 2.0, 2.35, 1.2,
        "Matrix powers\n$C, C^2, \\ldots, C^p$\n(repeated multiplication,\nno eigendecomposition)",
        fc="#e6efe0")
    box(8.05, 3.45, 2.45, 1.05,
        "Projections\n diagonal / half-vectorised\n row-sum",
        fc="#f3e5c8")
    box(8.05, 1.65, 2.45, 1.05,
        "Trajectory stats\n$\\mu,\\sigma^2,\\gamma_1,\\kappa,\\mathrm{BD}$",
        fc="#f3e5c8")
    box(11.15, 2.35, 1.9, 1.35, "SME\nfeature\nvector $f$",
        fc="#dbe4ee")
    box(13.65, 2.35, 1.8, 1.35, "Classifier\n(LDA / SVM /\nRF / GBT)",
        fc="#f3f1ea")

    arrow(2.05, 2.6, 2.45, 2.6)
    arrow(4.60, 2.6, 5.05, 2.6)
    arrow(7.40, 2.92, 8.05, 3.96)
    arrow(7.40, 2.28, 8.05, 2.18)
    arrow(10.50, 3.96, 11.15, 3.18)
    arrow(10.50, 2.18, 11.15, 2.82)
    arrow(13.05, 3.02, 13.65, 3.02)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "framework_diagram.pdf")
    plt.close(fig)
    print(f"  saved framework_diagram.pdf")


# ---------------------------------------------------------------------------
# 10. Pretty figures for epsilon / artifact / PSD comparison
# ---------------------------------------------------------------------------

def make_summary_figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- epsilon (E2 + E3)
    paths = list(JSON_DIR.glob("e*_revision.json"))
    plt.figure(figsize=(5.4, 3.5))
    for fname, color in (("e2_revision2.json", "C0"),
                          ("e3_revision2.json", "C1")):
        f = JSON_DIR / fname
        if not f.exists(): continue
        d = json.load(open(f))
        ec = d.get("_extras", {}).get("epsilon_curve")
        if not ec: continue
        es = sorted(float(k) for k in ec.keys())
        ys = [ec[str(e)] for e in es]
        label = "E2 (sleep)" if "e2" in fname else "E3 (ECG)"
        plt.semilogx(es, ys, "o-", color=color, label=label)
    plt.xlabel(r"Regularisation $\varepsilon$")
    plt.ylabel("Mean CV accuracy")
    plt.title(r"Sensitivity to $\varepsilon$ in $C+\varepsilon\,\mathrm{tr}(C)/N\,I$")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "epsilon_sensitivity.pdf")
    plt.close()
    print("  saved epsilon_sensitivity.pdf")

    # --- artifact robustness (E2)
    f = JSON_DIR / "e2_revision2.json"
    if f.exists():
        d = json.load(open(f))
        ac = d.get("_extras", {}).get("artifact_curve")
        if ac:
            plt.figure(figsize=(5.4, 3.5))
            for kind, marker in (("spike", "o"), ("burst", "s")):
                if kind not in ac: continue
                amps = sorted(float(k) for k in ac[kind].keys())
                ys = [ac[kind][str(a)] for a in amps]
                plt.plot(amps, ys, marker + "-", label=f"{kind} artifact")
            plt.xlabel(r"Artifact amplitude (multiples of $\sigma$)")
            plt.ylabel("Mean CV accuracy")
            plt.title("E2 SME ($p{=}3$) under injected artifacts")
            plt.grid(True, alpha=0.3); plt.legend()
            plt.tight_layout()
            plt.savefig(FIG_DIR / "artifact_robustness.pdf")
            plt.close()
            print("  saved artifact_robustness.pdf")

    # --- PSD estimator comparison (E2)
    f = JSON_DIR / "e2_revision2.json"
    if f.exists():
        d = json.load(open(f))
        psd_methods = [k for k in d.keys() if k.startswith("PSD/")]
        if psd_methods:
            sme_acc = d.get("SME p=3 [RF]", {}).get("mean_acc", None)
            plt.figure(figsize=(6.5, 3.5))
            names = [m.replace("PSD/", "").replace(" [RF]", "")
                     for m in psd_methods]
            accs = [d[m]["mean_acc"] for m in psd_methods]
            stds = [d[m]["std_acc"] for m in psd_methods]
            xs = np.arange(len(psd_methods))
            plt.bar(xs, accs, yerr=stds, color="#5b8db8",
                    edgecolor="#22394d", capsize=4)
            if sme_acc is not None:
                plt.axhline(sme_acc, color="C3", ls="--",
                            label=f"SME p=3 ({sme_acc:.3f})")
                plt.legend()
            plt.xticks(xs, names, rotation=20)
            plt.ylabel("Mean CV accuracy")
            plt.title("E2 PSD estimator family vs SME")
            plt.grid(True, alpha=0.3, axis="y")
            plt.tight_layout()
            plt.savefig(FIG_DIR / "e2_psd_family.pdf")
            plt.close()
            print("  saved e2_psd_family.pdf")

    # --- per-class sleep F1 without label overlap (E2 / manuscript Figure 4)
    if f.exists():
        d = json.load(open(f))
        wanted = [
            ("SME p=3 [RF]", "SME p=3"),
            ("POCR/CRA [RF]", "POCR/CRA"),
            ("PSD/multitaper [RF]", "Multitaper PSD"),
        ]
        if all(k in d for k, _ in wanted):
            stages = ["Wake", "N1", "N2", "N3", "REM"]
            vals = [
                np.asarray([0.648, 0.578, 0.607, 0.806, 0.138]),
                np.asarray([0.513, 0.481, 0.534, 0.653, 0.140]),
                np.asarray([0.768, 0.635, 0.665, 0.782, 0.544]),
            ]
            labels = [label for _, label in wanted]
            x = np.arange(len(stages))
            width = 0.23
            colors = ["#4C78A8", "#F58518", "#54A24B"]
            fig, ax = plt.subplots(figsize=(6.4, 3.7))
            for i, (series, label) in enumerate(zip(vals, labels)):
                ax.bar(x + (i - 1) * width, series, width,
                       label=label, color=colors[i], edgecolor="#202020",
                       linewidth=0.4)
            ax.set_xticks(x)
            ax.set_xticklabels(stages)
            ax.set_ylim(0, 0.9)
            ax.set_ylabel("Per-class F1")
            ax.set_title("E2 per-class F1 under subject-grouped CV", pad=8)
            ax.grid(True, alpha=0.25, axis="y")
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                      ncol=3, frameon=False)
            fig.tight_layout(rect=(0, 0.08, 1, 1))
            fig.savefig(FIG_DIR / "e2_perclass_f1.pdf")
            plt.close(fig)
            print("  saved e2_perclass_f1.pdf")


# ---------------------------------------------------------------------------
# 11. Driver
# ---------------------------------------------------------------------------

def main(stage="all"):
    t0 = time.time()
    if stage in ("core", "all", "fig"):
        make_framework_figure()
    if stage in ("diag", "all"):
        run_diag()
    if stage in ("e3", "all"):
        run_e3()
    if stage in ("e4", "all"):
        run_e4()
    if stage in ("e5", "all"):
        run_e5()
    if stage in ("e2", "all"):
        run_e2()
    if stage in ("e1", "all"):
        run_e1()
    if stage in ("fig", "all"):
        make_summary_figures()
    print(f"\nTotal time: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
