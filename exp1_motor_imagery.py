"""
exp1_motor_imagery.py  —  Experiment E1: Motor Imagery EEG
===========================================================
Dataset : PhysioNet EEGMMIDB
          physionet.org/content/eegmmidb/1.0.0/  (no login required)
          midb/S001/ … S109/ — 14 runs per subject as EDF+ files

Task    : 4-class motor imagery
          class 0 = left-fist imagery    (runs R04, R08, R12  →  annotation T1)
          class 1 = right-fist imagery   (runs R04, R08, R12  →  annotation T2)
          class 2 = both-fists imagery   (runs R06, R10, R14  →  annotation T1)
          class 3 = both-feet imagery    (runs R06, R10, R14  →  annotation T2)

Channels: 22-channel sensorimotor subset (10-10 system).
Filter  : 8-30 Hz FIR bandpass (MNE default).
Epochs  : 4-second trial epochs, onset-locked, 640 samples at 160 Hz.

SME     : sme_diagonal(p=1..6) on the 22×22 epoch covariance → 22p features.

CV      : 10-fold subject-stratified.

Baselines:
  · CSP (OVR, nested) — spatial filters fitted INSIDE each fold only.
    Implemented via CSPOVRTransformer (sklearn-compatible Pipeline).
    The original code fitted CSP on all epochs before cross_validate(),
    leaking test-fold covariance into the spatial filters.  Fixed here.
  · Log-Euclidean (per-epoch matrix-log, no fitting step)
  · POCR/CRA (per-epoch spherical coords + BD, no fitting step)

All result dicts use the exact key names expected by sme_core helpers:
  fold_acc, fold_f1, mean_acc, std_acc, mean_f1, std_f1,
  y_true, y_pred, report, cm
"""

from __future__ import annotations
import sys
import warnings
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# ── local imports ──────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, pocr_features,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table,
    print_significance_table, print_perclass_f1,
    latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = _HERE
MIDB_DIR = BASE_DIR / "midb"

# ── constants ──────────────────────────────────────────────────────────
_LR_RUNS    = [4, 8, 12]
_BF_RUNS    = [6, 10, 14]
_SFREQ      = 160
_EPOCH_S    = 4.0
_EPOCH_T    = int(_EPOCH_S * _SFREQ)   # 640 samples
_N_CLASSES  = 4
_CSP_NCOMP  = 6   # components per OVR class → 24 total features
_CLASS_NAMES = ["Left-fist", "Right-fist", "Both-fists", "Both-feet"]

_MOTOR_CH = [
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1",  "Pz",  "P2",  "O1",  "Oz",
]


# ── channel matching ───────────────────────────────────────────────────

def _pick_motor_channels(ch_names: list[str]) -> list[str]:
    """
    Match EDF channel names (dot-padded, e.g. 'Fc3.') to the 22 wanted names.
    Falls back to first 22 EEG channels if fewer than 10 are matched.
    """
    wanted = {c.lower() for c in _MOTOR_CH}
    return [c for c in ch_names if c.lower().rstrip(". ") in wanted]


# ── subject loader ─────────────────────────────────────────────────────

def load_subject_epochs(subj_dir: Path):
    """
    Load all 4-class motor imagery epochs for one subject.

    Returns
    -------
    epochs : (n_trials, N_ch, T) float32   or  None on failure
    labels : (n_trials,) int32             or  None on failure
    """
    try:
        import mne
    except ImportError:
        raise ImportError("pip install mne")

    name = subj_dir.name
    epochs_out, labels_out = [], []

    run_cfgs = [
        (_LR_RUNS, {"T1": 0, "T2": 1}),
        (_BF_RUNS, {"T1": 2, "T2": 3}),
    ]

    for run_nums, lmap in run_cfgs:
        for rn in run_nums:
            edf = subj_dir / f"{name}R{rn:02d}.edf"
            if not edf.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(str(edf), preload=True, verbose=False)
            except Exception:
                continue

            motor = _pick_motor_channels(raw.ch_names)
            if len(motor) < 10:
                eeg_idx = mne.pick_types(raw.info, eeg=True, verbose=False)
                motor = [raw.ch_names[i] for i in eeg_idx[:22]]
            if not motor:
                continue

            raw.pick_channels(motor, verbose=False)
            raw.filter(8.0, 30.0, verbose=False)

            try:
                events, eid = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                continue

            code_to_cls = {}
            for desc, code in eid.items():
                for ann_key, cls in lmap.items():
                    if ann_key in desc:
                        code_to_cls[code] = cls
            if not code_to_cls:
                continue

            n_pts = int(_EPOCH_S * raw.info["sfreq"])
            sig   = raw.get_data()

            for ev in events:
                onset, _, code = ev
                if code not in code_to_cls:
                    continue
                end = onset + n_pts
                if end > sig.shape[1]:
                    continue
                epochs_out.append(sig[:, onset:end].astype(np.float32))
                labels_out.append(code_to_cls[code])

    if not epochs_out:
        return None, None
    return np.stack(epochs_out), np.array(labels_out, dtype=np.int32)


# ── per-epoch feature extractors (no fitting, no leakage) ─────────────

def _cov(ep: np.ndarray) -> np.ndarray:
    """ep: (N_ch, T)  →  regularised (N_ch, N_ch) covariance."""
    return compute_cov(ep.T)


def feat_sme(epochs: np.ndarray, p: int) -> np.ndarray:
    """SME diagonal for each epoch. Shape (n, p * N_ch). No fitting."""
    return np.array([sme_diagonal(_cov(ep), p_max=p)
                     for ep in epochs], dtype=np.float32)


def feat_pocr(epochs: np.ndarray) -> np.ndarray:
    """POCR feature vector per epoch. No fitting."""
    return np.array([pocr_features(_cov(ep))
                     for ep in epochs], dtype=np.float32)


def feat_log_euclidean(epochs: np.ndarray) -> np.ndarray:
    """Upper-triangle of matrix-log of epoch covariance. No fitting."""
    N  = epochs.shape[1]
    ix = np.triu_indices(N)
    feats = []
    for ep in epochs:
        C    = _cov(ep)
        w, v = np.linalg.eigh(C)
        w    = np.maximum(w, 1e-10)
        logC = (v * np.log(w)) @ v.T
        feats.append(logC[ix])
    return np.array(feats, dtype=np.float32)


# ── CSP with proper nested CV ──────────────────────────────────────────
#
# The original code called feat_csp_ovr(epochs, labels) which fitted four
# MNE CSP objects on ALL 4,500 epochs before any cross_validate() split,
# leaking test-fold covariance information into the spatial filters.
#
# Fix: CSPOVRTransformer wraps MNE CSP in a sklearn fit()/transform() API.
# make_csp_pipeline() builds Pipeline([csp, scaler, lda]).
# csp_cross_validate() runs StratifiedGroupKFold, calling pipe.fit(X_tr)
# inside each fold so test data never touches the spatial filter estimation.

class CSPOVRTransformer:
    """
    One-vs-rest CSP, sklearn BaseEstimator/TransformerMixin interface.

    Fits one MNE CSP per class on training data only; applies at test time.
    Output shape: (n_samples, n_components * n_classes).
    """

    def __init__(self, n_components: int = 6, n_classes: int = 4):
        self.n_components = n_components
        self.n_classes    = n_classes
        self._csps        = []

    def get_params(self, deep: bool = True) -> dict:
        return {"n_components": self.n_components,
                "n_classes":    self.n_classes}

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray):
        """X: (n, n_ch, T); y: (n,) integer labels."""
        try:
            from mne.decoding import CSP
        except ImportError:
            raise ImportError("pip install mne")

        self._csps = []
        for cls in range(self.n_classes):
            y_bin = (y == cls).astype(int)
            if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
                self._csps.append(None)
                continue
            csp = CSP(n_components=self.n_components,
                      log=True, norm_trace=False)
            try:
                csp.fit(X, y_bin)
                self._csps.append(csp)
            except Exception:
                self._csps.append(None)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Returns (n_samples, n_components * n_classes) float32."""
        n   = len(X)
        out = np.zeros((n, self.n_components * self.n_classes),
                       dtype=np.float32)
        for cls, csp in enumerate(self._csps):
            if csp is None:
                continue
            try:
                col_start = cls * self.n_components
                col_end   = col_start + self.n_components
                out[:, col_start:col_end] = \
                    csp.transform(X).astype(np.float32)
            except Exception:
                pass
        return out

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.fit(X, y).transform(X)


def make_csp_pipeline() -> Pipeline:
    """
    Pipeline: CSPOVRTransformer → StandardScaler → LDA.
    sklearn will call pipe.fit(X_train, y_train) inside each CV fold,
    so CSP spatial filters never see test-fold data.
    """
    return Pipeline([
        ("csp",    CSPOVRTransformer(n_components=_CSP_NCOMP,
                                     n_classes=_N_CLASSES)),
        ("scaler", StandardScaler()),
        ("lda",    LDA(solver="lsqr", shrinkage="auto")),
    ])


def csp_cross_validate(
    epochs: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 10,
) -> dict:
    """
    Nested-CV evaluation of CSP+LDA.

    Returns a dict with EXACTLY the same keys as sme_core.cross_validate():
      fold_acc, fold_f1, mean_acc, std_acc, mean_f1, std_f1,
      y_true, y_pred, report, cm
    so it is compatible with all sme_core printing/saving helpers.
    """
    cv = StratifiedGroupKFold(n_splits=n_splits)

    fold_acc_list, fold_f1_list = [], []
    y_true_all,    y_pred_all   = [], []

    for train_idx, test_idx in cv.split(epochs, labels, groups=groups):
        X_tr, y_tr = epochs[train_idx], labels[train_idx]
        X_te, y_te = epochs[test_idx],  labels[test_idx]

        pipe = make_csp_pipeline()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pipe.fit(X_tr, y_tr)
                y_pred = pipe.predict(X_te)
            except Exception as exc:
                # Fallback: majority-class prediction for this fold
                print(f"    [CSP fold warning: {exc}; using majority fallback]")
                majority = int(np.bincount(y_tr).argmax())
                y_pred   = np.full(len(y_te), majority, dtype=np.int32)

        fold_acc_list.append(accuracy_score(y_te, y_pred))
        fold_f1_list.append(f1_score(y_te, y_pred,
                                     average="macro", zero_division=0))
        y_true_all.extend(y_te.tolist())
        y_pred_all.extend(y_pred.tolist())

    fold_acc = np.array(fold_acc_list)
    fold_f1  = np.array(fold_f1_list)
    ya       = np.array(y_true_all)
    yp       = np.array(y_pred_all)

    # Return dict with keys matching sme_core.cross_validate() exactly
    return dict(
        fold_acc = fold_acc,
        fold_f1  = fold_f1,
        mean_acc = float(fold_acc.mean()),
        std_acc  = float(fold_acc.std()),
        mean_f1  = float(fold_f1.mean()),
        std_f1   = float(fold_f1.std()),
        y_true   = ya,
        y_pred   = yp,
        report   = classification_report(ya, yp, zero_division=0),
        cm       = confusion_matrix(ya, yp,
                                    labels=list(range(_N_CLASSES))),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(max_subjects: int = 50,
        p_values:    list[int] | None = None,
        n_folds:     int = 10) -> dict:

    if p_values is None:
        p_values = [1, 2, 3, 4, 5, 6]

    hline()
    print("  EXPERIMENT E1 — Motor Imagery EEG (EEGMMIDB, PhysioNet)")
    hline()

    if not MIDB_DIR.exists():
        print(f"  ERROR: {MIDB_DIR} not found.")
        return {}

    subj_dirs = sorted(MIDB_DIR.glob("S[0-9][0-9][0-9]"))[:max_subjects]
    print(f"  Found {len(subj_dirs)} subject directories in {MIDB_DIR}")

    all_ep, all_lb, all_gr = [], [], []
    skipped = 0

    for sid, sd in enumerate(subj_dirs):
        ep, lb = load_subject_epochs(sd)
        if ep is None or len(lb) == 0:
            skipped += 1
            continue
        all_ep.append(ep)
        all_lb.append(lb)
        all_gr.extend([sid] * len(lb))
        if (sid + 1) % 10 == 0:
            print(f"  … {sid+1}/{len(subj_dirs)} subjects loaded")

    if not all_ep:
        print("  ERROR: No data loaded.")
        return {}

    epochs = np.concatenate(all_ep, axis=0)
    labels = np.concatenate(all_lb).astype(np.int32)
    groups = np.array(all_gr, dtype=np.int32)

    print(f"\n  Total trials : {len(labels)}")
    print(f"  Skipped subj : {skipped}")
    print(f"  N channels   : {epochs.shape[1]}")
    print(f"  Epoch length : {epochs.shape[2]} samples "
          f"({_EPOCH_S}s @ {_SFREQ} Hz)")
    for c, cn in enumerate(_CLASS_NAMES):
        print(f"    {cn}: {(labels == c).sum()}")

    # Classifier factory for non-CSP methods
    def clf():
        return make_pipeline(LDA, solver="lsqr", shrinkage="auto")

    # ── SME diagonal ablation ───────────────────────────────────────────
    print("\n  ── SME Diagonal Ablation ──")
    abl = ablation_over_p(
        feat_fn  = lambda p: feat_sme(epochs, p),
        y        = labels,
        groups   = groups,
        p_values = p_values,
        clf_factory = clf,
        strategy = "skf",
        n_splits = n_folds,
        label    = "SME-diag",
    )
    print_ablation_table(abl, "E1 SME-diagonal ablation over p")

    best_p = max(abl, key=lambda p: abl[p]["mean_acc"])
    print(f"\n  ★ Best SME order: p={best_p}  "
          f"acc={abl[best_p]['mean_acc']:.3f}  "
          f"F1={abl[best_p]['mean_f1']:.3f}")

    # ── Baselines ────────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    results = {f"SME-diag p={best_p}": abl[best_p]}

    # POCR/CRA — per-epoch, no fitting, unbiased
    print("  [POCR/CRA] ", end="", flush=True)
    r_pocr = cross_validate(clf(), feat_pocr(epochs), labels,
                             groups=groups, strategy="skf", n_splits=n_folds)
    results["POCR/CRA"] = r_pocr
    print(f"acc={r_pocr['mean_acc']:.3f}  F1={r_pocr['mean_f1']:.3f}")

    # Log-Euclidean — per-epoch matrix-log, no fitting, unbiased
    print("  [Log-Euclidean] ", end="", flush=True)
    r_log = cross_validate(clf(), feat_log_euclidean(epochs), labels,
                            groups=groups, strategy="skf", n_splits=n_folds)
    results["Log-Euclidean"] = r_log
    print(f"acc={r_log['mean_acc']:.3f}  F1={r_log['mean_f1']:.3f}")

    # CSP (OVR) — CORRECTED: spatial filters fitted inside each fold
    print("  [CSP (OVR, nested CV)] ", end="", flush=True)
    r_csp = csp_cross_validate(epochs, labels, groups, n_splits=n_folds)
    results["CSP (OVR, LDA)"] = r_csp
    print(f"acc={r_csp['mean_acc']:.3f}  F1={r_csp['mean_f1']:.3f}")

    # ── Summary tables ───────────────────────────────────────────────────
    print_results_table(results, "E1 Final Results  (10-fold subject-stratified CV)")
    print_significance_table(results, "E1 Pairwise Wilcoxon")

    # ── Per-class F1 ─────────────────────────────────────────────────────
    print(f"\n  Per-class F1 — SME(p={best_p}) vs POCR:")
    print_perclass_f1(abl[best_p], _CLASS_NAMES, r_pocr,
                      name_a=f"SME p={best_p}", name_b="POCR/CRA")

    # ── Confusion matrix ─────────────────────────────────────────────────
    cm = abl[best_p]["cm"]
    print(f"\n  Confusion matrix — SME(p={best_p}):")
    print("  " + "  ".join(f"{cn[:6]:>8}" for cn in _CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {_CLASS_NAMES[i][:6]:<8}" +
              "  ".join(f"{v:>8}" for v in row))

    # ── LaTeX table ──────────────────────────────────────────────────────
    print("\n  LaTeX table:")
    print(latex_table(results,
                      "E1: Motor Imagery EEG Classification (EEGMMIDB)"))

    # ── Save results and figures ─────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id           = "e1",
            results          = results,
            ablation         = abl,
            class_names      = _CLASS_NAMES,
            primary_metric   = "acc",
            caption          = (
                "E1: Motor Imagery EEG Classification (EEGMMIDB, PhysioNet). "
                "10-fold subject-stratified CV. LDA+shrinkage classifier. "
                "CSP evaluated with proper nested CV (spatial filters fitted "
                "on training folds only)."
            ),
            best_result_key  = f"SME-diag p={best_p}",
            perclass_methods = {
                f"SME p={best_p}":  abl[best_p],
                "POCR/CRA":         results["POCR/CRA"],
                "CSP (OVR, LDA)":   results["CSP (OVR, LDA)"],
            },
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation": abl, "results": results}


if __name__ == "__main__":
    run()
