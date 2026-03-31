"""
exp1_motor_imagery.py  ─  Experiment E1: Motor Imagery EEG
===========================================================
Dataset : PhysioNet EEGMMIDB
          physionet.org/content/eegmmidb/1.0.0/  (no login required)
          midb/S001/ … S109/ — 14 runs per subject as EDF+ files

Task    : 4-class motor imagery
          class 0 = left-fist imagery    (runs R04, R08, R12  →  annotation T1)
          class 1 = right-fist imagery   (runs R04, R08, R12  →  annotation T2)
          class 2 = both-fists imagery   (runs R06, R10, R14  →  annotation T1)
          class 3 = both-feet imagery    (runs R06, R10, R14  →  annotation T2)

Channels: 22-channel sensorimotor subset (10-10 system) — 22×22 covariance.
Filter  : 8-30 Hz (μ + β bands), 4th-order Butterworth IIR.
Epochs  : 4-second trial epochs, onset-locked, 640 samples at 160 Hz.

SME     : sme_diagonal(p=1..6) on the 22×22 epoch covariance → 22p features.

CV      : 10-fold subject-stratified (109 subjects ≫ n for LOSO).

Baselines:
  · CSP (Common Spatial Patterns) — 6 filters × 4 OVR classes = 24 features
  · Log-Euclidean (matrix log upper-triangle → PCA to 88 dims)
  · POCR/CRA (21 angular coordinates + BD of θ_21)

Ablation  : p ∈ {1, 2, 3, 4, 5, 6}
Statistics: Wilcoxon signed-rank SME(best-p) vs every baseline.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
from pathlib import Path

# ── local imports ──────────────────────────────────────────────────────
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, pocr_features,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table,
    print_significance_table, print_perclass_f1,
    wilcoxon_paired, bootstrap_ci, latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = _HERE  # scripts and data are in the same directory
MIDB_DIR = BASE_DIR / "midb"

# ── constants ──────────────────────────────────────────────────────────
# Run indices and their imagery class mappings (1-indexed run numbers)
# R04,R08,R12 → left(T1=0) vs right(T2=1) fist imagery
# R06,R10,R14 → both-fists(T1=2) vs both-feet(T2=3) imagery
_LR_RUNS  = [4, 8, 12]
_BF_RUNS  = [6, 10, 14]

_SFREQ    = 160          # Hz
_EPOCH_S  = 4.0          # seconds per trial
_EPOCH_T  = int(_EPOCH_S * _SFREQ)   # 640 samples

# 22 sensorimotor channels (10-10 names without trailing dots)
_MOTOR_CH = [
    "Fc3", "Fc1", "Fcz", "Fc2", "Fc4",
    "C5",  "C3",  "C1",  "Cz",  "C2",  "C4",  "C6",
    "Cp3", "Cp1", "Cpz", "Cp2", "Cp4",
    "P1",  "Pz",  "P2",  "O1",  "Oz",
]

_CLASS_NAMES = ["Left-fist", "Right-fist", "Both-fists", "Both-feet"]


# ── helpers ────────────────────────────────────────────────────────────

def _pick_motor_channels(ch_names: list[str]) -> list[str]:
    """
    Match available EDF channel names to the 22 sensorimotor channels.
    EDF files use names like 'Fc3.', 'C3..' etc. (dot-padded to 4 chars).
    Falls back to the first 22 EEG channels if matching is poor.
    """
    wanted = {c.lower() for c in _MOTOR_CH}
    matched = [c for c in ch_names
               if c.lower().rstrip(". ") in wanted]
    return matched


def load_subject_epochs(subj_dir: Path):
    """
    Load all 4-class motor imagery epochs for one subject.

    Returns
    -------
    epochs : (n_trials, n_ch, T) float32   or None
    labels : (n_trials,) int32             or None
    """
    try:
        import mne
    except ImportError:
        raise ImportError("pip install mne")

    name = subj_dir.name          # e.g. "S001"
    epochs_out, labels_out = [], []

    run_cfgs = (
        [(_LR_RUNS, {"T1": 0, "T2": 1})] +
        [(_BF_RUNS, {"T1": 2, "T2": 3})]
    )

    for run_nums, lmap in run_cfgs:
        for rn in run_nums:
            edf = subj_dir / f"{name}R{rn:02d}.edf"
            if not edf.exists():
                continue
            try:
                raw = mne.io.read_raw_edf(str(edf), preload=True,
                                          verbose=False)
            except Exception:
                continue

            # Pick channels
            motor = _pick_motor_channels(raw.ch_names)
            if len(motor) < 10:          # fall back to first 22 EEG
                eeg_idx = mne.pick_types(raw.info, eeg=True, verbose=False)
                motor   = [raw.ch_names[i] for i in eeg_idx[:22]]
            if not motor:
                continue

            raw.pick_channels(motor, verbose=False)
            raw.filter(8.0, 30.0, verbose=False)

            try:
                events, eid = mne.events_from_annotations(raw, verbose=False)
            except Exception:
                continue

            # Map annotation string → class label
            code_to_cls = {}
            for desc, code in eid.items():
                for ann_key, cls in lmap.items():
                    if ann_key in desc:
                        code_to_cls[code] = cls

            if not code_to_cls:
                continue

            sfreq_r = raw.info["sfreq"]
            n_pts   = int(_EPOCH_S * sfreq_r)
            sig     = raw.get_data()   # (N_ch, T_total)

            for ev in events:
                onset, _, code = ev
                if code not in code_to_cls:
                    continue
                end = onset + n_pts
                if end > sig.shape[1]:
                    continue
                ep = sig[:, onset:end].astype(np.float32)
                epochs_out.append(ep)
                labels_out.append(code_to_cls[code])

    if not epochs_out:
        return None, None

    return np.stack(epochs_out), np.array(labels_out, dtype=np.int32)


# ── feature extractors ─────────────────────────────────────────────────

def _epoch_cov(ep: np.ndarray) -> np.ndarray:
    """ep: (N_ch, T) → (N_ch, N_ch) covariance."""
    return compute_cov(ep.T)


def feat_sme(epochs: np.ndarray, p: int) -> np.ndarray:
    """SME diagonal for each epoch. Returns (n, p·N_ch)."""
    return np.array([sme_diagonal(_epoch_cov(ep), p_max=p)
                     for ep in epochs], dtype=np.float32)


def feat_pocr(epochs: np.ndarray) -> np.ndarray:
    """POCR feature vector (N_ch-1 angles + BD) per epoch."""
    return np.array([pocr_features(_epoch_cov(ep))
                     for ep in epochs], dtype=np.float32)


def feat_log_euclidean(epochs: np.ndarray) -> np.ndarray:
    """Upper-triangle of matrix-log of epoch covariance."""
    N  = epochs.shape[1]
    ix = np.triu_indices(N)
    feats = []
    for ep in epochs:
        C = _epoch_cov(ep)
        w, v = np.linalg.eigh(C)
        w    = np.maximum(w, 1e-10)
        logC = (v * np.log(w)[np.newaxis, :]) @ v.T
        feats.append(logC[ix])
    return np.array(feats, dtype=np.float32)


def feat_csp_ovr(epochs: np.ndarray, labels: np.ndarray,
                 n_comp: int = 6) -> np.ndarray:
    """
    One-vs-rest CSP using MNE.  Returns (n, n_comp × 4) log-variance features.
    Falls back to per-channel variance if MNE unavailable.
    """
    try:
        from mne.decoding import CSP
    except ImportError:
        return epochs.var(axis=-1).astype(np.float32)

    feats = np.zeros((len(epochs), n_comp * 4), dtype=np.float32)
    for cls in range(4):
        y_bin = (labels == cls).astype(int)
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            continue
        try:
            csp = CSP(n_components=n_comp, log=True, norm_trace=False)
            f   = csp.fit_transform(epochs, y_bin)
            feats[:, cls * n_comp : (cls + 1) * n_comp] = f
        except Exception:
            pass
    return feats


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(max_subjects: int = 50,
        p_values: list[int] | None = None,
        n_folds:  int = 10) -> dict:

    if p_values is None:
        p_values = [1, 2, 3, 4, 5, 6]

    hline()
    print("  EXPERIMENT E1 ─ Motor Imagery EEG (EEGMMIDB, PhysioNet)")
    hline()

    if not MIDB_DIR.exists():
        print(f"  ERROR: {MIDB_DIR} not found."); return {}

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
        print("  ERROR: No data loaded. Verify MNE is installed and data paths."); return {}

    epochs  = np.concatenate(all_ep, axis=0)
    labels  = np.concatenate(all_lb).astype(np.int32)
    groups  = np.array(all_gr, dtype=np.int32)

    print(f"\n  Total trials : {len(labels)}")
    print(f"  Skipped subj : {skipped}")
    print(f"  N channels   : {epochs.shape[1]}")
    print(f"  Epoch length : {epochs.shape[2]} samples ({_EPOCH_S}s @ {_SFREQ}Hz)")
    for c, cn in enumerate(_CLASS_NAMES):
        print(f"    {cn}: {(labels == c).sum()}")

    # ── Classifier ─────────────────────────────────────────────────────
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

    def clf():
        return make_pipeline(LDA, solver="lsqr", shrinkage="auto")

    # ── SME ablation ────────────────────────────────────────────────────
    print("\n  ── SME Diagonal Ablation ──")
    abl = ablation_over_p(
        feat_fn=lambda p: feat_sme(epochs, p),
        y=labels, groups=groups, p_values=p_values,
        clf_factory=clf, strategy="skf", n_splits=n_folds,
        label="SME-diag",
    )
    print_ablation_table(abl, "E1 SME-diagonal ablation over p")

    best_p = max(abl, key=lambda p: abl[p]["mean_acc"])
    print(f"\n  ★ Best SME order: p={best_p}  "
          f"acc={abl[best_p]['mean_acc']:.3f}  "
          f"F1={abl[best_p]['mean_f1']:.3f}")

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    results = {f"SME-diag p={best_p}": abl[best_p]}

    print("  [POCR/CRA] ", end="", flush=True)
    X_pocr   = feat_pocr(epochs)
    r_pocr   = cross_validate(clf(), X_pocr, labels, groups=groups,
                              strategy="skf", n_splits=n_folds)
    results["POCR/CRA"] = r_pocr
    print(f"acc={r_pocr['mean_acc']:.3f}  F1={r_pocr['mean_f1']:.3f}")

    print("  [Log-Euclidean] ", end="", flush=True)
    X_log    = feat_log_euclidean(epochs)
    r_log    = cross_validate(clf(), X_log, labels, groups=groups,
                              strategy="skf", n_splits=n_folds)
    results["Log-Euclidean"] = r_log
    print(f"acc={r_log['mean_acc']:.3f}  F1={r_log['mean_f1']:.3f}")

    print("  [CSP (OVR)] ", end="", flush=True)
    X_csp    = feat_csp_ovr(epochs, labels)
    r_csp    = cross_validate(clf(), X_csp, labels, groups=groups,
                              strategy="skf", n_splits=n_folds)
    results["CSP (OVR, LDA)"] = r_csp
    print(f"acc={r_csp['mean_acc']:.3f}  F1={r_csp['mean_f1']:.3f}")

    # ── Summary tables ──────────────────────────────────────────────────
    print_results_table(results, "E1 Final Results  (10-fold subject-stratified CV)")
    print_significance_table(results, "E1 Pairwise Wilcoxon")

    # ── Per-class F1 ────────────────────────────────────────────────────
    print(f"\n  Per-class F1 — SME(p={best_p}) vs POCR:")
    print_perclass_f1(abl[best_p], _CLASS_NAMES, r_pocr,
                      name_a=f"SME p={best_p}", name_b="POCR/CRA")

    # ── Confusion matrix ────────────────────────────────────────────────
    cm = abl[best_p]["cm"]
    print(f"\n  Confusion matrix — SME(p={best_p}):")
    print("  " + "  ".join(f"{cn[:6]:>8}" for cn in _CLASS_NAMES))
    for i, row in enumerate(cm):
        print(f"  {_CLASS_NAMES[i][:6]:<8}" + "  ".join(f"{v:>8}" for v in row))

    # ── LaTeX table ─────────────────────────────────────────────────────
    print("\n  LaTeX table:")
    print(latex_table(results, "E1: Motor Imagery EEG Classification (EEGMMIDB)"))

    # ── Save results & figures ───────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id="e1",
            results=results,
            ablation=abl,
            class_names=_CLASS_NAMES,
            primary_metric="acc",
            caption="E1: Motor Imagery EEG Classification (EEGMMIDB, PhysioNet). "
                    "10-fold subject-stratified CV. LDA+shrinkage classifier.",
            best_result_key=f"SME-diag p={best_p}",
            perclass_methods={
                f"SME p={best_p}": abl[best_p],
                "POCR/CRA":        results["POCR/CRA"],
                "CSP (OVR, LDA)":  results["CSP (OVR, LDA)"],
            },
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation": abl, "results": results}


if __name__ == "__main__":
    run()
