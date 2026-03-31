"""
exp4_cwru.py  ─  Experiment E4: Rotating Machinery Fault Diagnosis (CWRU)
=========================================================================
Dataset : Case Western Reserve University Bearing Dataset
          engineering.case.edu/bearingdatacenter  (no login required)
          cwru/100.mat, 105.mat, … — MATLAB format, 2-ch vibration at 12 kHz

Channels: Drive End (DE) + Fan End (FE) accelerometers → N=2
Segment : 1024 samples per segment (~85 ms), 50% overlap

Task    : 4-class fault classification
          0=Normal  1=Inner Race  2=Ball  3=Outer Race
          Sub-task: 3-class fault severity (0.007", 0.014", 0.021" diameter)

N = 2  →  2×2 covariance per segment — parallel to E3.
Key hypothesis: p=3,4 should dominate via kurtotic impact dynamics
(bearing faults produce impulsive, heavy-tailed signals).

SME     : sme_diagonal(p=1..6), single-segment and 10-segment super-window.
Trajectory: 10-segment super-window → trajectory_stats → kurtosis is key feature.

MP-deviation: Normal bearings should have Δ_k ≈ 0; faults Δ_k > 0.

CV      : 10-fold stratified.
Baselines: Statistical time-domain features, envelope spectrum features, POCR/CRA.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats
import scipy.io as sio

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, sme_trace, pocr_features,
    mp_deviation, trajectory_stats, bhattacharyya_distance_gaussian,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table, print_significance_table,
    print_perclass_f1, wilcoxon_paired, bootstrap_ci, latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR = _HERE  # scripts and data are in the same directory
CWRU_DIR = BASE_DIR / "cwru"

# ── constants ──────────────────────────────────────────────────────────
_SFREQ    = 12_000
_SEG_LEN  = 1024          # samples per segment
_HOP      = _SEG_LEN // 2  # 50% overlap
_SUPER    = 10             # super-window length in segments

# File-number → (fault_type, fault_size_thou, load_hp)
# Only Drive-End 12k data (the standard benchmark subset)
_FILE_MAP: dict[int, tuple[str, int, int]] = {
    # Normal baseline
    97:  ("N", 0, 0),  98:  ("N", 0, 1),  99:  ("N", 0, 2),  100: ("N", 0, 3),
    # Inner Race  7 mil
    105: ("IR", 7, 0), 106: ("IR", 7, 1), 107: ("IR", 7, 2), 108: ("IR", 7, 3),
    # Inner Race 14 mil
    169: ("IR", 14, 0), 170: ("IR", 14, 1), 171: ("IR", 14, 2), 172: ("IR", 14, 3),
    # Inner Race 21 mil
    209: ("IR", 21, 0), 210: ("IR", 21, 1), 211: ("IR", 21, 2), 212: ("IR", 21, 3),
    # Ball 7 mil
    118: ("B", 7, 0), 119: ("B", 7, 1), 120: ("B", 7, 2), 121: ("B", 7, 3),
    # Ball 14 mil
    185: ("B", 14, 0), 186: ("B", 14, 1), 187: ("B", 14, 2), 188: ("B", 14, 3),
    # Ball 21 mil
    222: ("B", 21, 0), 223: ("B", 21, 1), 224: ("B", 21, 2), 225: ("B", 21, 3),
    # Outer Race @6h  7 mil
    130: ("OR", 7, 0), 131: ("OR", 7, 1), 132: ("OR", 7, 2), 133: ("OR", 7, 3),
    # Outer Race @6h 14 mil
    197: ("OR", 14, 0), 198: ("OR", 14, 1), 199: ("OR", 14, 2), 200: ("OR", 14, 3),
    # Outer Race @6h 21 mil
    234: ("OR", 21, 0), 235: ("OR", 21, 1), 236: ("OR", 21, 2), 237: ("OR", 21, 3),
}

_CLS_MAP   = {"N": 0, "IR": 1, "B": 2, "OR": 3}
_CLS_NAMES = ["Normal", "Inner Race", "Ball", "Outer Race"]
_SEV_NAMES = ["7 mil", "14 mil", "21 mil"]


# ── loader ─────────────────────────────────────────────────────────────

def _extract_signal(mat: dict) -> np.ndarray | None:
    """
    Extract DE and FE time-series from a CWRU .mat dictionary.
    Returns (T, 2) float32, or (T, 2) with duplicated channel if FE absent.
    """
    de_key = fe_key = None
    for k in mat:
        if k.startswith("_"): continue
        kl = k.lower()
        if "de_time" in kl: de_key = k
        elif "fe_time" in kl: fe_key = k

    if de_key is None:
        # Fall back: any large 1-D array
        for k in mat:
            if k.startswith("_"): continue
            v = np.asarray(mat[k]).flatten()
            if v.size > 10_000:
                de_key = k; break
    if de_key is None:
        return None

    de = np.asarray(mat[de_key]).flatten().astype(np.float32)
    if fe_key and np.asarray(mat[fe_key]).flatten().size == de.size:
        fe = np.asarray(mat[fe_key]).flatten().astype(np.float32)
    else:
        fe = de.copy()   # single-channel fallback
    return np.column_stack([de, fe])


def load_cwru(data_dir: Path):
    """
    Load all available CWRU .mat files and segment them.

    Returns
    -------
    segs        : (n, _SEG_LEN, 2) float32
    labels      : (n,) int32        fault class 0-3
    fault_sizes : (n,) int32        0/7/14/21 mil
    file_ids    : (n,) int32        file number (for CV grouping)
    """
    mat_files = sorted(data_dir.glob("*.mat"))
    segs, labels, fault_sizes, file_ids = [], [], [], []

    for mf in mat_files:
        stem = mf.stem
        if not stem.isdigit(): continue
        fnum = int(stem)

        if fnum in _FILE_MAP:
            ftype, fsize, _ = _FILE_MAP[fnum]
        else:
            # Unknown file: skip (don't guess)
            print(f"    (skip unknown file: {mf.name})")
            continue

        try:
            mat = sio.loadmat(str(mf), squeeze_me=True)
        except Exception as e:
            print(f"    (cannot read {mf.name}: {e})")
            continue

        sig = _extract_signal(mat)
        if sig is None:
            print(f"    (no usable signal in {mf.name})")
            continue

        cls = _CLS_MAP[ftype]
        T   = sig.shape[0]
        for s in range(0, T - _SEG_LEN + 1, _HOP):
            segs.append(sig[s : s + _SEG_LEN])
            labels.append(cls)
            fault_sizes.append(fsize)
            file_ids.append(fnum)

    if not segs:
        return None, None, None, None

    return (np.array(segs,       dtype=np.float32),
            np.array(labels,     dtype=np.int32),
            np.array(fault_sizes,dtype=np.int32),
            np.array(file_ids,   dtype=np.int32))


# ── feature extractors ─────────────────────────────────────────────────

def feat_sme_single(segs: np.ndarray, p: int) -> np.ndarray:
    """Per-segment SME diagonal: (n, 2p)."""
    return np.array([sme_diagonal(compute_cov(s), p_max=p)
                     for s in segs], dtype=np.float32)


def feat_sme_traj(segs: np.ndarray, labels_full: np.ndarray, p: int):
    """
    Super-window trajectory features.
    Groups consecutive segments into blocks of _SUPER.
    Returns (X_traj, y_traj, ids_traj) where each row is one block.
    """
    n_blocks = len(segs) // _SUPER
    Xt, yt = [], []
    for b in range(n_blocks):
        block = segs[b * _SUPER : (b + 1) * _SUPER]   # (_SUPER, T, 2)
        traj  = np.array([sme_diagonal(compute_cov(s), p_max=p) for s in block])
        Xt.append(trajectory_stats(traj, include_bd=True))
        yt.append(int(sp_stats.mode(labels_full[b * _SUPER : (b+1)*_SUPER])[0]))
    return (np.array(Xt, dtype=np.float32),
            np.array(yt, dtype=np.int32))


def feat_stats(segs: np.ndarray) -> np.ndarray:
    """Classical time-domain features: mean, std, skew, kurt, RMS, crest factor per channel."""
    out = np.zeros((len(segs), 6 * 2), dtype=np.float32)
    for i, s in enumerate(segs):
        for ch in range(2):
            x    = s[:, ch]
            rms  = float(np.sqrt(np.mean(x**2)))
            peak = float(np.max(np.abs(x)))
            out[i, ch*6 : (ch+1)*6] = [
                x.mean(), x.std(),
                float(sp_stats.skew(x)),
                float(sp_stats.kurtosis(x, fisher=True)),
                rms,
                peak / (rms + 1e-10),   # crest factor
            ]
    return out


def feat_envelope_spectrum(segs: np.ndarray, n_features: int = 10) -> np.ndarray:
    """
    Envelope spectrum: Hilbert-transform magnitude FFT.
    Returns top-n_features peak amplitudes per channel.
    """
    from scipy.signal import hilbert
    out = np.zeros((len(segs), n_features * 2), dtype=np.float32)
    for i, s in enumerate(segs):
        for ch in range(2):
            env  = np.abs(hilbert(s[:, ch]))
            spec = np.abs(np.fft.rfft(env - env.mean()))
            spec = spec[:n_features]
            out[i, ch * n_features : (ch + 1) * n_features] = spec
    return out


def feat_pocr(segs: np.ndarray) -> np.ndarray:
    """POCR for 2×2: 1 angle + BD = 2 features."""
    return np.array([pocr_features(compute_cov(s))
                     for s in segs], dtype=np.float32)


# ── analysis ───────────────────────────────────────────────────────────

def mp_deviation_table(segs: np.ndarray, labels: np.ndarray,
                       p_max: int = 6) -> None:
    hline("─")
    print(f"  Marchenko–Pastur deviation  (N=2, T={_SEG_LEN}, γ={2/_SEG_LEN:.5f})")
    print(f"  Near-zero ⟹ noise-like covariance; positive ⟹ structured signal")
    hline("─")
    print(f"  {'Fault class':<16} " +
          "  ".join(f"Δ_{k}" for k in range(1, p_max+1)))
    hline("─")
    for cls, cn in enumerate(_CLS_NAMES):
        idx  = np.where(labels == cls)[0][:500]
        if len(idx) == 0: continue
        Δ    = np.array([mp_deviation(compute_cov(segs[i]), T=_SEG_LEN, p_max=p_max)
                         for i in idx])
        row  = "  ".join(f"{d:>+8.5f}" for d in Δ.mean(axis=0))
        print(f"  {cn:<16} {row}")
    hline("─")


def kurtosis_by_order_table(segs: np.ndarray, labels: np.ndarray,
                             p_max: int = 6) -> None:
    """
    Mean trajectory kurtosis of SME-diagonal at each order, per class.
    High kurtosis at p=3,4 confirms impulsive non-Gaussian fault signals.
    """
    hline("─")
    print(f"  Trajectory kurtosis of SME-diagonal by order p and fault class")
    hline("─")
    print(f"  {'Class':<16} " + "  ".join(f"p={k}" for k in range(1, p_max+1)))
    hline("─")
    for cls, cn in enumerate(_CLS_NAMES):
        idx = np.where(labels == cls)[0][:300]
        if len(idx) == 0: continue
        row = []
        for p in range(1, p_max+1):
            sf    = np.array([sme_diagonal(compute_cov(segs[i]), p_max=p)[-2:]
                              for i in idx])   # last order only (2 features)
            kurt  = sp_stats.kurtosis(sf, axis=0, fisher=True).mean()
            row.append(f"{kurt:>7.2f}")
        print(f"  {cn:<16} " + "  ".join(row))
    hline("─")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(p_values: list[int] | None = None,
        n_folds:  int = 10) -> dict:

    if p_values is None:
        p_values = [1, 2, 3, 4, 5, 6]

    hline()
    print("  EXPERIMENT E4 ─ Bearing Fault Diagnosis (CWRU)")
    hline()

    if not CWRU_DIR.exists():
        print(f"  ERROR: {CWRU_DIR} not found."); return {}

    print("  Loading CWRU .mat files …")
    segs, labels, fault_sizes, file_ids = load_cwru(CWRU_DIR)
    if segs is None:
        print("  ERROR: No data loaded."); return {}

    print(f"\n  Segments     : {len(labels)}")
    for i, cn in enumerate(_CLS_NAMES):
        print(f"    {cn}: {(labels==i).sum()}")

    # Balance classes
    rng   = np.random.default_rng(42)
    n_min = np.bincount(labels).min()
    keep  = np.concatenate([
        rng.choice(np.where(labels == c)[0], n_min, replace=False)
        for c in range(4)])
    segs       = segs[keep];       labels      = labels[keep]
    fault_sizes = fault_sizes[keep]; file_ids   = file_ids[keep]
    print(f"  After balancing: {len(labels)} segments ({n_min}/class)")

    # ── Classifier ─────────────────────────────────────────────────────
    from sklearn.svm import SVC
    def clf():
        return make_pipeline(SVC, kernel="rbf", C=50, gamma="scale",
                             class_weight="balanced", random_state=42)

    # ── SME single-segment ablation ─────────────────────────────────────
    print("\n  ── SME Single-Segment Ablation ──")
    abl_single = ablation_over_p(
        feat_fn=lambda p: feat_sme_single(segs, p),
        y=labels, groups=file_ids, p_values=p_values,
        clf_factory=clf, strategy="skf", n_splits=n_folds,
        label="SME-single",
    )
    print_ablation_table(abl_single, "E4 SME-single ablation over p")

    # ── SME super-window trajectory ablation ────────────────────────────
    print("\n  ── SME Trajectory (10-seg super-window) Ablation ──")
    abl_traj_results = {}
    for p in p_values:
        print(f"  [SME-traj p={p}] ", end="", flush=True)
        X_t, y_t = feat_sme_traj(segs, labels, p)
        if len(X_t) < 10:
            print("too few blocks"); continue
        r = cross_validate(clf(), X_t, y_t,
                           strategy="skf", n_splits=min(n_folds, len(X_t)//4))
        abl_traj_results[p] = r
        print(f"({X_t.shape[1]}d)  acc={r['mean_acc']:.3f}  F1={r['mean_f1']:.3f}")
    print_ablation_table(abl_traj_results, "E4 SME-trajectory ablation")

    best_ps = max(abl_single, key=lambda p: abl_single[p]["mean_acc"])
    best_pt = (max(abl_traj_results, key=lambda p: abl_traj_results[p]["mean_acc"])
               if abl_traj_results else best_ps)

    results = {f"SME-single p={best_ps}": abl_single[best_ps]}
    if abl_traj_results:
        results[f"SME-traj   p={best_pt}"] = abl_traj_results[best_pt]

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    for fname, fext in [("Statistical (6×2)", feat_stats),
                        ("Envelope spectrum", feat_envelope_spectrum),
                        ("POCR/CRA", feat_pocr)]:
        print(f"  [{fname}] ", end="", flush=True)
        X = fext(segs)
        r = cross_validate(clf(), X, labels, groups=file_ids,
                           strategy="skf", n_splits=n_folds)
        results[fname] = r
        print(f"acc={r['mean_acc']:.3f}  F1={r['mean_f1']:.3f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print_results_table(results, "E4 Final Results  (10-fold stratified CV)")
    print_significance_table(results, "E4 Pairwise Wilcoxon")

    print(f"\n  Per-class F1 — SME-single(p={best_ps}) vs POCR/CRA:")
    print_perclass_f1(abl_single[best_ps], _CLS_NAMES, results["POCR/CRA"],
                      name_a=f"SME-single p={best_ps}", name_b="POCR/CRA")

    # ── Theory-driven analyses ──────────────────────────────────────────
    mp_deviation_table(segs, labels)
    kurtosis_by_order_table(segs, labels)

    # ── Fault-severity sub-task ─────────────────────────────────────────
    sev_mask = fault_sizes > 0
    if sev_mask.sum() > 200:
        print("\n  ── Fault Severity Sub-task (3-class: 7/14/21 mil) ──")
        sev_labels = np.searchsorted([7, 14, 21], fault_sizes[sev_mask])
        sev_segs   = segs[sev_mask]
        sev_groups = file_ids[sev_mask]
        X_sev      = feat_sme_single(sev_segs, best_ps)
        r_sev      = cross_validate(clf(), X_sev, sev_labels,
                                    groups=sev_groups, strategy="skf",
                                    n_splits=min(5, n_folds))
        print(f"  Severity (SME p={best_ps}): "
              f"acc={r_sev['mean_acc']:.3f}  F1={r_sev['mean_f1']:.3f}")
        X_poc_sev  = feat_pocr(sev_segs)
        r_poc_sev  = cross_validate(clf(), X_poc_sev, sev_labels,
                                    groups=sev_groups, strategy="skf",
                                    n_splits=min(5, n_folds))
        print(f"  Severity (POCR):            "
              f"acc={r_poc_sev['mean_acc']:.3f}  F1={r_poc_sev['mean_f1']:.3f}")

    print("\n  LaTeX table:")
    print(latex_table(results, "E4: Bearing Fault Diagnosis (CWRU)"))

    # ── Compute MP delta matrix and kurtosis matrix for saving ───────────
    import numpy as _np
    _p_max_mp = min(6, max(p_values))
    mp_delta   = _np.zeros((4, _p_max_mp))
    kurt_mat   = _np.zeros((4, _p_max_mp))
    for cls in range(4):
        idx = _np.where(labels == cls)[0][:500]
        if len(idx) == 0: continue
        mp_delta[cls] = _np.array([
            mp_deviation(compute_cov(segs[i]), T=_SEG_LEN, p_max=_p_max_mp)
            for i in idx]).mean(axis=0)
        for pi, p in enumerate(range(1, _p_max_mp + 1)):
            sf = _np.array([sme_diagonal(compute_cov(segs[i]), p_max=p)[-2:]
                            for i in idx])
            kurt_mat[cls, pi] = float(sp_stats.kurtosis(
                sf, axis=0, fisher=True).mean())

    # ── Save results & figures ───────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id="e4",
            results=results,
            ablation=abl_single,
            class_names=_CLS_NAMES,
            primary_metric="acc",
            caption="E4: Bearing Fault Diagnosis (CWRU). "
                    "10-fold stratified CV. SVM-RBF classifier.",
            best_result_key=f"SME-single p={best_ps}",
            perclass_methods={
                f"SME-single p={best_ps}": abl_single[best_ps],
                "POCR/CRA":                results["POCR/CRA"],
                "Statistical (6×2)":       results["Statistical (6×2)"],
            },
            mp_delta=mp_delta,
            kurt_matrix=kurt_mat,
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation_single": abl_single,
            "ablation_traj": abl_traj_results,
            "results": results}


if __name__ == "__main__":
    run()
