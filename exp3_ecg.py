"""
exp3_ecg.py  ─  Experiment E3: ECG Arrhythmia Classification (MIT-BIH)
========================================================================
Dataset : PhysioNet MIT-BIH Arrhythmia Database
          physionet.org/content/mitdb/1.0.0/  (no login required)
          mit-bih/100.dat, 100.hea, 100.atr, …  (47 subjects, 48 records)

Signal  : 2-ch ambulatory ECG (MLII + V1/V2/V5), 360 Hz, 11-bit
Segments: 150 pre-R + 180 post-R = 330 samples (~0.92 s per beat)

Task    : 5-class AAMI beat classification
          N=Normal  S=Supraventricular  V=Ventricular  F=Fusion  Q=Pacemaker

N = 2  →  2×2 covariance per beat.
POCR on a 2×2 gives exactly 1 angular coordinate  →  most controlled comparison.

SME-diagonal(p) gives 2p features.  Gain per order is analytically traceable.
SME-vech(p)     gives 3p features  (lossless for symmetric 2×2).

Ablation: p ∈ {1,2,3,4,5,6,7,8}  for both projections.
MP-deviation Δ_k reported per AAMI class (Table in paper).

CV      : 5-fold patient-stratified (standard for MIT-BIH).
Baselines: RR-interval HRV, wavelet energy (db4 / FFT fallback), POCR/CRA.
"""

from __future__ import annotations
import os, sys, warnings
import numpy as np
from pathlib import Path
from scipy import stats as sp_stats

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, sme_vech, pocr_features,
    mp_deviation, bhattacharyya_distance_gaussian,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table, print_significance_table,
    print_perclass_f1, wilcoxon_paired, bootstrap_ci, latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR   = _HERE   # scripts and data are in the same directory
MITBIH_DIR = BASE_DIR / "mit-bih"

# ── constants ──────────────────────────────────────────────────────────
_SFREQ   = 360
_PRE_R   = 150
_POST_R  = 180
_BEAT_T  = _PRE_R + _POST_R   # 330 samples

# Standard 48 MIT-BIH record IDs (those that typically ship in the database)
_ALL_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234",
]

# AAMI 5-class mapping: annotation symbol → class index
_AAMI_MAP = {
    # N: Normal + bundle branch + escape
    "N":0, "L":0, "R":0, "e":0, "j":0,
    # S: Supraventricular ectopic
    "A":1, "a":1, "J":1, "S":1,
    # V: Ventricular ectopic
    "V":2, "E":2,
    # F: Fusion
    "F":3,
    # Q: Pacemaker / unclassifiable
    "/":4, "f":4, "Q":4,
}
_AAMI_NAMES = ["N (Normal)", "S (SVE)", "V (VE)", "F (Fusion)", "Q (Pace)"]


# ── loader ─────────────────────────────────────────────────────────────

def load_mitbih(data_dir: Path):
    """
    Load and segment all available MIT-BIH records.

    Returns
    -------
    beats   : (n, _BEAT_T, 2) float32
    labels  : (n,) int32  — AAMI class index
    subjects: (n,) int32  — patient number (for CV stratification)
    """
    try:
        import wfdb
    except ImportError:
        raise ImportError("pip install wfdb")

    beats, labels, subjects = [], [], []

    available = [r for r in _ALL_RECORDS
                 if (data_dir / f"{r}.dat").exists()]
    if not available:
        # Try loading whatever .dat files exist
        available = [p.stem for p in sorted(data_dir.glob("*.dat"))
                     if p.stem.isdigit()]

    print(f"  Found {len(available)} records in {data_dir}")

    for rec in available:
        rec_path = str(data_dir / rec)
        try:
            record = wfdb.rdrecord(rec_path)
            ann    = wfdb.rdann(rec_path, "atr")
        except Exception as e:
            print(f"    skip {rec}: {e}")
            continue

        sig = record.p_signal.astype(np.float32)   # (T, 2) or (T, 1)
        if sig.ndim == 1 or sig.shape[1] < 2:
            sig = np.column_stack([sig[:, 0], sig[:, 0]])

        for samp, sym in zip(ann.sample, ann.symbol):
            if sym not in _AAMI_MAP:
                continue
            s, e = samp - _PRE_R, samp + _POST_R
            if s < 0 or e > sig.shape[0]:
                continue
            beats.append(sig[s:e])
            labels.append(_AAMI_MAP[sym])
            subjects.append(int(rec))

    return (np.array(beats,    dtype=np.float32),
            np.array(labels,   dtype=np.int32),
            np.array(subjects, dtype=np.int32))


# ── feature extractors ─────────────────────────────────────────────────

def _cov(beat: np.ndarray) -> np.ndarray:
    """beat (330, 2) → 2×2 covariance."""
    return compute_cov(beat)


def feat_sme_diag(beats: np.ndarray, p: int) -> np.ndarray:
    """SME diagonal: (n, 2p)."""
    return np.array([sme_diagonal(_cov(b), p_max=p)
                     for b in beats], dtype=np.float32)


def feat_sme_vech(beats: np.ndarray, p: int) -> np.ndarray:
    """SME vech (upper triangle), lossless: (n, 3p) for 2×2."""
    return np.array([sme_vech(_cov(b), p_max=p)
                     for b in beats], dtype=np.float32)


def feat_pocr(beats: np.ndarray) -> np.ndarray:
    """POCR for 2×2: 1 angle + BD = 2 features."""
    return np.array([pocr_features(_cov(b))
                     for b in beats], dtype=np.float32)


def feat_rr(beats: np.ndarray, sfreq: float = _SFREQ) -> np.ndarray:
    """
    R-peak based RR-interval features within the beat segment.
    5 features: mean-RR, std-RR, min-RR, max-RR, pNN50.
    """
    from scipy.signal import find_peaks
    out = np.zeros((len(beats), 5), dtype=np.float32)
    for i, b in enumerate(beats):
        ecg   = b[:, 0]
        peaks, _ = find_peaks(ecg, distance=int(0.15 * sfreq),
                              height=np.percentile(ecg, 60))
        if len(peaks) < 2:
            out[i] = 0.0
            continue
        rr_ms = np.diff(peaks) / sfreq * 1000.0
        nn50  = np.sum(np.abs(np.diff(rr_ms)) > 50)
        out[i] = [rr_ms.mean(), rr_ms.std(),
                  rr_ms.min(), rr_ms.max(),
                  nn50 / max(len(rr_ms) - 1, 1)]
    return out


def feat_wavelet(beats: np.ndarray) -> np.ndarray:
    """
    db4 wavelet coefficient energies (4 levels × 2 channels = 8+ features).
    Falls back to FFT band energies if PyWavelets not installed.
    """
    try:
        import pywt
        def _wt_energy(sig):
            coeffs = pywt.wavedec(sig, "db4", level=4)
            return [np.sum(c ** 2) for c in coeffs]   # 5 values
        return np.array([
            np.concatenate([_wt_energy(b[:, 0]), _wt_energy(b[:, 1])])
            for b in beats], dtype=np.float32)
    except ImportError:
        # FFT band energies as fallback
        def _fft_bands(sig):
            f = np.abs(np.fft.rfft(sig))
            n = len(f)
            bands = np.array_split(f, 5)
            return [b.mean() for b in bands]
        return np.array([
            np.concatenate([_fft_bands(b[:, 0]), _fft_bands(b[:, 1])])
            for b in beats], dtype=np.float32)


# ── Marchenko-Pastur deviation table ──────────────────────────────────

def mp_deviation_table(beats: np.ndarray, labels: np.ndarray,
                       p_max: int = 6) -> None:
    hline("─")
    print(f"  Marchenko–Pastur deviation  Δ_k = Tr(C^k)/N − E_MP[λ^k]")
    print(f"  (N=2, T={_BEAT_T}, γ={2/_BEAT_T:.4f}; near-zero ⟹ noise-like)")
    hline("─")
    header = f"  {'Class':<20} " + "  ".join(f"Δ_{k}" for k in range(1, p_max+1))
    print(header)
    hline("─")
    for cls in range(5):
        idx    = np.where(labels == cls)[0]
        if len(idx) == 0: continue
        sample = idx[:min(500, len(idx))]
        deltas = np.array([mp_deviation(_cov(beats[i]), T=_BEAT_T, p_max=p_max)
                           for i in sample])
        md     = deltas.mean(axis=0)
        row    = "  ".join(f"{d:>+7.4f}" for d in md)
        print(f"  {_AAMI_NAMES[cls]:<20} {row}")
    hline("─")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(p_values: list[int] | None = None,
        n_folds:  int = 5) -> dict:

    if p_values is None:
        p_values = list(range(1, 9))

    hline()
    print("  EXPERIMENT E3 ─ ECG Arrhythmia Classification (MIT-BIH, PhysioNet)")
    hline()

    if not MITBIH_DIR.exists():
        print(f"  ERROR: {MITBIH_DIR} not found."); return {}

    beats, labels, subjects = load_mitbih(MITBIH_DIR)
    if len(beats) == 0:
        print("  ERROR: No beats loaded. Ensure `wfdb` is installed."); return {}

    print(f"  Raw beats    : {len(labels)}")
    for i, an in enumerate(_AAMI_NAMES):
        print(f"    {an}: {(labels==i).sum()}")

    # Balance: downsample N, keep all minority classes
    rng   = np.random.default_rng(42)
    n_min = np.bincount(labels)[1:].max()
    n_n   = min((labels == 0).sum(), n_min * 5)
    keep  = []
    for cls in range(5):
        idx = np.where(labels == cls)[0]
        n   = n_n if cls == 0 else len(idx)
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    keep     = np.array(sorted(keep))
    beats    = beats[keep]; labels = labels[keep]; subjects = subjects[keep]

    print(f"\n  After balancing: {len(labels)} beats")
    for i, an in enumerate(_AAMI_NAMES): print(f"    {an}: {(labels==i).sum()}")

    # ── Classifier ─────────────────────────────────────────────────────
    from sklearn.svm import SVC
    def clf():
        return make_pipeline(SVC, kernel="rbf", C=10, gamma="scale",
                             class_weight="balanced", random_state=42)

    # ── SME diagonal ablation ───────────────────────────────────────────
    print("\n  ── SME Diagonal Ablation ──")
    abl_diag = ablation_over_p(
        feat_fn=lambda p: feat_sme_diag(beats, p),
        y=labels, groups=subjects, p_values=p_values,
        clf_factory=clf, strategy="skf", n_splits=n_folds,
        label="SME-diag",
    )
    print_ablation_table(abl_diag, "E3 SME-diagonal ablation")

    # ── SME vech ablation ───────────────────────────────────────────────
    print("\n  ── SME Vech (upper-triangle) Ablation ──")
    abl_vech = ablation_over_p(
        feat_fn=lambda p: feat_sme_vech(beats, p),
        y=labels, groups=subjects, p_values=[1, 2, 3, 4, 5, 6],
        clf_factory=clf, strategy="skf", n_splits=n_folds,
        label="SME-vech",
    )
    print_ablation_table(abl_vech, "E3 SME-vech ablation")

    best_pd = max(abl_diag, key=lambda p: abl_diag[p]["mean_f1"])
    best_pv = max(abl_vech, key=lambda p: abl_vech[p]["mean_f1"])

    results = {
        f"SME-diag p={best_pd}": abl_diag[best_pd],
        f"SME-vech p={best_pv}": abl_vech[best_pv],
    }

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    for fname, fext in [("POCR/CRA", feat_pocr),
                        ("RR-interval", feat_rr),
                        ("Wavelet energy", feat_wavelet)]:
        print(f"  [{fname}] ", end="", flush=True)
        X = fext(beats)
        r = cross_validate(clf(), X, labels, groups=subjects,
                           strategy="skf", n_splits=n_folds)
        results[fname] = r
        print(f"acc={r['mean_acc']:.3f}  F1={r['mean_f1']:.3f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print_results_table(results, "E3 Final Results  (5-fold patient-stratified CV)")
    print_significance_table(results, "E3 Pairwise Wilcoxon")

    print(f"\n  Per-class F1 — SME-diag(p={best_pd}) vs POCR/CRA:")
    print_perclass_f1(abl_diag[best_pd], _AAMI_NAMES, results["POCR/CRA"],
                      name_a=f"SME-diag p={best_pd}", name_b="POCR/CRA")

    # ── MP deviation ────────────────────────────────────────────────────
    mp_deviation_table(beats, labels)

    # ── Hypothesis check: monotone accuracy in p ────────────────────────
    print("\n  Monotonicity check (Acc vs p, SME-diagonal):")
    for p, r in sorted(abl_diag.items()):
        bar = "█" * int(r["mean_acc"] * 40)
        print(f"    p={p}: {r['mean_acc']:.3f}  {bar}")

    print("\n  LaTeX table:")
    print(latex_table(results, "E3: ECG Arrhythmia Classification (MIT-BIH)"))

    # ── Compute MP delta matrix for saving ───────────────────────────────
    import numpy as _np
    _p_max_mp = 6
    mp_delta = _np.zeros((5, _p_max_mp))
    for cls in range(5):
        idx = _np.where(labels == cls)[0][:500]
        if len(idx) == 0: continue
        mp_delta[cls] = _np.array([
            mp_deviation(_cov(beats[i]), T=_BEAT_T, p_max=_p_max_mp)
            for i in idx]).mean(axis=0)

    # ── Save results & figures ───────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id="e3",
            results=results,
            ablation=abl_diag,
            ablation_b=abl_vech,
            label_a="SME-diagonal",
            label_b="SME-vech",
            class_names=_AAMI_NAMES,
            primary_metric="f1",
            caption="E3: ECG Arrhythmia Classification (MIT-BIH, PhysioNet). "
                    "5-fold patient-stratified CV. SVM-RBF classifier.",
            best_result_key=f"SME-diag p={best_pd}",
            perclass_methods={
                f"SME-diag p={best_pd}": abl_diag[best_pd],
                "POCR/CRA":              results["POCR/CRA"],
                "Wavelet energy":        results["Wavelet energy"],
            },
            mp_delta=mp_delta,
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation_diag": abl_diag, "ablation_vech": abl_vech, "results": results}


if __name__ == "__main__":
    run()
