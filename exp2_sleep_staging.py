"""
exp2_sleep_staging.py  ─  Experiment E2: Sleep Stage Classification
=====================================================================
Dataset : PhysioNet Sleep-EDF Database Expanded
          physionet.org/content/sleep-edfx/1.0.0/  (no login required)
          sleep_edf/sleep-cassette/  ← SC4{ss}{N}E0-PSG.edf
                                        SC4{ss}{N}EC-Hypnogram.edf
          sleep_edf/sleep-telemetry/ ← ST7{ss}{N}J0-PSG.edf
                                        ST7{ss}{N}JP-Hypnogram.edf

Channels: EEG Fpz-Cz, EEG Pz-Oz, EOG horizontal, EMG submental → N=4
Sampling: 100 Hz  |  Epoch: 30 seconds (3000 samples)

Task    : 5-class sleep staging (AASM-equivalent)
          0=Wake  1=N1  2=N2  3=N3 (N3+N4 merged)  4=REM

SME     : One 4×4 covariance per 30-s epoch  → sme_diagonal(p=1..6) = 24 features.
          Sub-epoch trajectory: 6 × 5-second windows inside each 30-s epoch
            → trajectory_stats(SME per sub-window) = 4×24 = 96 features.
          Full feature vector: 120 per epoch.

CV      : 10-fold subject-stratified.

Baselines:
  · Spectral power δ,θ,α,σ,β per channel (4ch × 5 bands = 20 features)
  · Hjorth parameters (activity, mobility, complexity) per channel (12 features)
  · POCR/CRA: mean/std of 3 angular coords + BD(θ₃) = 7 features

Ablation  : p ∈ {1,2,3,4,5,6}
Statistics: Wilcoxon, per-class F1 (N2 vs N3 is the key boundary).
"""

from __future__ import annotations
import os, sys, warnings, re
import numpy as np
from pathlib import Path
from scipy.signal import welch, butter, sosfiltfilt

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from sme_core import (
    compute_cov, sme_diagonal, trajectory_stats, pocr_features,
    make_pipeline, cross_validate, ablation_over_p,
    print_results_table, print_ablation_table, print_significance_table,
    print_perclass_f1, wilcoxon_paired, latex_table, hline,
)

warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────
BASE_DIR   = _HERE   # scripts and data are in the same directory
SLEEP_DIR  = BASE_DIR / "sleep_edf"
SC_DIR     = SLEEP_DIR / "sleep-cassette"
ST_DIR     = SLEEP_DIR / "sleep-telemetry"

# ── constants ──────────────────────────────────────────────────────────
_SFREQ     = 100          # Hz
_EPOCH_S   = 30           # seconds
_EPOCH_T   = _SFREQ * _EPOCH_S    # 3000
_SUBWIN_S  = 5            # sub-epoch window seconds
_SUBWIN_T  = _SFREQ * _SUBWIN_S   # 500
_N_SUBWIN  = _EPOCH_S // _SUBWIN_S  # 6

# Annotation → class mapping
_STAGE_MAP = {
    "Sleep stage W": 0, "W": 0,
    "Sleep stage 1": 1, "N1": 1,
    "Sleep stage 2": 2, "N2": 2,
    "Sleep stage 3": 3, "N3": 3,
    "Sleep stage 4": 3, "N4": 3,   # merge SWS
    "Sleep stage R": 4, "R": 4, "REM": 4,
}

_CHAN_KEYWORDS = ["fpz", "pz", "eog", "emg"]  # partial match, lower-case
_CHAN_NAMES    = ["EEG-Fpz", "EEG-Pz", "EOG", "EMG"]
_STAGE_NAMES  = ["Wake", "N1", "N2", "N3", "REM"]

_SPECTRAL_BANDS = [("δ", 0.5, 4), ("θ", 4, 8), ("α", 8, 13),
                   ("σ", 11, 16), ("β", 16, 30)]


# ── loaders ────────────────────────────────────────────────────────────

def _pick_channels(ch_names: list[str]) -> list[str]:
    """Match channels by keyword; return up to 4 channel names."""
    result = []
    for kw in _CHAN_KEYWORDS:
        for c in ch_names:
            if kw in c.lower() and c not in result:
                result.append(c)
                break
    # fall back: first 4
    if len(result) < 2:
        result = ch_names[:4]
    return result[:4]


def _load_psg_pair(psg_path: Path, hyp_path: Path):
    """
    Load one PSG/Hypnogram pair.
    Returns (data: (n_epochs, N_ch, T), labels: (n_epochs,)) or (None, None).
    """
    try:
        import mne
    except ImportError:
        raise ImportError("pip install mne")

    try:
        raw = mne.io.read_raw_edf(str(psg_path), preload=True, verbose=False)
        ann = mne.read_annotations(str(hyp_path))
    except Exception as e:
        return None, None

    raw.set_annotations(ann, emit_warning=False)
    sf  = raw.info["sfreq"]
    npt = int(_EPOCH_S * sf)

    # Pick 4 channels
    ch = _pick_channels(raw.ch_names)
    raw.pick_channels(ch, verbose=False)
    raw.resample(_SFREQ, verbose=False)
    sf  = _SFREQ
    npt = _EPOCH_T

    sig = raw.get_data()   # (N_ch, T_total)

    epochs_list, labels_list = [], []
    for annot in raw.annotations:
        desc = annot["description"].strip()
        if desc not in _STAGE_MAP:
            continue
        stage  = _STAGE_MAP[desc]
        onset  = int(annot["onset"] * sf)
        end    = onset + npt
        if end > sig.shape[1] or (end - onset) < npt:
            continue
        ep = sig[:, onset:end].astype(np.float32)
        epochs_list.append(ep)
        labels_list.append(stage)

    if not epochs_list:
        return None, None

    return np.stack(epochs_list), np.array(labels_list, dtype=np.int32)


def _find_hyp(psg_path: Path) -> Path | None:
    """Find the matching hypnogram EDF for a given PSG EDF."""
    name = psg_path.name
    # SC4001E0-PSG.edf → SC4001EC-Hypnogram.edf
    hyp_name = re.sub(r"(E\d|J\d)-PSG\.edf$",
                      lambda m: m.group(1).rstrip("0123456789") + "C-Hypnogram.edf"
                      if "E" in m.group(1) else
                      m.group(1).rstrip("0123456789") + "P-Hypnogram.edf",
                      name)
    hyp = psg_path.parent / hyp_name
    if hyp.exists():
        return hyp
    # glob fallback
    stem    = re.sub(r"-PSG\.edf$", "", name)
    pattern = stem.rstrip("0") + "*Hypnogram.edf"
    cands   = list(psg_path.parent.glob(pattern))
    return cands[0] if cands else None


def load_dataset(edf_dir: Path, max_subjects: int = 39):
    """
    Load all PSG pairs from a directory.
    Returns (data, labels, subjects) or (None, None, None).
    """
    psg_files = sorted(edf_dir.glob("*PSG.edf"))[:max_subjects * 2]
    # deduplicate subjects (take first night only for cassette)
    seen_subj = set()
    psg_filtered = []
    for p in psg_files:
        m = re.search(r"(\d{2})", p.name)
        subj = m.group(1) if m else p.name
        if subj not in seen_subj:
            seen_subj.add(subj)
            psg_filtered.append(p)
        if len(psg_filtered) >= max_subjects:
            break

    all_data, all_lb, all_sub = [], [], []
    for sid, psg in enumerate(psg_filtered):
        hyp = _find_hyp(psg)
        if hyp is None:
            continue
        data, lb = _load_psg_pair(psg, hyp)
        if data is None:
            continue
        all_data.append(data)
        all_lb.append(lb)
        all_sub.extend([sid] * len(lb))
        print(f"    loaded {psg.name}  ({len(lb)} epochs)", flush=True)

    if not all_data:
        return None, None, None

    return (np.concatenate(all_data, axis=0),
            np.concatenate(all_lb).astype(np.int32),
            np.array(all_sub, dtype=np.int32))


# ── feature extractors ─────────────────────────────────────────────────

def feat_sme(epochs: np.ndarray, p: int) -> np.ndarray:
    """
    30-s epoch → 4×4 covariance → SME diagonal (p·4 features)
    + trajectory_stats over 6 sub-windows (4·p·4 more).
    Total: 5·p·4 = 20p.
    """
    N   = epochs.shape[1]
    dim = 5 * p * N
    X   = np.zeros((len(epochs), dim), dtype=np.float32)

    for i, ep in enumerate(epochs):
        # Whole-epoch SME
        C_ep   = compute_cov(ep.T)
        f_ep   = sme_diagonal(C_ep, p_max=p)   # (p·N,)

        # Sub-epoch trajectory
        sub_f  = []
        for w in range(_N_SUBWIN):
            s  = w * _SUBWIN_T
            Cw = compute_cov(ep[:, s : s + _SUBWIN_T].T)
            sub_f.append(sme_diagonal(Cw, p_max=p))
        traj   = np.stack(sub_f)               # (6, p·N)
        f_traj = trajectory_stats(traj, include_bd=False)  # 4·p·N

        X[i] = np.concatenate([f_ep, f_traj])
    return X


def feat_pocr(epochs: np.ndarray) -> np.ndarray:
    """POCR feature per epoch: 3 angles + BD = 4 features."""
    return np.array([pocr_features(compute_cov(ep.T))
                     for ep in epochs], dtype=np.float32)


def _bandpower(sig: np.ndarray, lo: float, hi: float) -> float:
    f, psd = welch(sig, fs=_SFREQ, nperseg=min(256, len(sig)))
    idx    = (f >= lo) & (f <= hi)
    return float(psd[idx].mean()) if idx.any() else 0.0


def feat_spectral(epochs: np.ndarray) -> np.ndarray:
    """5 spectral bands × 4 channels = 20 features per epoch."""
    N   = epochs.shape[1]
    out = np.zeros((len(epochs), len(_SPECTRAL_BANDS) * N), dtype=np.float32)
    for i, ep in enumerate(epochs):
        col = 0
        for _, lo, hi in _SPECTRAL_BANDS:
            for ch in range(N):
                out[i, col] = _bandpower(ep[ch], lo, hi)
                col += 1
    return out


def _hjorth_params(sig: np.ndarray) -> tuple[float, float, float]:
    act  = float(np.var(sig))
    d1   = np.diff(sig)
    mob  = float(np.sqrt(np.var(d1) / (act + 1e-12)))
    d2   = np.diff(d1)
    cplx = float(np.sqrt(np.var(d2) / (np.var(d1) + 1e-12)) / (mob + 1e-12))
    return act, mob, cplx


def feat_hjorth(epochs: np.ndarray) -> np.ndarray:
    """Activity, Mobility, Complexity per channel → 12 features."""
    N   = epochs.shape[1]
    out = np.zeros((len(epochs), 3 * N), dtype=np.float32)
    for i, ep in enumerate(epochs):
        for ch in range(N):
            a, m, c = _hjorth_params(ep[ch])
            out[i, ch * 3 : (ch + 1) * 3] = [a, m, c]
    return out


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(max_subjects: int = 39,
        p_values: list[int] | None = None,
        n_folds:  int = 10) -> dict:

    if p_values is None:
        p_values = [1, 2, 3, 4, 5, 6]

    hline()
    print("  EXPERIMENT E2 ─ Sleep Stage Classification (Sleep-EDF-X, PhysioNet)")
    hline()

    # Try sleep-cassette first (larger, preferred)
    edf_dir = SC_DIR if SC_DIR.exists() else ST_DIR
    if not edf_dir.exists():
        print(f"  ERROR: {SLEEP_DIR} not found."); return {}

    print(f"  Loading from: {edf_dir}  (max {max_subjects} subjects)")
    data, labels, subjects = load_dataset(edf_dir, max_subjects)
    if data is None:
        print("  ERROR: No data loaded."); return {}

    print(f"\n  Subjects : {len(np.unique(subjects))}")
    print(f"  Epochs   : {len(labels)}")
    for i, sn in enumerate(_STAGE_NAMES):
        print(f"    {sn}: {(labels==i).sum()}")

    # Balance: down-sample Wake (always heavily over-represented)
    rng    = np.random.default_rng(42)
    n_min  = np.bincount(labels)[1:].min()          # minority non-Wake class
    n_w    = min((labels == 0).sum(), n_min * 5)     # keep ≤5× minority
    keep   = []
    for cls in range(5):
        idx = np.where(labels == cls)[0]
        n   = n_w if cls == 0 else len(idx)
        keep.extend(rng.choice(idx, min(n, len(idx)), replace=False).tolist())
    keep     = np.array(sorted(keep))
    data     = data[keep]; labels = labels[keep]; subjects = subjects[keep]
    print(f"\n  After Wake-balancing: {len(labels)} epochs")
    for i, sn in enumerate(_STAGE_NAMES):
        print(f"    {sn}: {(labels==i).sum()}")

    # ── Classifier ─────────────────────────────────────────────────────
    from sklearn.ensemble import RandomForestClassifier as RF
    def clf():
        return make_pipeline(RF, n_estimators=200, max_depth=12,
                             class_weight="balanced", n_jobs=-1,
                             random_state=42)

    # ── SME ablation ────────────────────────────────────────────────────
    print("\n  ── SME Ablation (epoch + sub-epoch trajectory) ──")
    abl = ablation_over_p(
        feat_fn=lambda p: feat_sme(data, p),
        y=labels, groups=subjects, p_values=p_values,
        clf_factory=clf, strategy="skf", n_splits=n_folds,
        label="SME-epoch",
    )
    print_ablation_table(abl, "E2 SME ablation over p")

    best_p = max(abl, key=lambda p: abl[p]["mean_f1"])
    print(f"\n  ★ Best order p={best_p}  "
          f"F1={abl[best_p]['mean_f1']:.3f}  "
          f"acc={abl[best_p]['mean_acc']:.3f}")

    # ── Baselines ───────────────────────────────────────────────────────
    print("\n  ── Baselines ──")
    results = {f"SME p={best_p}": abl[best_p]}

    for fname, fextract in [
        ("POCR/CRA",          feat_pocr),
        ("Spectral (5 bands)", feat_spectral),
        ("Hjorth params",      feat_hjorth),
    ]:
        print(f"  [{fname}] ", end="", flush=True)
        X = fextract(data)
        r = cross_validate(clf(), X, labels, groups=subjects,
                           strategy="skf", n_splits=n_folds)
        results[fname] = r
        print(f"acc={r['mean_acc']:.3f}  F1={r['mean_f1']:.3f}")

    # ── Summary ─────────────────────────────────────────────────────────
    print_results_table(results, "E2 Final Results  (10-fold, macro-F1 primary)")
    print_significance_table(results, "E2 Pairwise Wilcoxon")

    # ── Per-class F1 with N2/N3 focus ───────────────────────────────────
    print(f"\n  Per-class F1 — SME(p={best_p}) vs POCR/CRA:")
    print_perclass_f1(abl[best_p], _STAGE_NAMES, results["POCR/CRA"],
                      name_a=f"SME p={best_p}", name_b="POCR/CRA")

    # ── Key hypothesis check: N2 vs N3 F1 ───────────────────────────────
    from sklearn.metrics import f1_score as _f1
    for cls, sn in enumerate(_STAGE_NAMES):
        f_sme  = _f1(abl[best_p]["y_true"], abl[best_p]["y_pred"],
                     labels=[cls], average="macro", zero_division=0)
        f_spec = _f1(results["Spectral (5 bands)"]["y_true"],
                     results["Spectral (5 bands)"]["y_pred"],
                     labels=[cls], average="macro", zero_division=0)
        marker = "  ← KEY" if cls in (2, 3) else ""
        print(f"    {sn:<5}: SME={f_sme:.3f}  Spectral={f_spec:.3f}{marker}")

    print("\n  LaTeX table:")
    print(latex_table(results, "E2: Sleep Stage Classification (Sleep-EDF Expanded)"))

    # ── Save results & figures ───────────────────────────────────────────
    try:
        from sme_results import save_experiment
        save_experiment(
            exp_id="e2",
            results=results,
            ablation=abl,
            class_names=_STAGE_NAMES,
            primary_metric="f1",
            caption="E2: Sleep Stage Classification (Sleep-EDF Expanded, PhysioNet). "
                    "10-fold subject-stratified CV. Random Forest classifier.",
            best_result_key=f"SME p={best_p}",
            perclass_methods={
                f"SME p={best_p}":          abl[best_p],
                "POCR/CRA":                 results["POCR/CRA"],
                "Spectral (5 bands)":        results["Spectral (5 bands)"],
            },
            highlight_cls=[2, 3],   # N2 and N3 — the key boundary
        )
    except Exception as e:
        print(f"  (save_experiment failed: {e})")

    return {"ablation": abl, "results": results}


if __name__ == "__main__":
    run()
