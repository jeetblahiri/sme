"""
run_all_experiments.py  ─  SME Framework: Master Experiment Runner
===================================================================
Runs all five experiments, collects results, runs unit tests on the
core library, prints a consolidated grand-summary table, and writes
all results to disk.

Usage
─────
  python run_all_experiments.py              # all 5 experiments
  python run_all_experiments.py --exp e3    # single experiment
  python run_all_experiments.py --quick     # reduced subjects + p-range
  python run_all_experiments.py --test-only # unit tests only

Expected layout (all scripts + data in same parent directory)
─────────────────────────────────────────────────────────────
  extending_POCR/
  ├── sme_exp/                  ← PUT THIS DIRECTORY HERE
  │   ├── sme_core.py
  │   ├── exp1_motor_imagery.py
  │   ├── exp2_sleep_staging.py
  │   ├── exp3_ecg.py
  │   ├── exp4_cwru.py
  │   ├── exp5_har.py
  │   └── run_all_experiments.py   ← this file
  ├── cwru/
  ├── midb/
  ├── mit-bih/
  ├── sleep_edf/
  └── uci_har/

Install
───────
  pip install numpy scipy scikit-learn mne wfdb
  pip install pywt      # optional: wavelet baseline in E3
"""

from __future__ import annotations
import argparse, sys, os, time, traceback
from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from sme_core import (
    compute_cov, sme_diagonal, sme_vech, sme_rowsum,
    compute_pocr, bhattacharyya_distance_gaussian,
    mp_deviation, trajectory_stats,
    wilcoxon_paired, bootstrap_ci,
    print_results_table, print_ablation_table, hline,
)


# ══════════════════════════════════════════════════════════════════════════════
# UNIT TESTS
# ══════════════════════════════════════════════════════════════════════════════

def test_sme_core() -> bool:
    """Run sanity checks on sme_core.py.  Returns True iff all pass."""
    print("\n" + "="*65)
    print("  UNIT TESTS — sme_core.py")
    print("="*65)
    passed = 0; failed = 0

    def ok(cond: bool, msg: str):
        nonlocal passed, failed
        if cond:
            print(f"  ✓ {msg}"); passed += 1
        else:
            print(f"  ✗ FAIL: {msg}"); failed += 1

    np.random.seed(0)
    N, T = 6, 500
    X = np.random.randn(T, N)
    C = compute_cov(X)

    # 1. SME p=1 equals diag(C)
    f1 = sme_diagonal(C, p_max=1)
    ok(f1.shape == (N,), f"SME-diag p=1 shape = ({N},)")
    ok(np.allclose(f1, np.diag(C), atol=1e-6), "SME-diag p=1 == diag(C)")

    # 2. SME p=2 equals diag(C²)
    f2 = sme_diagonal(C, p_max=2)[N:]
    ok(np.allclose(f2, np.diag(C @ C), atol=1e-6), "SME-diag p=2 == diag(C²)")

    # 3. SME p=3 equals diag(C³)
    f3 = sme_diagonal(C, p_max=3)[2*N:]
    ok(np.allclose(f3, np.diag(C @ C @ C), atol=1e-5), "SME-diag p=3 == diag(C³)")

    # 4. Row-sum at p=1 equals C @ 1
    rs = sme_rowsum(C, p_max=1)
    ok(np.allclose(rs, C @ np.ones(N), atol=1e-6), "SME-rowsum p=1 == C·1")

    # 5. Vech shape for N=6, p=2 is 2×(6×7/2) = 42
    vech = sme_vech(C, p_max=2)
    ok(vech.shape == (42,), f"SME-vech p=2 shape = (42,)")

    # 6. POCR all-positive
    pocr, thetas = compute_pocr(C)
    ok((pocr >= -1e-10).all(), "POCR in positive orthant")
    ok(thetas.shape == (N-1,), f"POCR theta shape = ({N-1},)")
    ok((thetas >= -1e-6).all() and (thetas <= 90+1e-6).all(),
       "POCR angles ∈ [0°, 90°]")

    # 7. BD from Gaussian ≈ 0 for Gaussian data
    data = np.random.randn(100_000) * 30 + 45
    bd   = bhattacharyya_distance_gaussian(data)
    ok(bd < 0.02, f"BD(Gaussian) ≈ 0  (got {bd:.4f})")

    # 8a. Verify Narayana formula gives known closed-form moments
    #     For MP(γ): m_1=1, m_2=1+γ, m_3=1+3γ+γ²
    from sme_core import _mp_moment
    g = 0.1
    ok(abs(_mp_moment(1, g) - 1.0) < 1e-10,
       f"MP k=1 = 1  (got {_mp_moment(1, g):.6f})")
    ok(abs(_mp_moment(2, g) - (1 + g)) < 1e-10,
       f"MP k=2 = 1+γ = {1+g:.4f}  (got {_mp_moment(2, g):.6f})")
    ok(abs(_mp_moment(3, g) - (1 + 3*g + g**2)) < 1e-10,
       f"MP k=3 = 1+3γ+γ² = {1+3*g+g**2:.4f}  (got {_mp_moment(3, g):.6f})")

    # 8b. MP deviation ≈ 0 on average for white noise
    #     Single-realisation deviation can reach ~0.2 (finite-sample variance);
    #     average over 50 runs is well within 0.10.
    deltas_avg = np.mean([
        mp_deviation(compute_cov(np.random.randn(T, N)), T=T, p_max=4)
        for _ in range(50)
    ], axis=0)
    ok(np.all(np.abs(deltas_avg) < 0.10),
       f"MP-deviation avg(50 runs) ≈ 0  (max|Δ|={np.abs(deltas_avg).max():.4f})")

    # 9. Trajectory stats shape
    traj = np.random.randn(10, 12)
    ts   = trajectory_stats(traj, include_bd=True)
    ok(ts.shape == (5*12,), f"trajectory_stats shape = (5×12,) = (60,)")

    # 10. Bootstrap CI is ordered
    scores = np.random.uniform(0.7, 0.9, 20)
    lo, hi = bootstrap_ci(scores)
    ok(lo < hi, f"bootstrap_ci: lo={lo:.3f} < hi={hi:.3f}")

    # 11. Wilcoxon detects significant difference
    a = np.random.uniform(0.7, 0.8, 20)
    b = a - 0.15
    res = wilcoxon_paired(a, b)
    ok(res["sig"], f"Wilcoxon detects Δ=0.15 difference  (p={res['pval']:.4f})")

    # 12. 2×2 POCR recovers single angle
    C22 = np.array([[2.0, 0.5], [0.5, 1.0]])
    pocr22, theta22 = compute_pocr(C22)
    ok(theta22.shape == (1,), "2×2 POCR yields 1 angular coordinate")

    print(f"\n  {passed} passed, {failed} failed")
    hline()
    return failed == 0


# ══════════════════════════════════════════════════════════════════════════════
# GRAND SUMMARY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _best_sme_key(results: dict) -> str | None:
    sme_keys = [k for k in results if "SME" in str(k)]
    if not sme_keys: return None
    return max(sme_keys, key=lambda k: results[k]["mean_acc"])


def grand_summary(all_results: dict, total_time_s: float) -> None:
    hline()
    print("  GRAND SUMMARY")
    hline()
    print(f"  {'Exp':<6} {'Best SME method':<30} {'SME Acc':>9} "
          f"{'POCR Acc':>10} {'Δ Acc':>8}  sig?")
    hline("─")

    for exp_id in ["E1", "E2", "E3", "E4", "E5"]:
        if exp_id not in all_results: continue
        res  = all_results[exp_id]["results"]
        bkey = _best_sme_key(res)
        pocr = next((r for k, r in res.items() if "POCR" in k), None)
        if not bkey or not pocr: continue

        r_sme  = res[bkey]
        delta  = r_sme["mean_acc"] - pocr["mean_acc"]
        wtest  = wilcoxon_paired(r_sme["fold_acc"], pocr["fold_acc"])
        arrow  = "↑*" if (wtest["sig"] and delta > 0) else ("↑" if delta > 0 else "↓")
        print(f"  {exp_id:<6} {bkey[:30]:<30} {r_sme['mean_acc']:>9.3f} "
              f"{pocr['mean_acc']:>10.3f} {delta:>+8.3f}  {arrow}")

    hline()
    print(f"  Total elapsed: {total_time_s/60:.1f} min")
    print("  (* Wilcoxon signed-rank p < 0.05)")
    hline()


def save_all_results(all_results: dict) -> None:
    """Write per-experiment result summaries to text files."""
    for exp_id, payload in all_results.items():
        fname = f"results_{exp_id.lower()}.txt"
        with open(fname, "w") as f:
            f.write(f"# SME Framework Results — {exp_id}\n")
            for method, r in payload["results"].items():
                lo, hi = bootstrap_ci(r["fold_acc"])
                f.write(f"{method}: "
                        f"acc={r['mean_acc']:.4f}±{r['std_acc']:.4f}  "
                        f"F1={r['mean_f1']:.4f}±{r['std_f1']:.4f}  "
                        f"CI=[{lo:.4f},{hi:.4f}]\n")
                f.write(f"  folds: {r['fold_acc'].tolist()}\n")
        print(f"  Saved {fname}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="SME Framework Experiments",
        formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--exp", choices=["e1","e2","e3","e4","e5","all"],
                   default="all",
                   help="Which experiment(s) to run")
    p.add_argument("--quick", action="store_true",
                   help="Reduced subjects (E1:20, E2:15) and p-max=4 for testing")
    p.add_argument("--p-max", type=int, default=6,
                   help="Maximum SME moment order to ablate (default 6)")
    p.add_argument("--n-folds", type=int, default=10,
                   help="CV folds for E1–E4 (E5 always LOSO=30)")
    p.add_argument("--test-only", action="store_true",
                   help="Run unit tests only, skip experiments")
    return p.parse_args()


def main():
    args   = parse_args()
    t0     = time.time()
    passed = test_sme_core()
    if not passed:
        print("\n  ⚠  Unit tests FAILED — fix sme_core.py before running experiments.")
        sys.exit(1)

    if args.test_only:
        return

    p_max    = 4 if args.quick else args.p_max
    n_folds  = 5 if args.quick else args.n_folds
    p_vals   = list(range(1, p_max + 1))

    print(f"\n  Running: {args.exp}  |  p_values={p_vals}  |  n_folds={n_folds}"
          + ("  [QUICK MODE]" if args.quick else ""))

    all_results: dict = {}

    # ── E1 ──────────────────────────────────────────────────────────────
    if args.exp in ("e1", "all"):
        try:
            from exp1_motor_imagery import run as run_e1
            payload = run_e1(
                max_subjects=20 if args.quick else 50,
                p_values=p_vals, n_folds=n_folds)
            if payload:
                all_results["E1"] = payload
        except Exception:
            print("  E1 failed:"); traceback.print_exc()

    # ── E2 ──────────────────────────────────────────────────────────────
    if args.exp in ("e2", "all"):
        try:
            from exp2_sleep_staging import run as run_e2
            payload = run_e2(
                max_subjects=15 if args.quick else 39,
                p_values=p_vals, n_folds=n_folds)
            if payload:
                all_results["E2"] = payload
        except Exception:
            print("  E2 failed:"); traceback.print_exc()

    # ── E3 ──────────────────────────────────────────────────────────────
    if args.exp in ("e3", "all"):
        try:
            from exp3_ecg import run as run_e3
            p_e3 = p_vals + ([7, 8] if not args.quick else [])
            payload = run_e3(p_values=p_e3,
                             n_folds=min(n_folds, 5))
            if payload:
                all_results["E3"] = payload
        except Exception:
            print("  E3 failed:"); traceback.print_exc()

    # ── E4 ──────────────────────────────────────────────────────────────
    if args.exp in ("e4", "all"):
        try:
            from exp4_cwru import run as run_e4
            payload = run_e4(p_values=p_vals, n_folds=n_folds)
            if payload:
                all_results["E4"] = payload
        except Exception:
            print("  E4 failed:"); traceback.print_exc()

    # ── E5 ──────────────────────────────────────────────────────────────
    if args.exp in ("e5", "all"):
        try:
            from exp5_har import run as run_e5
            p_e5 = [p for p in p_vals if p <= 5]
            payload = run_e5(p_values=p_e5)
            if payload:
                all_results["E5"] = payload
        except Exception:
            print("  E5 failed:"); traceback.print_exc()

    # ── Grand summary ────────────────────────────────────────────────────
    if all_results:
        grand_summary(all_results, time.time() - t0)
        save_all_results(all_results)

        # Write consolidated LaTeX table
        from sme_core import latex_table
        with open("results_latex.tex", "w") as f:
            for exp_id, payload in all_results.items():
                f.write(f"\n% === {exp_id} ===\n")
                f.write(latex_table(
                    payload["results"],
                    caption=f"Results for Experiment {exp_id}"
                ))
                f.write("\n")
        print("  Consolidated LaTeX tables → results_latex.tex")
    else:
        print("\n  No experiment results collected.")


if __name__ == "__main__":
    main()
