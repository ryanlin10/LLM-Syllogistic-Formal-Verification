#!/usr/bin/env python3
"""
Master runner for all experiments.

Execution order:
  1. Exp 4 (LoRA SVD) - no GPU needed, runs quickly
  2. Inference experiments 1+2+3 (single vLLM load)
  3. Exp 5 (Probing classifiers, transformers)

Usage:
  python3 experiments/run_all.py
  python3 experiments/run_all.py --skip-inference   # skip exps 1-3
  python3 experiments/run_all.py --skip-probing     # skip exp 5
  python3 experiments/run_all.py --only-svd         # only exp 4
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS_DIR = Path(__file__).resolve().parent


def run_script(script_path: Path, label: str) -> bool:
    print(f"\n{'='*70}")
    print(f"  RUNNING: {label}")
    print(f"  Script:  {script_path}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"\n  COMPLETED in {elapsed/60:.1f} min: {label}")
        return True
    else:
        print(f"\n  FAILED (exit {result.returncode}) after {elapsed/60:.1f} min: {label}")
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-inference", action="store_true",
                        help="Skip inference experiments (1, 2, 3)")
    parser.add_argument("--skip-probing", action="store_true",
                        help="Skip probing classifier experiment (5)")
    parser.add_argument("--only-svd", action="store_true",
                        help="Only run SVD analysis (experiment 4)")
    args = parser.parse_args()

    results = {}

    # Exp 4: LoRA SVD analysis (no model loading, fast)
    ok = run_script(EXPERIMENTS_DIR / "exp4_lora_svd" / "run.py", "Exp 4: LoRA SVD Analysis")
    results["exp4_lora_svd"] = ok

    if args.only_svd:
        print("\nDone (--only-svd flag set).")
        return

    # Exps 1+2+3: Inference experiments (single vLLM load)
    if not args.skip_inference:
        ok = run_script(
            EXPERIMENTS_DIR / "run_inference_experiments.py",
            "Exps 1+2+3: Premise Perturbation + Permutation + Monotonicity"
        )
        results["exp1_premise_perturbation"] = ok
        results["exp2_positional_permutation"] = ok
        results["exp3_monotonicity"] = ok

    # Exp 5: Probing classifiers (transformers, heavy GPU)
    if not args.skip_probing:
        ok = run_script(EXPERIMENTS_DIR / "exp5_probing" / "run.py",
                        "Exp 5: Probing Classifiers")
        results["exp5_probing"] = ok

    # Summary
    print(f"\n{'='*70}")
    print("  FINAL STATUS")
    print(f"{'='*70}")
    all_ok = True
    for exp, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {exp:40s}: {status}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n  All experiments completed successfully.")
    else:
        print("\n  Some experiments failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
