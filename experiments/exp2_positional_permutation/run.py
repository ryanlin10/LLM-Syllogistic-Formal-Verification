#!/usr/bin/env python3
"""
Experiment 2: Positional Permutation

Pick 100 random data points from stage0 (with >=2 premises). For each:
  1. Run the stage0 model on premises in ORIGINAL order → conclusion A
  2. Randomly shuffle premise order
  3. Run model on SHUFFLED order → conclusion B
  4. Compare A and B with ground truth, and with each other

Metrics:
  - Original accuracy (A matches ground truth)
  - Permuted accuracy  (B matches ground truth)
  - Stability: % of examples where A == B (order-invariant output)
  - Accuracy consistency: same correctness before and after permutation

A well-generalising model should be highly stable (order of premises shouldn't
affect a deductively valid conclusion).
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_inference_utils import (
    load_stage0_data,
    extract_ground_truth_conclusion,
    format_user_message,
    parse_model_conclusion,
    conclusions_match,
    normalize_text,
    make_predictor,
    STAGE0_ADAPTER,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_EXAMPLES = 100
SEED = 43
MAX_TOKENS = 256
TEMPERATURE = 0.0
BATCH_SIZE = 8


def permute_premises(premises: list, rng: random.Random) -> list:
    """Return a random permutation of premises (always a different order if possible)."""
    if len(premises) == 1:
        return premises[:]
    perm = premises[:]
    while perm == premises:
        rng.shuffle(perm)
    return perm


def main():
    rng = random.Random(SEED)

    # Load data — require >=2 premises for permutation to be meaningful
    examples = load_stage0_data(n=N_EXAMPLES, seed=SEED, min_premises=2)

    originals = []   # (user_message, ground_truth, original_order)
    permuteds = []   # (user_message_permuted, permuted_indices)

    for ex in examples:
        premises = [p["text"] for p in ex["premises"]]
        gt = extract_ground_truth_conclusion(ex)

        orig_msg = format_user_message(premises)
        originals.append((orig_msg, gt, list(range(len(premises)))))

        perm_premises = permute_premises(premises, rng)
        # Track the permutation order
        perm_idx = [premises.index(p) for p in perm_premises]
        perm_msg = format_user_message(perm_premises)
        permuteds.append((perm_msg, perm_idx))

    # Load predictor
    print(f"\nLoading model with stage0 adapter...")
    predictor = make_predictor(lora_adapter=STAGE0_ADAPTER)

    # Batch inference — original order
    print(f"Running inference on {len(originals)} original-order examples...")
    all_orig_msgs = [msg for msg, _, _ in originals]
    all_orig_responses = []
    for i in range(0, len(all_orig_msgs), BATCH_SIZE):
        batch = all_orig_msgs[i:i+BATCH_SIZE]
        responses = predictor.generate_batch(
            messages=batch,
            system_prompt="You are a logical reasoning assistant. Given the following premises, derive their valid conclusion.",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
        )
        all_orig_responses.extend(responses)
        print(f"  Original batch {i//BATCH_SIZE + 1}/{(len(all_orig_msgs) + BATCH_SIZE - 1)//BATCH_SIZE} done")

    # Batch inference — permuted order
    print(f"Running inference on {len(permuteds)} permuted-order examples...")
    all_perm_msgs = [msg for msg, _ in permuteds]
    all_perm_responses = []
    for i in range(0, len(all_perm_msgs), BATCH_SIZE):
        batch = all_perm_msgs[i:i+BATCH_SIZE]
        responses = predictor.generate_batch(
            messages=batch,
            system_prompt="You are a logical reasoning assistant. Given the following premises, derive their valid conclusion.",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
        )
        all_perm_responses.extend(responses)
        print(f"  Permuted batch {i//BATCH_SIZE + 1}/{(len(all_perm_msgs) + BATCH_SIZE - 1)//BATCH_SIZE} done")

    # Evaluate
    results = []
    orig_correct = 0
    perm_correct = 0
    stable = 0       # A and B are the same
    consistent = 0   # correctness same before and after

    for i, (ex, (orig_msg, gt, orig_idx), (perm_msg, perm_idx), orig_resp, perm_resp) in enumerate(
        zip(examples, originals, permuteds, all_orig_responses, all_perm_responses)
    ):
        orig_conc = parse_model_conclusion(orig_resp)
        perm_conc = parse_model_conclusion(perm_resp)

        orig_match = conclusions_match(orig_conc, gt)
        perm_match = conclusions_match(perm_conc, gt)
        is_stable = conclusions_match(orig_conc, perm_conc, threshold=0.90)
        is_consistent = (orig_match == perm_match)

        if orig_match:
            orig_correct += 1
        if perm_match:
            perm_correct += 1
        if is_stable:
            stable += 1
        if is_consistent:
            consistent += 1

        results.append({
            "id": ex.get("id", str(i)),
            "n_premises": len(ex["premises"]),
            "premises": [p["text"] for p in ex["premises"]],
            "ground_truth": gt,
            "permutation": perm_idx,
            "original_model_output": orig_resp,
            "permuted_model_output": perm_resp,
            "original_conclusion": orig_conc,
            "permuted_conclusion": perm_conc,
            "original_correct": orig_match,
            "permuted_correct": perm_match,
            "stable": is_stable,
            "consistent": is_consistent,
        })

    n = len(results)
    orig_acc = orig_correct / n
    perm_acc = perm_correct / n
    stability = stable / n
    consistency = consistent / n

    summary = {
        "n_examples": n,
        "original_accuracy": orig_acc,
        "permuted_accuracy": perm_acc,
        "accuracy_delta": orig_acc - perm_acc,
        "stability": stability,
        "consistency": consistency,
        "both_correct": sum(1 for r in results if r["original_correct"] and r["permuted_correct"]) / n,
        "neither_correct": sum(1 for r in results if not r["original_correct"] and not r["permuted_correct"]) / n,
        "only_original_correct": sum(1 for r in results if r["original_correct"] and not r["permuted_correct"]) / n,
        "only_permuted_correct": sum(1 for r in results if not r["original_correct"] and r["permuted_correct"]) / n,
    }

    print(f"\n=== RESULTS ===")
    print(f"N examples: {n}")
    print(f"Original accuracy:   {orig_acc:.3f} ({orig_correct}/{n})")
    print(f"Permuted accuracy:   {perm_acc:.3f} ({perm_correct}/{n})")
    print(f"Accuracy delta:      {orig_acc - perm_acc:+.3f}")
    print(f"Stability (same output): {stability:.3f} ({stable}/{n})")
    print(f"Consistency (same correctness): {consistency:.3f} ({consistent}/{n})")

    # Save
    with open(OUT_DIR / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Text summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("=== Experiment 2: Positional Permutation ===\n\n")
        f.write(f"N examples: {n}\n")
        f.write(f"Original accuracy (original premise order): {orig_acc:.3f} ({orig_correct}/{n})\n")
        f.write(f"Permuted accuracy (shuffled premise order):  {perm_acc:.3f} ({perm_correct}/{n})\n")
        f.write(f"Accuracy delta (orig - perm):                {orig_acc - perm_acc:+.3f}\n")
        f.write(f"Stability (same conclusion output):           {stability:.3f} ({stable}/{n})\n")
        f.write(f"Consistency (same correctness pre/post):      {consistency:.3f} ({consistent}/{n})\n\n")
        f.write("Outcome breakdown:\n")
        f.write(f"  Both correct:            {summary['both_correct']:.3f}\n")
        f.write(f"  Neither correct:         {summary['neither_correct']:.3f}\n")
        f.write(f"  Only original correct:   {summary['only_original_correct']:.3f}\n")
        f.write(f"  Only permuted correct:   {summary['only_permuted_correct']:.3f}\n\n")
        f.write("Interpretation:\n")
        f.write("  - High stability (~1.0): model output is order-invariant (robust)\n")
        f.write("  - Low stability:  model is sensitive to premise order (positional bias)\n")
        f.write("  - High accuracy delta: model was relying on premise position\n\n")

        f.write("=== Sample Cases ===\n\n")
        for r in results[:10]:
            f.write(f"ID: {r['id']} | #premises={r['n_premises']} | permutation={r['permutation']}\n")
            f.write(f"  Ground truth:         {r['ground_truth']}\n")
            f.write(f"  Original conclusion:  {r['original_conclusion']} (correct={r['original_correct']})\n")
            f.write(f"  Permuted conclusion:  {r['permuted_conclusion']} (correct={r['permuted_correct']})\n")
            f.write(f"  Stable: {r['stable']}\n\n")

    # --- Plots ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Accuracy comparison
    ax = axes[0]
    bars = ax.bar(
        ["Original\nOrder", "Shuffled\nOrder"],
        [orig_acc, perm_acc],
        color=["#2196F3", "#9C27B0"],
        width=0.4,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Before vs After\nPremise Permutation")
    for bar, val in zip(bars, [orig_acc, perm_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()

    # 2. Stability and consistency
    ax2 = axes[1]
    metrics = {"Stability\n(same output)": stability,
               "Consistency\n(same correctness)": consistency}
    colors2 = ["#4CAF50", "#FF9800"]
    bars2 = ax2.bar(list(metrics.keys()), list(metrics.values()), color=colors2, width=0.4)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Rate")
    ax2.set_title("Stability and Consistency\nAcross Permutations")
    for bar, val in zip(bars2, metrics.values()):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                 f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")

    # 3. Outcome breakdown (stacked bar)
    ax3 = axes[2]
    categories = ["Both\ncorrect", "Neither\ncorrect", "Only orig\ncorrect", "Only perm\ncorrect"]
    values = [summary["both_correct"], summary["neither_correct"],
              summary["only_original_correct"], summary["only_permuted_correct"]]
    colors3 = ["#4CAF50", "#F44336", "#2196F3", "#9C27B0"]
    bars3 = ax3.bar(categories, values, color=colors3, width=0.5)
    ax3.set_ylim(0, max(values) * 1.3)
    ax3.set_ylabel("Fraction")
    ax3.set_title("Correctness Outcome Breakdown")
    for bar, val in zip(bars3, values):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.005,
                 f"{val:.3f}", ha="center", fontsize=10)

    plt.suptitle("Positional Permutation Experiment (n=100, Stage-0 Model)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {OUT_DIR}/accuracy_comparison.png")

    # Stability by number of premises
    stability_by_n = {}
    for r in results:
        np_ = r["n_premises"]
        if np_ not in stability_by_n:
            stability_by_n[np_] = []
        stability_by_n[np_].append(int(r["stable"]))

    counts = sorted(stability_by_n.keys())
    stab_vals = [np.mean(stability_by_n[c]) for c in counts]
    n_vals = [len(stability_by_n[c]) for c in counts]

    fig2, ax4 = plt.subplots(figsize=(8, 5))
    bars4 = ax4.bar(counts, stab_vals, color="#673AB7", alpha=0.8)
    ax4.set_xlabel("Number of premises")
    ax4.set_ylabel("Stability (same conclusion across permutations)")
    ax4.set_title("Permutation Stability by Number of Premises")
    for bar, val, nb in zip(bars4, stab_vals, n_vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.2f}\n(n={nb})", ha="center", fontsize=8)
    ax4.set_ylim(0, 1.15)
    ax4.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "stability_by_premise_count.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {OUT_DIR}/stability_by_premise_count.png")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
