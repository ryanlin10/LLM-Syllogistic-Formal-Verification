#!/usr/bin/env python3
"""
Experiment 1: Premise Perturbation

Pick 100 random data points from stage0. For each:
  1. Run the stage0 model on original premises → predicted conclusion
  2. Perturb one premise (entity/relation/predicate substitution)
  3. Run model on perturbed premises → new predicted conclusion
  4. Compare accuracy and sensitivity

Metrics:
  - Original accuracy: model conclusion matches ground truth
  - Post-perturbation accuracy: model still matches ground truth (should drop)
  - Sensitivity: model conclusion changed after perturbation (model responds to premise changes)
"""

import json
import random
import sys
import os
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
    perturb_one_premise,
    make_predictor,
    STAGE0_ADAPTER,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_EXAMPLES = 100
SEED = 42
MAX_TOKENS = 256
TEMPERATURE = 0.0
BATCH_SIZE = 8


def main():
    rng = random.Random(SEED)

    # Load data
    examples = load_stage0_data(n=N_EXAMPLES, seed=SEED, min_premises=2)

    # Prepare original and perturbed inputs
    originals = []      # (user_message, ground_truth)
    perturbations = []  # (user_message_perturbed, perturb_info)

    for ex in examples:
        premises = [p["text"] for p in ex["premises"]]
        gt = extract_ground_truth_conclusion(ex)

        orig_msg = format_user_message(premises)
        originals.append((orig_msg, gt))

        perturbed_premises, pidx, orig_text, new_text = perturb_one_premise(premises, rng)
        pert_msg = format_user_message(perturbed_premises)
        perturbations.append((pert_msg, {
            "perturbed_premise_idx": pidx,
            "original_text": orig_text,
            "new_text": new_text,
        }))

    # Load predictor
    print(f"\nLoading model with stage0 adapter...")
    predictor = make_predictor(lora_adapter=STAGE0_ADAPTER)

    # Batch inference — original
    print(f"Running inference on {len(originals)} original examples...")
    all_orig_msgs = [msg for msg, _ in originals]
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

    # Batch inference — perturbed
    print(f"Running inference on {len(perturbations)} perturbed examples...")
    all_pert_msgs = [msg for msg, _ in perturbations]
    all_pert_responses = []
    for i in range(0, len(all_pert_msgs), BATCH_SIZE):
        batch = all_pert_msgs[i:i+BATCH_SIZE]
        responses = predictor.generate_batch(
            messages=batch,
            system_prompt="You are a logical reasoning assistant. Given the following premises, derive their valid conclusion.",
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
        )
        all_pert_responses.extend(responses)
        print(f"  Perturbed batch {i//BATCH_SIZE + 1}/{(len(all_pert_msgs) + BATCH_SIZE - 1)//BATCH_SIZE} done")

    # Evaluate
    results = []
    orig_correct = 0
    pert_correct = 0
    sensitive = 0  # conclusion changed after perturbation

    for i, (ex, (orig_msg, gt), (pert_msg, pinfo), orig_resp, pert_resp) in enumerate(
        zip(examples, originals, perturbations, all_orig_responses, all_pert_responses)
    ):
        orig_conc = parse_model_conclusion(orig_resp)
        pert_conc = parse_model_conclusion(pert_resp)

        orig_match = conclusions_match(orig_conc, gt)
        pert_match = conclusions_match(pert_conc, gt)
        changed = not conclusions_match(orig_conc, pert_conc, threshold=0.90)

        if orig_match:
            orig_correct += 1
        if pert_match:
            pert_correct += 1
        if changed:
            sensitive += 1

        results.append({
            "id": ex.get("id", str(i)),
            "premises": [p["text"] for p in ex["premises"]],
            "ground_truth": gt,
            "perturbed_premise_idx": pinfo["perturbed_premise_idx"],
            "original_premise_text": pinfo["original_text"],
            "perturbed_premise_text": pinfo["new_text"],
            "original_model_output": orig_resp,
            "perturbed_model_output": pert_resp,
            "original_conclusion": orig_conc,
            "perturbed_conclusion": pert_conc,
            "original_correct": orig_match,
            "perturbed_correct": pert_match,
            "conclusion_changed": changed,
        })

    n = len(results)
    orig_acc = orig_correct / n
    pert_acc = pert_correct / n
    sensitivity = sensitive / n

    summary = {
        "n_examples": n,
        "original_accuracy": orig_acc,
        "perturbed_accuracy": pert_acc,
        "accuracy_drop": orig_acc - pert_acc,
        "sensitivity": sensitivity,
        "correct_and_sensitive": sum(
            1 for r in results if r["original_correct"] and r["conclusion_changed"]
        ) / n,
    }

    print(f"\n=== RESULTS ===")
    print(f"N examples: {n}")
    print(f"Original accuracy:   {orig_acc:.3f} ({orig_correct}/{n})")
    print(f"Perturbed accuracy:  {pert_acc:.3f} ({pert_correct}/{n})")
    print(f"Accuracy drop:       {orig_acc - pert_acc:.3f}")
    print(f"Sensitivity (conclusion changed): {sensitivity:.3f} ({sensitive}/{n})")

    # Save results
    with open(OUT_DIR / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Write text summary
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("=== Experiment 1: Premise Perturbation ===\n\n")
        f.write(f"N examples: {n}\n")
        f.write(f"Original accuracy (model matches GT):     {orig_acc:.3f} ({orig_correct}/{n})\n")
        f.write(f"Post-perturbation accuracy (matches GT):  {pert_acc:.3f} ({pert_correct}/{n})\n")
        f.write(f"Accuracy drop (orig - perturbed):         {orig_acc - pert_acc:.3f}\n")
        f.write(f"Sensitivity (conclusion changed):          {sensitivity:.3f} ({sensitive}/{n})\n\n")
        f.write("Interpretation:\n")
        f.write(f"  - If sensitivity is HIGH (~1.0): model responds to premise changes (generalizes)\n")
        f.write(f"  - If sensitivity is LOW (~0.0): model may be memorizing conclusions\n")
        f.write(f"  - Accuracy drop shows how much premise perturbation hurts performance\n\n")

        # Show 5 example perturbations
        f.write("=== Sample Perturbation Cases ===\n\n")
        for r in results[:10]:
            f.write(f"ID: {r['id']}\n")
            f.write(f"  Ground truth: {r['ground_truth']}\n")
            f.write(f"  Perturbed premise [{r['perturbed_premise_idx']}]:\n")
            f.write(f"    Before: {r['original_premise_text']}\n")
            f.write(f"    After:  {r['perturbed_premise_text']}\n")
            f.write(f"  Original conclusion:  {r['original_conclusion']} (correct={r['original_correct']})\n")
            f.write(f"  Perturbed conclusion: {r['perturbed_conclusion']} (correct={r['perturbed_correct']})\n")
            f.write(f"  Conclusion changed: {r['conclusion_changed']}\n\n")

    # --- Plots ---
    # 1. Bar chart: accuracy before vs after
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    bars = ax.bar(
        ["Original\nAccuracy", "Post-Perturbation\nAccuracy"],
        [orig_acc, pert_acc],
        color=["#2196F3", "#F44336"],
        width=0.4,
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Conclusion Accuracy Before vs After\nPremise Perturbation")
    for bar, val in zip(bars, [orig_acc, pert_acc]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02, f"{val:.3f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.legend()

    # 2. Stacked breakdown: what happened to each example
    categories = {
        "Correct → Changed": sum(1 for r in results if r["original_correct"] and r["conclusion_changed"]),
        "Correct → Unchanged": sum(1 for r in results if r["original_correct"] and not r["conclusion_changed"]),
        "Wrong → Changed": sum(1 for r in results if not r["original_correct"] and r["conclusion_changed"]),
        "Wrong → Unchanged": sum(1 for r in results if not r["original_correct"] and not r["conclusion_changed"]),
    }
    ax2 = axes[1]
    colors2 = ["#4CAF50", "#8BC34A", "#FF9800", "#F44336"]
    wedges, texts, autotexts = ax2.pie(
        categories.values(),
        labels=categories.keys(),
        colors=colors2,
        autopct=lambda p: f"{p:.1f}%\n({int(round(p*n/100))})",
        startangle=90,
    )
    ax2.set_title("Breakdown of Example Outcomes\nAfter Premise Perturbation")

    plt.suptitle("Premise Perturbation Experiment (n=100, Stage-0 Model)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {OUT_DIR}/accuracy_comparison.png")

    # 3. Sensitivity by number of premises
    premise_counts = [len(ex["premises"]) for ex in examples]
    changed_by_count = {}
    for r, ex in zip(results, examples):
        c = len(ex["premises"])
        if c not in changed_by_count:
            changed_by_count[c] = []
        changed_by_count[c].append(int(r["conclusion_changed"]))

    counts = sorted(changed_by_count.keys())
    sens_by_count = [np.mean(changed_by_count[c]) for c in counts]
    n_by_count = [len(changed_by_count[c]) for c in counts]

    fig2, ax3 = plt.subplots(figsize=(8, 5))
    bars3 = ax3.bar(counts, sens_by_count, color="#9C27B0", alpha=0.8)
    ax3.set_xlabel("Number of premises")
    ax3.set_ylabel("Sensitivity (fraction where conclusion changed)")
    ax3.set_title("Sensitivity to Perturbation by Number of Premises")
    for bar, val, nb in zip(bars3, sens_by_count, n_by_count):
        ax3.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f"{val:.2f}\n(n={nb})", ha="center", fontsize=8)
    ax3.set_ylim(0, 1.1)
    ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "sensitivity_by_premise_count.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {OUT_DIR}/sensitivity_by_premise_count.png")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
