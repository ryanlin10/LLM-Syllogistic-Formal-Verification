#!/usr/bin/env python3
"""
Experiment 3: Monotonicity Test (Add Redundant Premise)

Logic is monotone: adding true but irrelevant premises to a valid argument
must not invalidate the conclusion.

Pick 10 random data points from stage0. For each:
  1. Run model on original premises → conclusion A
  2. Add one redundant, semantically unrelated premise
  3. Run model on extended premises → conclusion B
  4. Compare A vs B (ideally A == B) and both vs ground truth

Output: detailed comparison table in text + JSON.
"""

import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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
    REDUNDANT_PREMISES,
    STAGE0_ADAPTER,
)

OUT_DIR = Path(__file__).resolve().parent / "results"
OUT_DIR.mkdir(exist_ok=True)

N_EXAMPLES = 10
SEED = 44
MAX_TOKENS = 256
TEMPERATURE = 0.0

# Extended list of redundant premises for variety
ALL_REDUNDANT = REDUNDANT_PREMISES + [
    "the ocean is large",
    "metals conduct electricity",
    "the alphabet has twenty-six letters",
    "autumn follows summer",
    "cats are smaller than elephants",
]


def main():
    rng = random.Random(SEED)

    examples = load_stage0_data(n=N_EXAMPLES, seed=SEED, min_premises=1)

    originals = []    # (user_message, ground_truth)
    augmented = []    # (user_message_with_redundant, redundant_premise)

    for ex in examples:
        premises = [p["text"] for p in ex["premises"]]
        gt = extract_ground_truth_conclusion(ex)

        orig_msg = format_user_message(premises)
        originals.append((orig_msg, gt))

        # Pick a redundant premise (different from any existing premise text)
        available = [r for r in ALL_REDUNDANT if r not in premises]
        redundant = rng.choice(available if available else ALL_REDUNDANT)

        # Insert at a random position (not necessarily last)
        insert_pos = rng.randint(0, len(premises))
        extended = premises[:insert_pos] + [redundant] + premises[insert_pos:]
        aug_msg = format_user_message(extended)
        augmented.append((aug_msg, redundant, insert_pos))

    # Load predictor
    print(f"\nLoading model with stage0 adapter...")
    predictor = make_predictor(lora_adapter=STAGE0_ADAPTER)

    SYSTEM = "You are a logical reasoning assistant. Given the following premises, derive their valid conclusion."

    # Original inference
    print(f"Running inference on {N_EXAMPLES} original examples...")
    orig_responses = predictor.generate_batch(
        messages=[msg for msg, _ in originals],
        system_prompt=SYSTEM,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    # Augmented inference
    print(f"Running inference on {N_EXAMPLES} augmented examples...")
    aug_responses = predictor.generate_batch(
        messages=[msg for msg, _, _ in augmented],
        system_prompt=SYSTEM,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        top_p=0.95,
    )

    # Evaluate
    results = []
    monotone_preserved = 0  # A == B (redundant premise didn't change conclusion)
    both_correct = 0
    orig_correct_count = 0
    aug_correct_count = 0

    for i, (ex, (orig_msg, gt), (aug_msg, redundant, insert_pos), orig_resp, aug_resp) in enumerate(
        zip(examples, originals, augmented, orig_responses, aug_responses)
    ):
        orig_conc = parse_model_conclusion(orig_resp)
        aug_conc = parse_model_conclusion(aug_resp)

        orig_match = conclusions_match(orig_conc, gt)
        aug_match = conclusions_match(aug_conc, gt)
        monotone = conclusions_match(orig_conc, aug_conc, threshold=0.85)

        if monotone:
            monotone_preserved += 1
        if orig_match:
            orig_correct_count += 1
        if aug_match:
            aug_correct_count += 1
        if orig_match and aug_match:
            both_correct += 1

        results.append({
            "id": ex.get("id", str(i)),
            "n_original_premises": len(ex["premises"]),
            "premises": [p["text"] for p in ex["premises"]],
            "redundant_premise": redundant,
            "redundant_inserted_at_position": insert_pos,
            "ground_truth": gt,
            "original_model_output": orig_resp,
            "augmented_model_output": aug_resp,
            "original_conclusion": orig_conc,
            "augmented_conclusion": aug_conc,
            "original_correct": orig_match,
            "augmented_correct": aug_match,
            "monotonicity_preserved": monotone,
        })

    n = len(results)
    summary = {
        "n_examples": n,
        "original_accuracy": orig_correct_count / n,
        "augmented_accuracy": aug_correct_count / n,
        "monotonicity_rate": monotone_preserved / n,
        "both_correct": both_correct / n,
    }

    print(f"\n=== RESULTS ===")
    print(f"N examples: {n}")
    print(f"Original accuracy:    {orig_correct_count}/{n}")
    print(f"Augmented accuracy:   {aug_correct_count}/{n}")
    print(f"Monotonicity preserved: {monotone_preserved}/{n}")

    # Save
    with open(OUT_DIR / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Detailed text comparison table
    with open(OUT_DIR / "comparison_table.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT 3: MONOTONICITY TEST (ADD REDUNDANT PREMISE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Summary: {n} examples, Stage-0 model\n")
        f.write(f"  Original accuracy:            {orig_correct_count}/{n}\n")
        f.write(f"  Augmented accuracy:           {aug_correct_count}/{n}\n")
        f.write(f"  Monotonicity preserved (A≈B): {monotone_preserved}/{n}\n\n")
        f.write("Interpretation:\n")
        f.write("  - If monotonicity is high: adding irrelevant premises preserves conclusions\n")
        f.write("  - If monotonicity is low:  the model is distracted by irrelevant content\n\n")

        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(results):
            f.write(f"EXAMPLE {i+1}/{n}  (id={r['id']})\n")
            f.write("-" * 60 + "\n")
            f.write(f"Original premises ({r['n_original_premises']}):\n")
            for j, p in enumerate(r["premises"]):
                f.write(f"  [{j+1}] {p}\n")
            f.write(f"\nRedundant premise (inserted at position {r['redundant_inserted_at_position']}):\n")
            f.write(f"  >>> {r['redundant_premise']}\n")
            f.write(f"\nGround truth conclusion:\n")
            f.write(f"  {r['ground_truth']}\n")
            f.write(f"\nModel output — ORIGINAL premises:\n")
            f.write(f"  Conclusion: {r['original_conclusion']}\n")
            f.write(f"  Matches GT: {r['original_correct']}\n")
            f.write(f"\nModel output — WITH REDUNDANT premise:\n")
            f.write(f"  Conclusion: {r['augmented_conclusion']}\n")
            f.write(f"  Matches GT: {r['augmented_correct']}\n")
            f.write(f"\nMonotonicity preserved (orig ≈ augmented): {r['monotonicity_preserved']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    print(f"Table saved to {OUT_DIR}/comparison_table.txt")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Accuracy comparison
    ax = axes[0]
    bars = ax.bar(
        ["Original\nPremises", "With Redundant\nPremise"],
        [orig_correct_count / n, aug_correct_count / n],
        color=["#2196F3", "#FF9800"],
        width=0.4,
    )
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy: Original vs Augmented\n(n={n}, Stage-0 Model)")
    for bar, val in zip(bars, [orig_correct_count / n, aug_correct_count / n]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")

    # 2. Per-example monotonicity visual
    ax2 = axes[1]
    colors_by_status = []
    labels_for_legend = {}
    y_positions = list(range(n))

    for j, r in enumerate(results):
        if r["monotonicity_preserved"] and r["original_correct"]:
            c = "#4CAF50"
            label = "Monotone + Correct"
        elif r["monotonicity_preserved"] and not r["original_correct"]:
            c = "#8BC34A"
            label = "Monotone + Wrong"
        elif not r["monotonicity_preserved"] and r["original_correct"]:
            c = "#FF9800"
            label = "Non-monotone + Correct"
        else:
            c = "#F44336"
            label = "Non-monotone + Wrong"
        colors_by_status.append(c)
        labels_for_legend[label] = c

    ax2.barh(y_positions, [1] * n, color=colors_by_status, height=0.8)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"Ex {j+1}" for j in range(n)], fontsize=8)
    ax2.set_xticks([])
    ax2.set_title("Per-example Monotonicity Status")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in labels_for_legend.values()]
    ax2.legend(handles, list(labels_for_legend.keys()), loc="lower right", fontsize=8)

    plt.suptitle("Monotonicity Test: Adding Redundant Premise", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(OUT_DIR / "monotonicity_results.png"), dpi=150)
    plt.close()
    print(f"Plot saved to {OUT_DIR}/monotonicity_results.png")

    print(f"\nAll results saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
