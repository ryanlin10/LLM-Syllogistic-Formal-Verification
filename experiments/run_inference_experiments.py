#!/usr/bin/env python3
"""
Combined runner for inference-based experiments (1, 2, 3).

Loads the stage-0 model ONCE via vLLM, then runs all three experiments
sequentially to avoid the overhead of 3 separate model loads.

Saves all results to their respective experiment subfolders.
"""

import json
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
    REDUNDANT_PREMISES,
)

SYSTEM = (
    "You are a logical reasoning assistant. "
    "Given the following premises, derive their valid conclusion."
)
MAX_TOKENS = 128   # Conclusions are short (~10-30 tokens); 128 is generous
TEMPERATURE = 0.0
BATCH_SIZE = 16    # Larger outer batches; inner predictor uses batch_size=8

EXP1_OUT = Path(__file__).resolve().parent / "exp1_premise_perturbation" / "results"
EXP2_OUT = Path(__file__).resolve().parent / "exp2_positional_permutation" / "results"
EXP3_OUT = Path(__file__).resolve().parent / "exp3_monotonicity" / "results"
for d in [EXP1_OUT, EXP2_OUT, EXP3_OUT]:
    d.mkdir(exist_ok=True)

ALL_REDUNDANT = REDUNDANT_PREMISES + [
    "the ocean is large",
    "metals conduct electricity",
    "the alphabet has twenty-six letters",
    "autumn follows summer",
    "cats are smaller than elephants",
]


def batch_infer(predictor, messages, batch_size=BATCH_SIZE):
    responses = []
    for i in range(0, len(messages), batch_size):
        batch = messages[i:i+batch_size]
        r = predictor.generate_batch(
            messages=batch,
            system_prompt=SYSTEM,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            top_p=0.95,
        )
        responses.extend(r)
        print(f"  [{i+len(batch)}/{len(messages)}] done")
    return responses


def permute_premises(premises, rng):
    if len(premises) <= 1:
        return premises[:]
    perm = premises[:]
    while perm == premises:
        rng.shuffle(perm)
    return perm


# ---------------------------------------------------------------------------
# Prepare all inputs upfront
# ---------------------------------------------------------------------------

def prepare_all_inputs():
    rng1 = random.Random(42)
    rng2 = random.Random(43)
    rng3 = random.Random(44)

    # --- Exp 1: Premise perturbation (100 examples, min 2 premises) ---
    print("\n[EXP1] Loading data...")
    ex1 = load_stage0_data(n=100, seed=42, min_premises=2)
    exp1_orig_msgs, exp1_pert_msgs = [], []
    exp1_gts, exp1_pinfo = [], []
    for e in ex1:
        premises = [p["text"] for p in e["premises"]]
        gt = extract_ground_truth_conclusion(e)
        exp1_orig_msgs.append(format_user_message(premises))
        exp1_gts.append(gt)
        pert_p, pidx, orig_t, new_t = perturb_one_premise(premises, rng1)
        exp1_pert_msgs.append(format_user_message(pert_p))
        exp1_pinfo.append({"idx": pidx, "orig": orig_t, "new": new_t})

    # --- Exp 2: Positional permutation (100 examples, min 2 premises) ---
    print("[EXP2] Loading data...")
    ex2 = load_stage0_data(n=100, seed=43, min_premises=2)
    exp2_orig_msgs, exp2_perm_msgs = [], []
    exp2_gts, exp2_perm_idx = [], []
    for e in ex2:
        premises = [p["text"] for p in e["premises"]]
        gt = extract_ground_truth_conclusion(e)
        exp2_orig_msgs.append(format_user_message(premises))
        exp2_gts.append(gt)
        perm = permute_premises(premises, rng2)
        perm_idx = [premises.index(p) if p in premises else -1 for p in perm]
        exp2_perm_msgs.append(format_user_message(perm))
        exp2_perm_idx.append(perm_idx)

    # --- Exp 3: Monotonicity (10 examples) ---
    print("[EXP3] Loading data...")
    ex3 = load_stage0_data(n=10, seed=44, min_premises=1)
    exp3_orig_msgs, exp3_aug_msgs = [], []
    exp3_gts, exp3_rinfo = [], []
    for e in ex3:
        premises = [p["text"] for p in e["premises"]]
        gt = extract_ground_truth_conclusion(e)
        exp3_orig_msgs.append(format_user_message(premises))
        exp3_gts.append(gt)
        avail = [r for r in ALL_REDUNDANT if r not in premises]
        red = rng3.choice(avail or ALL_REDUNDANT)
        ins_pos = rng3.randint(0, len(premises))
        extended = premises[:ins_pos] + [red] + premises[ins_pos:]
        exp3_aug_msgs.append(format_user_message(extended))
        exp3_rinfo.append({"redundant": red, "insert_pos": ins_pos})

    return (
        ex1, exp1_orig_msgs, exp1_pert_msgs, exp1_gts, exp1_pinfo,
        ex2, exp2_orig_msgs, exp2_perm_msgs, exp2_gts, exp2_perm_idx,
        ex3, exp3_orig_msgs, exp3_aug_msgs, exp3_gts, exp3_rinfo,
    )


# ---------------------------------------------------------------------------
# Analysis functions (same logic as individual scripts)
# ---------------------------------------------------------------------------

def analyse_exp1(ex1, exp1_orig_msgs, exp1_pert_msgs, exp1_gts, exp1_pinfo,
                 orig_responses, pert_responses):
    results = []
    orig_correct = pert_correct = sensitive = 0
    for i, (e, gt, pinfo, orig_r, pert_r) in enumerate(
        zip(ex1, exp1_gts, exp1_pinfo, orig_responses, pert_responses)
    ):
        oc = parse_model_conclusion(orig_r)
        pc = parse_model_conclusion(pert_r)
        om = conclusions_match(oc, gt)
        pm = conclusions_match(pc, gt)
        ch = not conclusions_match(oc, pc, threshold=0.90)
        orig_correct += om; pert_correct += pm; sensitive += ch
        results.append({
            "id": e.get("id", str(i)),
            "premises": [p["text"] for p in e["premises"]],
            "ground_truth": gt,
            "perturbed_premise_idx": pinfo["idx"],
            "original_premise_text": pinfo["orig"],
            "perturbed_premise_text": pinfo["new"],
            "original_model_output": orig_r,
            "perturbed_model_output": pert_r,
            "original_conclusion": oc,
            "perturbed_conclusion": pc,
            "original_correct": bool(om),
            "perturbed_correct": bool(pm),
            "conclusion_changed": bool(ch),
        })
    n = len(results)
    summary = {
        "n_examples": n,
        "original_accuracy": orig_correct / n,
        "perturbed_accuracy": pert_correct / n,
        "accuracy_drop": (orig_correct - pert_correct) / n,
        "sensitivity": sensitive / n,
        "correct_and_sensitive": sum(1 for r in results if r["original_correct"] and r["conclusion_changed"]) / n,
    }
    return results, summary


def analyse_exp2(ex2, exp2_gts, exp2_perm_idx, orig_responses, perm_responses):
    results = []
    orig_correct = perm_correct = stable = consistent = 0
    for i, (e, gt, pidx, orig_r, perm_r) in enumerate(
        zip(ex2, exp2_gts, exp2_perm_idx, orig_responses, perm_responses)
    ):
        oc = parse_model_conclusion(orig_r)
        pc = parse_model_conclusion(perm_r)
        om = conclusions_match(oc, gt)
        pm = conclusions_match(pc, gt)
        st = conclusions_match(oc, pc, threshold=0.90)
        co = (om == pm)
        orig_correct += om; perm_correct += pm; stable += st; consistent += co
        results.append({
            "id": e.get("id", str(i)),
            "n_premises": len(e["premises"]),
            "premises": [p["text"] for p in e["premises"]],
            "ground_truth": gt,
            "permutation": pidx,
            "original_model_output": orig_r,
            "permuted_model_output": perm_r,
            "original_conclusion": oc,
            "permuted_conclusion": pc,
            "original_correct": bool(om),
            "permuted_correct": bool(pm),
            "stable": bool(st),
            "consistent": bool(co),
        })
    n = len(results)
    summary = {
        "n_examples": n,
        "original_accuracy": orig_correct / n,
        "permuted_accuracy": perm_correct / n,
        "accuracy_delta": (orig_correct - perm_correct) / n,
        "stability": stable / n,
        "consistency": consistent / n,
        "both_correct": sum(1 for r in results if r["original_correct"] and r["permuted_correct"]) / n,
        "neither_correct": sum(1 for r in results if not r["original_correct"] and not r["permuted_correct"]) / n,
        "only_original_correct": sum(1 for r in results if r["original_correct"] and not r["permuted_correct"]) / n,
        "only_permuted_correct": sum(1 for r in results if not r["original_correct"] and r["permuted_correct"]) / n,
    }
    return results, summary


def analyse_exp3(ex3, exp3_gts, exp3_rinfo, orig_responses, aug_responses):
    results = []
    monotone_preserved = orig_correct_count = aug_correct_count = both_correct = 0
    for i, (e, gt, ri, orig_r, aug_r) in enumerate(
        zip(ex3, exp3_gts, exp3_rinfo, orig_responses, aug_responses)
    ):
        oc = parse_model_conclusion(orig_r)
        ac = parse_model_conclusion(aug_r)
        om = conclusions_match(oc, gt)
        am = conclusions_match(ac, gt)
        mo = conclusions_match(oc, ac, threshold=0.85)
        monotone_preserved += mo; orig_correct_count += om; aug_correct_count += am
        if om and am: both_correct += 1
        results.append({
            "id": e.get("id", str(i)),
            "n_original_premises": len(e["premises"]),
            "premises": [p["text"] for p in e["premises"]],
            "redundant_premise": ri["redundant"],
            "redundant_inserted_at_position": ri["insert_pos"],
            "ground_truth": gt,
            "original_model_output": orig_r,
            "augmented_model_output": aug_r,
            "original_conclusion": oc,
            "augmented_conclusion": ac,
            "original_correct": bool(om),
            "augmented_correct": bool(am),
            "monotonicity_preserved": bool(mo),
        })
    n = len(results)
    summary = {
        "n_examples": n,
        "original_accuracy": orig_correct_count / n,
        "augmented_accuracy": aug_correct_count / n,
        "monotonicity_rate": monotone_preserved / n,
        "both_correct": both_correct / n,
    }
    return results, summary


# ---------------------------------------------------------------------------
# Save and plot functions
# ---------------------------------------------------------------------------

def save_exp1(results, summary):
    with open(EXP1_OUT / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(EXP1_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n = summary["n_examples"]
    orig_acc = summary["original_accuracy"]
    pert_acc = summary["perturbed_accuracy"]
    sensitivity = summary["sensitivity"]
    orig_correct = int(round(orig_acc * n))
    pert_correct = int(round(pert_acc * n))
    sensitive = int(round(sensitivity * n))

    with open(EXP1_OUT / "summary.txt", "w") as f:
        f.write("=== Experiment 1: Premise Perturbation ===\n\n")
        f.write(f"N examples: {n}\n")
        f.write(f"Original accuracy:                        {orig_acc:.3f} ({orig_correct}/{n})\n")
        f.write(f"Post-perturbation accuracy (matches GT):  {pert_acc:.3f} ({pert_correct}/{n})\n")
        f.write(f"Accuracy drop (orig - perturbed):         {summary['accuracy_drop']:.3f}\n")
        f.write(f"Sensitivity (conclusion changed):          {sensitivity:.3f} ({sensitive}/{n})\n\n")
        f.write("Interpretation:\n")
        f.write("  HIGH sensitivity (~1.0): model responds to premise changes → generalises\n")
        f.write("  LOW sensitivity (~0.0): model may be memorising conclusions\n\n")
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

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bars = ax.bar(["Original\nAccuracy", "Post-Perturbation\nAccuracy"],
                  [orig_acc, pert_acc], color=["#2196F3", "#F44336"], width=0.4)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Conclusion Accuracy Before vs After\nPremise Perturbation")
    for bar, val in zip(bars, [orig_acc, pert_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, val+0.02, f"{val:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance"); ax.legend()

    categories = {
        "Correct → Changed": sum(1 for r in results if r["original_correct"] and r["conclusion_changed"]),
        "Correct → Unchanged": sum(1 for r in results if r["original_correct"] and not r["conclusion_changed"]),
        "Wrong → Changed": sum(1 for r in results if not r["original_correct"] and r["conclusion_changed"]),
        "Wrong → Unchanged": sum(1 for r in results if not r["original_correct"] and not r["conclusion_changed"]),
    }
    ax2 = axes[1]
    ax2.pie(categories.values(), labels=categories.keys(),
            colors=["#4CAF50", "#8BC34A", "#FF9800", "#F44336"],
            autopct=lambda p: f"{p:.1f}%\n({int(round(p*n/100))})", startangle=90)
    ax2.set_title("Breakdown of Example Outcomes")
    plt.suptitle("Premise Perturbation Experiment (n=100, Stage-0 Model)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(EXP1_OUT / "accuracy_comparison.png"), dpi=150)
    plt.close()

    # Sensitivity by premise count
    premise_counts = [len(r["premises"]) for r in results]
    changed_by_count = {}
    for r in results:
        c = len(r["premises"])
        changed_by_count.setdefault(c, []).append(int(r["conclusion_changed"]))
    counts = sorted(changed_by_count.keys())
    fig2, ax3 = plt.subplots(figsize=(8, 5))
    bars3 = ax3.bar(counts, [np.mean(changed_by_count[c]) for c in counts], color="#9C27B0", alpha=0.8)
    ax3.set_xlabel("Number of premises"); ax3.set_ylabel("Sensitivity")
    ax3.set_title("Sensitivity to Perturbation by Number of Premises")
    for bar, c in zip(bars3, counts):
        v = np.mean(changed_by_count[c])
        ax3.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.2f}\n(n={len(changed_by_count[c])})", ha="center", fontsize=8)
    ax3.set_ylim(0, 1.1); ax3.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(EXP1_OUT / "sensitivity_by_premise_count.png"), dpi=150)
    plt.close()
    print(f"[EXP1] Results saved to {EXP1_OUT}/")


def save_exp2(results, summary):
    with open(EXP2_OUT / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(EXP2_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n = summary["n_examples"]
    orig_acc = summary["original_accuracy"]
    perm_acc = summary["permuted_accuracy"]
    stability = summary["stability"]

    with open(EXP2_OUT / "summary.txt", "w") as f:
        f.write("=== Experiment 2: Positional Permutation ===\n\n")
        f.write(f"N examples: {n}\n")
        f.write(f"Original accuracy (original premise order): {orig_acc:.3f} ({int(orig_acc*n)}/{n})\n")
        f.write(f"Permuted accuracy (shuffled premise order):  {perm_acc:.3f} ({int(perm_acc*n)}/{n})\n")
        f.write(f"Accuracy delta (orig - perm):                {summary['accuracy_delta']:+.3f}\n")
        f.write(f"Stability (same conclusion output):           {stability:.3f} ({int(stability*n)}/{n})\n")
        f.write(f"Consistency (same correctness pre/post):      {summary['consistency']:.3f}\n\n")
        f.write("Outcome breakdown:\n")
        for k in ["both_correct", "neither_correct", "only_original_correct", "only_permuted_correct"]:
            f.write(f"  {k:30s}: {summary[k]:.3f}\n")
        f.write("\nInterpretation:\n")
        f.write("  High stability (~1.0): model output is order-invariant (robust)\n")
        f.write("  Low stability: model is sensitive to premise order (positional bias)\n\n")
        f.write("=== Sample Cases ===\n\n")
        for r in results[:10]:
            f.write(f"ID: {r['id']} | #premises={r['n_premises']}\n")
            f.write(f"  Ground truth:         {r['ground_truth']}\n")
            f.write(f"  Original conclusion:  {r['original_conclusion']} (correct={r['original_correct']})\n")
            f.write(f"  Permuted conclusion:  {r['permuted_conclusion']} (correct={r['permuted_correct']})\n")
            f.write(f"  Stable: {r['stable']}\n\n")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax = axes[0]
    bars = ax.bar(["Original\nOrder", "Shuffled\nOrder"], [orig_acc, perm_acc],
                  color=["#2196F3", "#9C27B0"], width=0.4)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy Before vs After\nPremise Permutation")
    for bar, val in zip(bars, [orig_acc, perm_acc]):
        ax.text(bar.get_x() + bar.get_width()/2, val+0.02, f"{val:.3f}", ha="center", fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance"); ax.legend()

    ax2 = axes[1]
    metrics = {"Stability\n(same output)": stability, "Consistency\n(same correctness)": summary["consistency"]}
    bars2 = ax2.bar(list(metrics.keys()), list(metrics.values()), color=["#4CAF50", "#FF9800"], width=0.4)
    ax2.set_ylim(0, 1.1); ax2.set_ylabel("Rate")
    ax2.set_title("Stability and Consistency")
    for bar, val in zip(bars2, metrics.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, val+0.02, f"{val:.3f}", ha="center", fontweight="bold")

    ax3 = axes[2]
    cats = ["Both\ncorrect", "Neither\ncorrect", "Only orig\ncorrect", "Only perm\ncorrect"]
    vals = [summary["both_correct"], summary["neither_correct"],
            summary["only_original_correct"], summary["only_permuted_correct"]]
    bars3 = ax3.bar(cats, vals, color=["#4CAF50", "#F44336", "#2196F3", "#9C27B0"], width=0.5)
    ax3.set_ylim(0, max(vals) * 1.3 + 0.05); ax3.set_ylabel("Fraction")
    ax3.set_title("Correctness Outcome Breakdown")
    for bar, val in zip(bars3, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, val+0.005, f"{val:.3f}", ha="center", fontsize=10)

    plt.suptitle("Positional Permutation Experiment (n=100, Stage-0 Model)", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(EXP2_OUT / "accuracy_comparison.png"), dpi=150)
    plt.close()

    stability_by_n = {}
    for r in results:
        np_ = r["n_premises"]
        stability_by_n.setdefault(np_, []).append(int(r["stable"]))
    counts = sorted(stability_by_n.keys())
    fig2, ax4 = plt.subplots(figsize=(8, 5))
    bars4 = ax4.bar(counts, [np.mean(stability_by_n[c]) for c in counts], color="#673AB7", alpha=0.8)
    ax4.set_xlabel("Number of premises"); ax4.set_ylabel("Stability")
    ax4.set_title("Permutation Stability by Number of Premises")
    for bar, c in zip(bars4, counts):
        v = np.mean(stability_by_n[c])
        ax4.text(bar.get_x() + bar.get_width()/2, v+0.01, f"{v:.2f}\n(n={len(stability_by_n[c])})", ha="center", fontsize=8)
    ax4.set_ylim(0, 1.15); ax4.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(EXP2_OUT / "stability_by_premise_count.png"), dpi=150)
    plt.close()
    print(f"[EXP2] Results saved to {EXP2_OUT}/")


def save_exp3(results, summary):
    with open(EXP3_OUT / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(EXP3_OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    n = summary["n_examples"]
    with open(EXP3_OUT / "comparison_table.txt", "w") as f:
        f.write("=" * 80 + "\n")
        f.write("EXPERIMENT 3: MONOTONICITY TEST (ADD REDUNDANT PREMISE)\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Summary: {n} examples, Stage-0 model\n")
        f.write(f"  Original accuracy:            {summary['original_accuracy']:.2f} ({int(summary['original_accuracy']*n)}/{n})\n")
        f.write(f"  Augmented accuracy:           {summary['augmented_accuracy']:.2f} ({int(summary['augmented_accuracy']*n)}/{n})\n")
        f.write(f"  Monotonicity preserved (A≈B): {summary['monotonicity_rate']:.2f} ({int(summary['monotonicity_rate']*n)}/{n})\n\n")
        f.write("Interpretation:\n")
        f.write("  High monotonicity: adding irrelevant premises preserves conclusions (robust)\n")
        f.write("  Low monotonicity:  model is distracted by irrelevant content\n\n")
        f.write("=" * 80 + "\n\n")
        for i, r in enumerate(results):
            f.write(f"EXAMPLE {i+1}/{n}  (id={r['id']})\n")
            f.write("-" * 60 + "\n")
            f.write(f"Original premises ({r['n_original_premises']}):\n")
            for j, p in enumerate(r["premises"]):
                f.write(f"  [{j+1}] {p}\n")
            f.write(f"\nRedundant premise (position {r['redundant_inserted_at_position']}):\n")
            f.write(f"  >>> {r['redundant_premise']}\n")
            f.write(f"\nGround truth: {r['ground_truth']}\n")
            f.write(f"\nModel — ORIGINAL: {r['original_conclusion']} (correct={r['original_correct']})\n")
            f.write(f"Model — WITH REDUNDANT: {r['augmented_conclusion']} (correct={r['augmented_correct']})\n")
            f.write(f"Monotonicity preserved: {r['monotonicity_preserved']}\n")
            f.write("\n" + "=" * 80 + "\n\n")

    # Summary text
    with open(EXP3_OUT / "summary.txt", "w") as f:
        f.write("=== Experiment 3: Monotonicity Test ===\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    bars = ax.bar(["Original\nPremises", "With Redundant\nPremise"],
                  [summary["original_accuracy"], summary["augmented_accuracy"]],
                  color=["#2196F3", "#FF9800"], width=0.4)
    ax.set_ylim(0, 1.2); ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy: Original vs Augmented\n(n={n}, Stage-0 Model)")
    for bar, val in zip(bars, [summary["original_accuracy"], summary["augmented_accuracy"]]):
        ax.text(bar.get_x() + bar.get_width()/2, val+0.03, f"{val:.2f}", ha="center", fontsize=12, fontweight="bold")

    ax2 = axes[1]
    color_map = {(True, True): "#4CAF50", (True, False): "#8BC34A",
                 (False, True): "#FF9800", (False, False): "#F44336"}
    label_map = {(True, True): "Monotone+Correct", (True, False): "Monotone+Wrong",
                 (False, True): "Non-monotone+Correct", (False, False): "Non-monotone+Wrong"}
    y_positions = list(range(n))
    colors_by_status = [color_map[(r["monotonicity_preserved"], r["original_correct"])] for r in results]
    labels_for_legend = {label_map[k]: v for k, v in color_map.items()}
    ax2.barh(y_positions, [1] * n, color=colors_by_status, height=0.8)
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels([f"Ex {j+1}" for j in range(n)], fontsize=9)
    ax2.set_xticks([]); ax2.set_title("Per-example Monotonicity Status")
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in labels_for_legend.values()]
    ax2.legend(handles, list(labels_for_legend.keys()), loc="lower right", fontsize=8)
    plt.suptitle("Monotonicity Test: Adding Redundant Premise", fontsize=13)
    plt.tight_layout()
    plt.savefig(str(EXP3_OUT / "monotonicity_results.png"), dpi=150)
    plt.close()
    print(f"[EXP3] Results saved to {EXP3_OUT}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    (
        ex1, exp1_orig_msgs, exp1_pert_msgs, exp1_gts, exp1_pinfo,
        ex2, exp2_orig_msgs, exp2_perm_msgs, exp2_gts, exp2_perm_idx,
        ex3, exp3_orig_msgs, exp3_aug_msgs, exp3_gts, exp3_rinfo,
    ) = prepare_all_inputs()

    print(f"\nTotal inference calls: {len(exp1_orig_msgs) + len(exp1_pert_msgs) + len(exp2_orig_msgs) + len(exp2_perm_msgs) + len(exp3_orig_msgs) + len(exp3_aug_msgs)}")

    print(f"\nLoading model with stage0 adapter (single load for all 3 experiments)...")
    predictor = make_predictor(lora_adapter=STAGE0_ADAPTER)

    print("\n[EXP1] Running original inference...")
    exp1_orig_resp = batch_infer(predictor, exp1_orig_msgs)
    print("[EXP1] Running perturbed inference...")
    exp1_pert_resp = batch_infer(predictor, exp1_pert_msgs)

    print("\n[EXP2] Running original-order inference...")
    exp2_orig_resp = batch_infer(predictor, exp2_orig_msgs)
    print("[EXP2] Running permuted-order inference...")
    exp2_perm_resp = batch_infer(predictor, exp2_perm_msgs)

    print("\n[EXP3] Running original inference...")
    exp3_orig_resp = batch_infer(predictor, exp3_orig_msgs)
    print("[EXP3] Running augmented inference...")
    exp3_aug_resp = batch_infer(predictor, exp3_aug_msgs)

    # Analyse and save
    print("\nAnalysing results...")
    r1, s1 = analyse_exp1(ex1, exp1_orig_msgs, exp1_pert_msgs, exp1_gts, exp1_pinfo,
                           exp1_orig_resp, exp1_pert_resp)
    save_exp1(r1, s1)
    print(f"[EXP1] Original acc={s1['original_accuracy']:.3f}, Perturbed acc={s1['perturbed_accuracy']:.3f}, Sensitivity={s1['sensitivity']:.3f}")

    r2, s2 = analyse_exp2(ex2, exp2_gts, exp2_perm_idx, exp2_orig_resp, exp2_perm_resp)
    save_exp2(r2, s2)
    print(f"[EXP2] Original acc={s2['original_accuracy']:.3f}, Permuted acc={s2['permuted_accuracy']:.3f}, Stability={s2['stability']:.3f}")

    r3, s3 = analyse_exp3(ex3, exp3_gts, exp3_rinfo, exp3_orig_resp, exp3_aug_resp)
    save_exp3(r3, s3)
    print(f"[EXP3] Original acc={s3['original_accuracy']:.2f}, Augmented acc={s3['augmented_accuracy']:.2f}, Monotonicity={s3['monotonicity_rate']:.2f}")

    print("\n=== All inference experiments complete ===")


if __name__ == "__main__":
    main()
