#!/usr/bin/env python3
"""Batch comparison of finetuned vs base model on semi-formal logic prompts."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import VLLMPredictor
from src.utils.config_loader import load_config

SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Given the following premises, "
    "derive their valid conclusion."
)

PROMPTS = [
    {
        "name": "1. Modus Ponens (nested conjunction)",
        "pattern": "modus_ponens",
        "input": (
            "<PREMISE> {if {the river is flooding and the dam is weakened}, "
            "then the town must evacuate} </PREMISE> "
            "<PREMISE> {the river is flooding and the dam is weakened} </PREMISE>"
        ),
        "expected": "the town must evacuate",
    },
    {
        "name": "2. Modus Tollens",
        "pattern": "modus_tollens",
        "input": (
            "<PREMISE> {if {the sensor detects motion}, then {the alarm is triggered}} </PREMISE> "
            "<PREMISE> {it is not the case that the alarm is triggered} </PREMISE>"
        ),
        "expected": "it is not the case that the sensor detects motion",
    },
    {
        "name": "3. Disjunctive Syllogism + Modus Ponens",
        "pattern": "disjunctive_syllogism",
        "input": (
            "<PREMISE> {if {the server crashes}, then "
            "{the backup activates or the data is lost}} </PREMISE> "
            "<PREMISE> the server crashes </PREMISE> "
            "<PREMISE> {it is not the case that the data is lost} </PREMISE>"
        ),
        "expected": "the backup activates",
    },
    {
        "name": "4. FOL Universal Syllogism (hypothetical syllogism)",
        "pattern": "universal_syllogism",
        "input": (
            "<PREMISE> {for all x, {if x is a reptile, then x is cold-blooded}} </PREMISE> "
            "<PREMISE> {for all x, {if x is a lizard, then x is a reptile}} </PREMISE>"
        ),
        "expected": "for all x, if x is a lizard, then x is cold-blooded",
    },
    {
        "name": "5. Double Modus Ponens (nested implication)",
        "pattern": "nested_modus_ponens",
        "input": (
            "<PREMISE> {if {the experiment succeeds}, then "
            "{if {the results are replicated}, then {the hypothesis is confirmed}}} </PREMISE> "
            "<PREMISE> the experiment succeeds </PREMISE> "
            "<PREMISE> the results are replicated </PREMISE>"
        ),
        "expected": "the hypothesis is confirmed",
    },
    {
        "name": "6. Conjunction Introduction",
        "pattern": "conjunction_intro",
        "input": (
            "<PREMISE> the satellite is in orbit </PREMISE> "
            "<PREMISE> the communication link is stable </PREMISE>"
        ),
        "expected": "the satellite is in orbit and the communication link is stable",
    },
    {
        "name": "7. Biconditional Elimination",
        "pattern": "biconditional_elim",
        "input": (
            "<PREMISE> {the circuit is complete if and only if the current flows} </PREMISE> "
            "<PREMISE> the circuit is complete </PREMISE>"
        ),
        "expected": "the current flows",
    },
    {
        "name": "8. Incomplete Premises (missing antecedent)",
        "pattern": "incomplete_missing",
        "input": (
            "<PREMISE> {if {the bridge is structurally sound}, "
            "then {vehicles may cross safely}} </PREMISE>"
        ),
        "expected": "(insufficient premises — no conclusion derivable)",
    },
    {
        "name": "9. Contradictory Premises",
        "pattern": "incomplete_contradictory",
        "input": (
            "<PREMISE> the reactor is stable </PREMISE> "
            "<PREMISE> {it is not the case that the reactor is stable} </PREMISE>"
        ),
        "expected": "(contradiction — any conclusion follows / ex falso quodlibet)",
    },
    {
        "name": "10. FOL Existential + Universal",
        "pattern": "existential_universal",
        "input": (
            "<PREMISE> {for all x, {if x is radioactive, then x is hazardous}} </PREMISE> "
            "<PREMISE> {there exist y such that y is radioactive} </PREMISE>"
        ),
        "expected": "there exist y such that y is hazardous",
    },
]


def run_batch(predictor, prompts, temperature=0.3, max_tokens=512):
    """Run all prompts through a predictor and return results."""
    results = []
    for i, prompt in enumerate(prompts):
        print(f"  Running prompt {i+1}/{len(prompts)}: {prompt['name']}")
        output = predictor.generate(
            message=prompt["input"],
            system_prompt=SYSTEM_PROMPT,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        results.append(output.strip())
    return results


def format_results(prompts, finetuned_results, base_results):
    """Format comparison results as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("FINETUNED vs BASE MODEL COMPARISON")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base model: mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    lines.append(f"LoRA adapter: mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260120_021836/final")
    lines.append(f"System prompt: {SYSTEM_PROMPT}")
    lines.append(f"Temperature: 0.3")
    lines.append("=" * 80)

    for i, prompt in enumerate(prompts):
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"PROMPT {prompt['name']}")
        lines.append(f"Pattern: {prompt['pattern']}")
        lines.append("-" * 80)
        lines.append("")
        lines.append(f"INPUT:")
        lines.append(f"  {prompt['input']}")
        lines.append("")
        lines.append(f"EXPECTED:")
        lines.append(f"  {prompt['expected']}")
        lines.append("")
        lines.append(f"FINETUNED MODEL OUTPUT:")
        for line in finetuned_results[i].split("\n"):
            lines.append(f"  {line}")
        lines.append("")
        lines.append(f"BASE MODEL OUTPUT:")
        for line in base_results[i].split("\n"):
            lines.append(f"  {line}")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF COMPARISON")
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    project_root = Path(__file__).parent.parent
    config = load_config(str(project_root / "config.yaml"))

    model_path = config["model"]["base_model"]
    lora_path = config["model"].get("lora_adapter")
    if lora_path and not Path(lora_path).is_absolute():
        lora_path = str(project_root / lora_path)

    # --- Finetuned model ---
    print(f"\n{'='*60}")
    print("Loading FINETUNED model (base + LoRA)...")
    print(f"{'='*60}")
    finetuned_predictor = VLLMPredictor(
        model_path=model_path,
        lora_adapter_path=lora_path,
        tensor_parallel_size=1,
    )
    finetuned_results = run_batch(finetuned_predictor, PROMPTS)

    # Free GPU memory
    del finetuned_predictor
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()

    # --- Base model ---
    print(f"\n{'='*60}")
    print("Loading BASE model (no LoRA)...")
    print(f"{'='*60}")
    base_predictor = VLLMPredictor(
        model_path=model_path,
        lora_adapter_path=None,
        tensor_parallel_size=1,
    )
    base_results = run_batch(base_predictor, PROMPTS)

    del base_predictor
    gc.collect()
    torch.cuda.empty_cache()

    # --- Write results ---
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model_comparison.txt"

    text = format_results(PROMPTS, finetuned_results, base_results)
    output_path.write_text(text)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print(text)


if __name__ == "__main__":
    main()
