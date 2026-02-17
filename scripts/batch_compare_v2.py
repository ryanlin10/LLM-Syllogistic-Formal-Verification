#!/usr/bin/env python3
"""Batch comparison v2: diverse logic challenges, puzzles, and edge cases."""

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
    # --- Logic Puzzles ---
    {
        "name": "1. Knights and Knaves (truth-teller identification)",
        "category": "logic_puzzle",
        "input": (
            "<PREMISE> {if Alice is a knight, then Alice tells the truth} </PREMISE> "
            "<PREMISE> {if Alice is a knave, then Alice lies} </PREMISE> "
            "<PREMISE> Alice says {Bob is a knave} </PREMISE> "
            "<PREMISE> {Alice is a knight or Alice is a knave} </PREMISE> "
            "<PREMISE> Bob says {Alice is a knave} </PREMISE>"
        ),
        "expected": "Alice and Bob are of different types (one knight, one knave)",
    },
    {
        "name": "2. Transitive ordering (who is tallest?)",
        "category": "logic_puzzle",
        "input": (
            "<PREMISE> {for all x, {for all y, {if x is taller than y, then {it is not the case that y is taller than x}}}} </PREMISE> "
            "<PREMISE> Alice is taller than Bob </PREMISE> "
            "<PREMISE> Bob is taller than Carol </PREMISE> "
            "<PREMISE> {for all x, {for all y, {for all z, {if {x is taller than y and y is taller than z}, then x is taller than z}}}} </PREMISE>"
        ),
        "expected": "Alice is taller than Carol",
    },
    {
        "name": "3. Elimination puzzle (process of elimination)",
        "category": "logic_puzzle",
        "input": (
            "<PREMISE> {the suspect is Alice or the suspect is Bob or the suspect is Carol} </PREMISE> "
            "<PREMISE> {it is not the case that the suspect is Alice} </PREMISE> "
            "<PREMISE> {if the suspect is Bob, then the weapon is found at the scene} </PREMISE> "
            "<PREMISE> {it is not the case that the weapon is found at the scene} </PREMISE>"
        ),
        "expected": "the suspect is Carol",
    },
    # --- Multi-step Reasoning ---
    {
        "name": "4. Proof by contradiction (reductio ad absurdum)",
        "category": "reasoning",
        "input": (
            "<PREMISE> {if {the number is rational}, then {the number can be expressed as p over q}} </PREMISE> "
            "<PREMISE> {if {the number can be expressed as p over q}, then {p squared equals 2 times q squared}} </PREMISE> "
            "<PREMISE> {if {p squared equals 2 times q squared}, then {p is even}} </PREMISE> "
            "<PREMISE> {if {p is even}, then {q is even}} </PREMISE> "
            "<PREMISE> {if {p is even and q is even}, then {p over q is not in lowest terms}} </PREMISE> "
            "<PREMISE> {if {p over q is not in lowest terms}, then {it is not the case that the number can be expressed as p over q}} </PREMISE>"
        ),
        "expected": "if the number is rational, then contradiction (it is not the case that the number is rational)",
    },
    {
        "name": "5. Chained conditionals (domino reasoning)",
        "category": "reasoning",
        "input": (
            "<PREMISE> {if the temperature drops below freezing, then the pipes freeze} </PREMISE> "
            "<PREMISE> {if the pipes freeze, then the water supply is cut} </PREMISE> "
            "<PREMISE> {if the water supply is cut, then the hospital activates emergency reserves} </PREMISE> "
            "<PREMISE> {if the hospital activates emergency reserves, then non-critical surgeries are postponed} </PREMISE> "
            "<PREMISE> the temperature drops below freezing </PREMISE>"
        ),
        "expected": "non-critical surgeries are postponed",
    },
    {
        "name": "6. Contrapositive chain",
        "category": "reasoning",
        "input": (
            "<PREMISE> {if the algorithm terminates, then the output is correct} </PREMISE> "
            "<PREMISE> {if the output is correct, then the test suite passes} </PREMISE> "
            "<PREMISE> {it is not the case that the test suite passes} </PREMISE>"
        ),
        "expected": "it is not the case that the algorithm terminates",
    },
    # --- Malformed / Incomplete Inputs ---
    {
        "name": "7. Unclosed PREMISE tag",
        "category": "malformed",
        "input": (
            "<PREMISE> {if it rains, then the ground is wet} "
            "<PREMISE> it rains </PREMISE>"
        ),
        "expected": "(test robustness to unclosed tag — ideally still derives: the ground is wet)",
    },
    {
        "name": "8. Missing PREMISE tags entirely",
        "category": "malformed",
        "input": (
            "All birds can fly. Penguins are birds. Therefore, penguins can fly."
        ),
        "expected": "(test behavior with plain English instead of semi-formal language)",
    },
    {
        "name": "9. Partial semi-formal with dangling connective",
        "category": "malformed",
        "input": (
            "<PREMISE> {if the door is locked, then </PREMISE> "
            "<PREMISE> the door is locked </PREMISE>"
        ),
        "expected": "(test behavior with truncated/malformed formula)",
    },
    {
        "name": "10. Empty premise mixed with valid ones",
        "category": "malformed",
        "input": (
            "<PREMISE> </PREMISE> "
            "<PREMISE> {if the battery is charged, then the device turns on} </PREMISE> "
            "<PREMISE> the battery is charged </PREMISE>"
        ),
        "expected": "the device turns on (should ignore empty premise)",
    },
    # --- Harder Reasoning ---
    {
        "name": "11. De Morgan's Law application",
        "category": "hard_reasoning",
        "input": (
            "<PREMISE> {it is not the case that {the file is corrupted and the backup failed}} </PREMISE> "
            "<PREMISE> the backup failed </PREMISE>"
        ),
        "expected": "it is not the case that the file is corrupted",
    },
    {
        "name": "12. Dilemma (constructive dilemma)",
        "category": "hard_reasoning",
        "input": (
            "<PREMISE> {if the economy grows, then unemployment decreases} </PREMISE> "
            "<PREMISE> {if the economy shrinks, then government spending increases} </PREMISE> "
            "<PREMISE> {the economy grows or the economy shrinks} </PREMISE>"
        ),
        "expected": "unemployment decreases or government spending increases",
    },
    {
        "name": "13. FOL with multiple quantifiers",
        "category": "hard_reasoning",
        "input": (
            "<PREMISE> {for all x, {if x is a student, then {there exist y such that {y is a course and x is enrolled in y}}}} </PREMISE> "
            "<PREMISE> Alice is a student </PREMISE>"
        ),
        "expected": "there exist y such that y is a course and Alice is enrolled in y",
    },
    {
        "name": "14. Negation of disjunction (neither-nor)",
        "category": "hard_reasoning",
        "input": (
            "<PREMISE> {it is not the case that {the patient has condition A or the patient has condition B}} </PREMISE>"
        ),
        "expected": "it is not the case that the patient has condition A, and it is not the case that the patient has condition B",
    },
    {
        "name": "15. Vacuous truth / trivially satisfied",
        "category": "edge_case",
        "input": (
            "<PREMISE> {for all x, {if x is a unicorn, then x can fly}} </PREMISE> "
            "<PREMISE> {it is not the case that {there exist x such that x is a unicorn}} </PREMISE>"
        ),
        "expected": "(vacuously true — no unicorns exist, so the universal is trivially satisfied; no new conclusion)",
    },
]


def run_batch(predictor, prompts, temperature=0.3, max_tokens=512):
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
    lines = []
    lines.append("=" * 90)
    lines.append("FINETUNED vs BASE MODEL COMPARISON — V2: Diverse Logic Challenges")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Base model: mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    lines.append(f"LoRA adapter: mistralai_Mistral_Small_3.2_24B_Instruct_2506_20260120_021836/final_fixed")
    lines.append(f"System prompt: {SYSTEM_PROMPT}")
    lines.append(f"Temperature: 0.3 | Prompts: {len(prompts)}")
    lines.append("=" * 90)

    current_category = None
    for i, prompt in enumerate(prompts):
        if prompt["category"] != current_category:
            current_category = prompt["category"]
            label = {
                "logic_puzzle": "LOGIC PUZZLES",
                "reasoning": "MULTI-STEP REASONING",
                "malformed": "MALFORMED / INCOMPLETE INPUTS",
                "hard_reasoning": "HARDER REASONING",
                "edge_case": "EDGE CASES",
            }.get(current_category, current_category.upper())
            lines.append("")
            lines.append(f"{'#' * 90}")
            lines.append(f"## {label}")
            lines.append(f"{'#' * 90}")

        lines.append("")
        lines.append("-" * 90)
        lines.append(f"PROMPT {prompt['name']}")
        lines.append(f"Category: {prompt['category']}")
        lines.append("-" * 90)
        lines.append("")
        lines.append(f"INPUT:")
        # Wrap long inputs
        inp = prompt["input"]
        while len(inp) > 100:
            split_at = inp.rfind(" ", 0, 100)
            if split_at == -1:
                split_at = 100
            lines.append(f"  {inp[:split_at]}")
            inp = inp[split_at:].lstrip()
        if inp:
            lines.append(f"  {inp}")
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
    lines.append("=" * 90)
    lines.append("END OF COMPARISON")
    lines.append("=" * 90)
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
    output_path = output_dir / "model_comparison_v2.txt"

    text = format_results(PROMPTS, finetuned_results, base_results)
    output_path.write_text(text)

    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")
    print(text)


if __name__ == "__main__":
    main()
