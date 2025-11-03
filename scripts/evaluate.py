#!/usr/bin/env python3
"""Script to evaluate model outputs."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.evaluator import evaluate_model


def main():
    # This would typically load model outputs from a file
    # For demo, we'll create placeholder outputs
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    # Load test data
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # In practice, you'd run inference to generate these
    # For now, this is a placeholder
    print("Note: This script expects model outputs. Run inference first or provide output file.")
    print("Usage: python evaluate.py --model_outputs outputs.jsonl --ground_truth data/test.jsonl")


if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs", type=str, required=True)
    parser.add_argument("--ground_truth", type=str, required=True)
    parser.add_argument("--config", type=str, default="./config.yaml")
    
    args = parser.parse_args()
    
    # Load config with environment variable support
    from src.utils.config_loader import load_config
    
    # Load model outputs
    model_outputs = []
    with open(args.model_outputs, "r") as f:
        for line in f:
            data = json.loads(line)
            model_outputs.append(data.get("output", ""))
    
    # Evaluate
    metrics = evaluate_model(model_outputs, args.ground_truth, config_path=args.config)
    metrics.print_summary()
    
    # Save results
    output_path = Path(args.model_outputs).parent / "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    print(f"\nSaved detailed results to {output_path}")

