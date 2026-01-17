#!/usr/bin/env python3
"""Demo script for inference."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import LLMPredictor
from src.utils.config_loader import load_config


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))

    model_path = config["model"]["base_model"]

    predictor = LLMPredictor(model_path)

    prompts = [
        "What are the main benefits of cloud computing?",
        "Explain the concept of machine learning in simple terms.",
        "What is the capital of France?",
    ]

    print("Running inference...\n")

    for prompt in prompts:
        print(f"Prompt: {prompt}")
        response = predictor.generate(prompt)
        print(f"Response: {response}")
        print("\n" + "-"*50 + "\n")


if __name__ == "__main__":
    main()

