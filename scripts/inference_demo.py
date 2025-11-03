#!/usr/bin/env python3
"""Demo script for inference."""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import StructuredLLMPredictor
import yaml


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    # Load config with environment variable support
    from src.utils.config_loader import load_config
    
    config = load_config(str(config_path))
    
    # Model path (adjust based on your training)
    model_path = config["training"]["output_dir"] + "/final"
    
    if not Path(model_path).exists():
        print(f"Model not found at {model_path}")
        print("Please train the model first using scripts/train_finetune.py")
        return
    
    # Create predictor
    predictor = StructuredLLMPredictor(model_path, config_path=str(config_path))
    
    # Example queries
    questions = [
        "What are the main benefits of our product?",
        "What security measures are required?",
        "What evidence supports the revenue growth?",
    ]
    
    print("Running inference examples...\n")
    
    results = []
    for question in questions:
        print(f"Question: {question}")
        result = predictor.generate(question, verify=True)
        
        print(f"\nParsed Output:")
        print(json.dumps(result["parsed"], indent=2) if result["parsed"] else "Parse failed")
        
        if result["verification"]:
            print(f"\nVerification: {result['verification']['verdict']}")
            print(f"Confidence: {result['verification']['confidence']:.3f}")
        
        print("\n" + "-"*50 + "\n")
        results.append(result)
    
    # Save results
    output_path = Path(__file__).parent.parent / "outputs" / "inference_demo.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()

