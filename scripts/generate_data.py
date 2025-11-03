#!/usr/bin/env python3
"""Script to generate synthetic training data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.generator import SyntheticDataGenerator, GenerationConfig
from src.data.curation import DataCurator
import yaml


def main():
    # Load config
    config_path = Path(__file__).parent.parent / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Example documents (in production, load from files)
    sample_documents = [
        "The company's revenue increased by 15% in Q3. This growth was driven by strong performance in the technology division. The division saw a 25% increase in sales compared to the previous quarter.",
        "All employees must complete security training annually. The training covers data protection, password management, and incident reporting. Failure to complete training may result in access restrictions.",
        "The new product launch is scheduled for next month. Market research indicates strong demand in the target demographic. Pre-orders have exceeded initial projections by 30%.",
    ]
    
    questions = [
        "What drove the revenue increase?",
        "What are the consequences of not completing security training?",
        "What evidence supports the product launch timing?",
    ]
    
    # Generation config
    gen_config = GenerationConfig(
        model_name=config["model"]["base_model"],
        num_examples=config.get("data_generation", {}).get("num_examples", 100),
        include_adversarial=True,
        adversarial_ratio=0.2
    )
    
    # Generate
    print("Generating synthetic data...")
    generator = SyntheticDataGenerator(gen_config)
    annotations = generator.generate_batch(
        sample_documents,
        questions,
        output_path=config["data"]["synthetic_data_path"]
    )
    
    print(f"Generated {len(annotations)} synthetic annotations")


if __name__ == "__main__":
    main()

