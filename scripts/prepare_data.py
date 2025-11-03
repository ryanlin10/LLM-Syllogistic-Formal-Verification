#!/usr/bin/env python3
"""Script to prepare and split training data."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.curation import DataCurator
import yaml


def main():
    # Load config with environment variable support
    from src.utils.config_loader import load_config
    
    config_path = Path(__file__).parent.parent / "config.yaml"
    config = load_config(str(config_path))
    
    curator = DataCurator(seed=config["data"]["seed"])
    
    # Input data (could be synthetic, annotated, logic datasets, or combined)
    input_files = [
        config["data"]["synthetic_data_path"],
    ]
    
    # Add logic datasets if enabled
    if config["data"].get("use_logic_datasets", False):
        logic_path = config["data"].get("logic_datasets_path", "./data/logic_datasets_train.jsonl")
        if Path(logic_path).exists():
            input_files.append(logic_path)
            print(f"Will include logic datasets from: {logic_path}")
        else:
            print(f"Warning: Logic datasets path {logic_path} not found.")
            print(f"  Run: python scripts/load_logic_datasets.py to generate it")
    
    # Combine all data
    all_data = []
    for file_path in input_files:
        if Path(file_path).exists():
            data = curator.load_jsonl(file_path)
            all_data.extend(data)
            print(f"Loaded {len(data)} examples from {file_path}")
        else:
            print(f"Warning: {file_path} not found")
    
    if not all_data:
        print("No data found. Please generate or provide training data first.")
        return
    
    print(f"Total examples: {len(all_data)}")
    
    # Create splits
    data_dir = Path(config["paths"]["data_dir"])
    curator.create_train_split(
        input_path=None,  # We'll pass data directly
        output_dir=str(data_dir),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        validate=True,
        balance=False
    )
    
    # Save combined data temporarily for splitting
    import tempfile
    import json
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        for ann in all_data:
            f.write(json.dumps(ann) + "\n")
        temp_path = f.name
    
    # Now create splits
    curator.create_train_split(
        input_path=temp_path,
        output_dir=str(data_dir),
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        validate=True,
        balance=False
    )
    
    # Clean up
    Path(temp_path).unlink()
    
    print("Data preparation complete!")


if __name__ == "__main__":
    main()

