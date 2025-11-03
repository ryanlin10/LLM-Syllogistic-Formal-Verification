#!/usr/bin/env python3
"""Script to load and convert logic datasets (LogiQA, LogicNLI, LogiBench) to DAG format."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.logic_datasets import LogicDatasetAggregator
from src.utils.config_loader import load_config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and convert logic datasets to DAG format")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["logiqa", "logicnli", "logibench", "all"],
        default=["all"],
        help="Which datasets to load"
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (default: data/logic_datasets_{split}.jsonl)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config.yaml",
        help="Config file path"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        data_dir = Path(config.get("paths", {}).get("data_dir", "./data"))
        output_path = data_dir / f"logic_datasets_{args.split}.jsonl"
    
    # Load datasets
    aggregator = LogicDatasetAggregator()
    
    datasets_to_load = args.datasets
    if "all" in datasets_to_load:
        datasets_to_load = ["logiqa", "logicnli", "logibench"]
    
    print(f"Loading datasets: {datasets_to_load}")
    print(f"Split: {args.split}")
    print(f"Output: {output_path}")
    
    aggregator.save_converted(str(output_path), split=args.split)
    
    print(f"\n✓ Completed! Converted datasets saved to {output_path}")
    print(f"  You can now use this file with prepare_data.py or training scripts")


if __name__ == "__main__":
    main()

