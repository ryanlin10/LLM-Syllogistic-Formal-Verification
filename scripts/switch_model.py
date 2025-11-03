#!/usr/bin/env python3
"""Utility script to easily switch models via .env file."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config_loader import (
    set_model_in_env,
    list_available_models,
    get_model_name,
    load_config
)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Switch the model used by SyLLM pipeline via .env file"
    )
    parser.add_argument(
        "model_name",
        nargs="?",
        type=str,
        help="Model name to switch to (e.g., 'deepseek-ai/deepseek-v3')"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models"
    )
    parser.add_argument(
        "--current",
        action="store_true",
        help="Show current model"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="./.env",
        help="Path to .env file (default: ./.env)"
    )
    
    args = parser.parse_args()
    
    # List available models
    if args.list:
        print("\nAvailable models:")
        print("=" * 60)
        models = list_available_models()
        for model_name, description in models.items():
            print(f"  {model_name:50s} - {description}")
        print("=" * 60)
        print("\nUsage: python scripts/switch_model.py <model_name>")
        return
    
    # Show current model
    if args.current:
        config = load_config()
        current_model = get_model_name(config)
        print(f"\nCurrent model: {current_model}")
        if Path(args.env).exists():
            print(f"  (from {args.env})")
        else:
            print(f"  (from config.yaml)")
        return
    
    # Switch model
    if args.model_name:
        set_model_in_env(args.model_name, args.env)
        print(f"\n✓ Model switched to: {args.model_name}")
        print(f"\nTo use this model, run your training/inference scripts.")
        print(f"The model name is now set in {args.env}")
    else:
        print("Error: Please provide a model name or use --list/--current")
        print("\nUsage:")
        print("  python scripts/switch_model.py <model_name>  # Switch model")
        print("  python scripts/switch_model.py --list        # List available models")
        print("  python scripts/switch_model.py --current      # Show current model")
        return


if __name__ == "__main__":
    main()

