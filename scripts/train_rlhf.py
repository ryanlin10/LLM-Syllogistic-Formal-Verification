#!/usr/bin/env python3
"""Script to run RLHF training."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.rlhf import train_rlhf


def main():
    config_path = Path(__file__).parent.parent / "config.yaml"
    train_rlhf(str(config_path))


if __name__ == "__main__":
    main()


