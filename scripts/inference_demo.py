#!/usr/bin/env python3
"""Simple inference demo using vLLM."""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.predictor import VLLMPredictor
from src.utils.config_loader import load_config


DEFAULT_SYSTEM_PROMPT = (
    "You are a logical reasoning assistant. Given the following premises, "
    "derive their valid conclusion."
)


def main():
    parser = argparse.ArgumentParser(description="Run inference with vLLM")
    parser.add_argument("message", nargs="?", help="Input message")
    parser.add_argument(
        "--system-prompt", "-s", default=DEFAULT_SYSTEM_PROMPT, help="System prompt"
    )
    parser.add_argument("--model", "-m", help="Base model path (overrides config)")
    parser.add_argument("--lora", "-l", help="LoRA adapter path (overrides config)")
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA adapter")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p")
    parser.add_argument(
        "--tensor-parallel", "-tp", type=int, default=1, help="Tensor parallel size"
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    config_path = project_root / "config.yaml"
    config = load_config(str(config_path))

    model_path = args.model or config["model"]["base_model"]
    lora_adapter_path = None
    if not args.no_lora:
        lora_adapter_path = args.lora or config["model"].get("lora_adapter")
        # Resolve relative paths against project root
        if lora_adapter_path and not Path(lora_adapter_path).is_absolute():
            lora_adapter_path = str(project_root / lora_adapter_path)

    print(f"Loading model: {model_path}")
    if lora_adapter_path:
        print(f"With LoRA adapter: {lora_adapter_path}")

    predictor = VLLMPredictor(
        model_path=model_path,
        lora_adapter_path=lora_adapter_path,
        tensor_parallel_size=args.tensor_parallel,
    )

    message = args.message
    if not message:
        print("Enter message (Ctrl+D to submit):")
        message = sys.stdin.read().strip()

    response = predictor.generate(
        message=message,
        system_prompt=args.system_prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    print(response)


if __name__ == "__main__":
    main()
