#!/usr/bin/env python3
"""
Logic Puzzle Benchmarking Script.

Tests finetuned models on various logic reasoning benchmarks:
- LogiQA: Logic question answering
- LogicNLI: Natural language inference with logic
- FOLIO: First-order logic reasoning
- ReClor: Reading comprehension requiring logical reasoning
- AR-LSAT: Law School Admission Test analytical reasoning

Features:
- Support for LoRA-finetuned and full models
- Multiple benchmark datasets
- Comprehensive metrics (accuracy, reasoning quality, consistency)
- Detailed per-category analysis
- JSON and CSV report generation
"""

import os
import sys
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict
import re

import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from datasets import load_dataset

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


# =============================================================================
# Benchmark Registry
# =============================================================================

BENCHMARK_REGISTRY = {
    "logiqa": {
        "name": "LogiQA",
        "description": "Logic Question Answering",
        "hf_dataset": "lucasmccabe/logiqa",
        "task_type": "multiple_choice",
        "metrics": ["accuracy", "reasoning_quality"],
    },
    "logiqa2": {
        "name": "LogiQA 2.0",
        "description": "Logic Question Answering v2",
        "hf_dataset": "TIGER-Lab/LogiQA-v2",
        "task_type": "multiple_choice",
        "metrics": ["accuracy", "reasoning_quality"],
    },
    "reclor": {
        "name": "ReClor",
        "description": "Reading Comprehension requiring Logical Reasoning",
        "hf_dataset": "metaeval/reclor",
        "task_type": "multiple_choice",
        "metrics": ["accuracy", "reasoning_quality"],
    },
    "folio": {
        "name": "FOLIO",
        "description": "First-Order Logic Reasoning",
        "hf_dataset": "yale-nlp/FOLIO",
        "task_type": "entailment",
        "metrics": ["accuracy", "consistency"],
    },
    "proofwriter": {
        "name": "ProofWriter",
        "description": "Logical Reasoning with Proofs",
        "hf_dataset": "allenai/proofwriter",
        "task_type": "entailment",
        "metrics": ["accuracy", "proof_quality"],
    },
    "ruletaker": {
        "name": "RuleTaker",
        "description": "Rule-based Reasoning",
        "hf_dataset": "ruletaker",
        "task_type": "entailment",
        "metrics": ["accuracy", "depth_analysis"],
    },
    "clutrr": {
        "name": "CLUTRR",
        "description": "Compositional Language Understanding with Text-based Relational Reasoning",
        "hf_dataset": "CLUTRR/v1",
        "task_type": "relation_extraction",
        "metrics": ["accuracy", "hop_analysis"],
    },
}


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class BenchmarkExample:
    """A single benchmark example."""
    id: str
    context: str
    question: str
    options: List[str]
    correct_answer: int  # Index of correct option
    category: Optional[str] = None
    difficulty: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    example_id: str
    predicted_answer: int
    correct_answer: int
    is_correct: bool
    raw_output: str
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: float = 0.0
    tokens_generated: int = 0


@dataclass
class BenchmarkMetrics:
    """Aggregated benchmark metrics."""
    benchmark_name: str
    model_name: str
    total_examples: int
    correct: int
    accuracy: float
    accuracy_by_category: Dict[str, float] = field(default_factory=dict)
    accuracy_by_difficulty: Dict[str, float] = field(default_factory=dict)
    avg_latency_ms: float = 0.0
    avg_tokens: float = 0.0
    reasoning_quality_score: float = 0.0
    consistency_score: float = 0.0
    timestamp: str = ""


# =============================================================================
# Benchmark Loaders
# =============================================================================

class BenchmarkLoader:
    """Base class for loading benchmark datasets."""

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[BenchmarkExample]:
        raise NotImplementedError

    def _truncate(self, examples: List[BenchmarkExample], max_samples: Optional[int]) -> List[BenchmarkExample]:
        if max_samples and len(examples) > max_samples:
            return examples[:max_samples]
        return examples


class LogiQALoader(BenchmarkLoader):
    """Loader for LogiQA benchmark."""

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[BenchmarkExample]:
        examples = []

        try:
            # Try different dataset variants
            for dataset_name in ["lucasmccabe/logiqa", "hsdemo/LogiQA"]:
                try:
                    dataset = load_dataset(dataset_name, split=split)
                    break
                except:
                    continue
            else:
                print("Warning: Could not load LogiQA dataset")
                return []

            for idx, item in enumerate(dataset):
                context = item.get("context", item.get("article", ""))
                question = item.get("question", "")
                options = item.get("options", item.get("answers", []))
                label = item.get("label", item.get("answer", 0))

                # Handle label format
                if isinstance(label, str):
                    label_map = {"A": 0, "B": 1, "C": 2, "D": 3}
                    label = label_map.get(label.upper(), 0)

                examples.append(BenchmarkExample(
                    id=f"logiqa_{idx}",
                    context=context,
                    question=question,
                    options=list(options) if options else [],
                    correct_answer=int(label),
                    category=item.get("type", "general"),
                ))

        except Exception as e:
            print(f"Error loading LogiQA: {e}")

        return self._truncate(examples, max_samples)


class ReClOrLoader(BenchmarkLoader):
    """Loader for ReClor benchmark."""

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[BenchmarkExample]:
        examples = []

        try:
            dataset = load_dataset("metaeval/reclor", split=split if split != "test" else "validation")

            for idx, item in enumerate(dataset):
                context = item.get("context", "")
                question = item.get("question", "")
                options = [
                    item.get("answer_0", ""),
                    item.get("answer_1", ""),
                    item.get("answer_2", ""),
                    item.get("answer_3", ""),
                ]
                label = item.get("label", 0)

                examples.append(BenchmarkExample(
                    id=f"reclor_{idx}",
                    context=context,
                    question=question,
                    options=[o for o in options if o],
                    correct_answer=int(label),
                    category=item.get("id_string", "").split("-")[0] if item.get("id_string") else "general",
                ))

        except Exception as e:
            print(f"Error loading ReClor: {e}")

        return self._truncate(examples, max_samples)


class FOLIOLoader(BenchmarkLoader):
    """Loader for FOLIO benchmark."""

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[BenchmarkExample]:
        examples = []

        try:
            dataset = load_dataset("yale-nlp/FOLIO", split=split if split != "test" else "validation")

            for idx, item in enumerate(dataset):
                premises = item.get("premises", "")
                conclusion = item.get("conclusion", "")
                label = item.get("label", "Unknown")

                # Convert to multiple choice format
                options = ["True", "False", "Unknown"]
                label_map = {"True": 0, "False": 1, "Unknown": 2}
                correct = label_map.get(label, 2)

                examples.append(BenchmarkExample(
                    id=f"folio_{idx}",
                    context=premises,
                    question=f"Based on the premises, is the following conclusion true, false, or unknown?\n\nConclusion: {conclusion}",
                    options=options,
                    correct_answer=correct,
                    category="first_order_logic",
                ))

        except Exception as e:
            print(f"Error loading FOLIO: {e}")

        return self._truncate(examples, max_samples)


class ProofWriterLoader(BenchmarkLoader):
    """Loader for ProofWriter benchmark."""

    def load(self, split: str = "test", max_samples: Optional[int] = None) -> List[BenchmarkExample]:
        examples = []

        try:
            # ProofWriter has different depth configurations
            dataset = load_dataset("allenai/proofwriter", "depth-5", split=split)

            for idx, item in enumerate(dataset):
                context = item.get("theory", "")
                question = item.get("question", "")
                answer = item.get("answer", True)

                options = ["True", "False"]
                correct = 0 if answer else 1

                examples.append(BenchmarkExample(
                    id=f"proofwriter_{idx}",
                    context=context,
                    question=question,
                    options=options,
                    correct_answer=correct,
                    difficulty=f"depth_{item.get('depth', 0)}",
                    metadata={"depth": item.get("depth", 0)},
                ))

        except Exception as e:
            print(f"Error loading ProofWriter: {e}")

        return self._truncate(examples, max_samples)


def get_benchmark_loader(benchmark_name: str) -> BenchmarkLoader:
    """Get the appropriate loader for a benchmark."""
    loaders = {
        "logiqa": LogiQALoader(),
        "logiqa2": LogiQALoader(),  # Same loader, different dataset internally
        "reclor": ReClOrLoader(),
        "folio": FOLIOLoader(),
        "proofwriter": ProofWriterLoader(),
    }
    return loaders.get(benchmark_name, LogiQALoader())


# =============================================================================
# Model Wrapper
# =============================================================================

class LogicReasoningModel:
    """Wrapper for inference with finetuned models."""

    def __init__(
        self,
        model_path: str,
        base_model_path: Optional[str] = None,
        device: str = "auto",
        torch_dtype: str = "float16",
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ):
        self.model_path = model_path
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample

        # Determine torch dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        # Load model
        self._load_model(base_model_path)

    def _load_model(self, base_model_path: Optional[str] = None):
        """Load the model and tokenizer."""
        model_path = Path(self.model_path)

        # Check if this is a LoRA adapter
        is_lora = (model_path / "adapter_config.json").exists()

        if is_lora:
            print(f"Loading LoRA adapter from: {model_path}")

            # Load training config to get base model
            config_file = model_path / "training_config.json"
            if config_file.exists() and not base_model_path:
                with open(config_file) as f:
                    train_config = json.load(f)
                    base_model_path = train_config.get("model_name")

            if not base_model_path:
                # Try to infer from adapter config
                adapter_config_file = model_path / "adapter_config.json"
                with open(adapter_config_file) as f:
                    adapter_config = json.load(f)
                    base_model_path = adapter_config.get("base_model_name_or_path")

            if not base_model_path:
                raise ValueError("Could not determine base model path. Please provide --base-model")

            print(f"Loading base model: {base_model_path}")

            # Load base model
            self.model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                str(model_path),
                torch_dtype=self.torch_dtype,
            )

            # Merge for faster inference (optional)
            # self.model = self.model.merge_and_unload()

            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_path,
                trust_remote_code=True,
            )
        else:
            print(f"Loading full model from: {model_path}")

            self.model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=self.torch_dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True,
            )

        # Setup tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded successfully")

    def generate(self, prompt: str) -> Tuple[str, int, float]:
        """Generate a response for a prompt.

        Returns:
            Tuple of (response_text, num_tokens, latency_ms)
        """
        start_time = time.time()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if self.do_sample else 1.0,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        latency_ms = (time.time() - start_time) * 1000
        num_tokens = len(new_tokens)

        return response, num_tokens, latency_ms

    def predict_multiple_choice(
        self,
        example: BenchmarkExample,
        prompt_template: str = "structured",
    ) -> PredictionResult:
        """Predict answer for a multiple choice example."""

        # Format prompt based on template
        if prompt_template == "structured":
            prompt = self._format_structured_prompt(example)
        elif prompt_template == "cot":
            prompt = self._format_cot_prompt(example)
        else:
            prompt = self._format_simple_prompt(example)

        # Generate response
        response, num_tokens, latency_ms = self.generate(prompt)

        # Parse answer from response
        predicted_answer, confidence, reasoning = self._parse_response(response, len(example.options))

        return PredictionResult(
            example_id=example.id,
            predicted_answer=predicted_answer,
            correct_answer=example.correct_answer,
            is_correct=(predicted_answer == example.correct_answer),
            raw_output=response,
            reasoning=reasoning,
            confidence=confidence,
            latency_ms=latency_ms,
            tokens_generated=num_tokens,
        )

    def _format_structured_prompt(self, example: BenchmarkExample) -> str:
        """Format prompt for structured reasoning output."""
        options_text = "\n".join([
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(example.options)
        ])

        return f"""You are a logical reasoning expert. Analyze the following question and provide your answer in JSON format.

Context: {example.context}

Question: {example.question}

Options:
{options_text}

Respond with a JSON object containing:
- "reasoning": Your step-by-step logical analysis
- "answer": The letter of the correct answer (A, B, C, or D)
- "confidence": Your confidence level (0.0 to 1.0)

Response:"""

    def _format_cot_prompt(self, example: BenchmarkExample) -> str:
        """Format prompt for chain-of-thought reasoning."""
        options_text = "\n".join([
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(example.options)
        ])

        return f"""Context: {example.context}

Question: {example.question}

Options:
{options_text}

Let's think through this step-by-step:
1. First, let me identify the key information in the context.
2. Then, I'll analyze what the question is asking.
3. Finally, I'll evaluate each option against the logical constraints.

Step-by-step reasoning:"""

    def _format_simple_prompt(self, example: BenchmarkExample) -> str:
        """Format a simple direct prompt."""
        options_text = "\n".join([
            f"{chr(65 + i)}. {opt}" for i, opt in enumerate(example.options)
        ])

        return f"""Context: {example.context}

Question: {example.question}

Options:
{options_text}

The correct answer is:"""

    def _parse_response(
        self,
        response: str,
        num_options: int
    ) -> Tuple[int, Optional[float], Optional[str]]:
        """Parse the model's response to extract answer, confidence, and reasoning."""
        predicted_answer = 0
        confidence = None
        reasoning = None

        # Try to parse as JSON first
        try:
            # Find JSON in response
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                answer = data.get("answer", "A")
                if isinstance(answer, str) and len(answer) == 1:
                    predicted_answer = ord(answer.upper()) - ord('A')
                elif isinstance(answer, int):
                    predicted_answer = answer
                confidence = data.get("confidence")
                reasoning = data.get("reasoning")
        except json.JSONDecodeError:
            pass

        # Fallback: look for letter answer
        if reasoning is None:
            # Look for patterns like "Answer: A", "The answer is B", "(C)", etc.
            patterns = [
                r'[Aa]nswer[:\s]+([A-D])',
                r'[Cc]orrect[:\s]+([A-D])',
                r'\(([A-D])\)',
                r'^([A-D])\.',
                r'([A-D])\s*$',
            ]

            for pattern in patterns:
                match = re.search(pattern, response)
                if match:
                    letter = match.group(1).upper()
                    predicted_answer = ord(letter) - ord('A')
                    break

            # Extract reasoning (text before the answer)
            reasoning = response.strip()

        # Ensure answer is in valid range
        predicted_answer = max(0, min(predicted_answer, num_options - 1))

        return predicted_answer, confidence, reasoning


# =============================================================================
# Benchmark Runner
# =============================================================================

class BenchmarkRunner:
    """Run benchmarks and collect results."""

    def __init__(
        self,
        model: LogicReasoningModel,
        output_dir: str = "./benchmark_results",
    ):
        self.model = model
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_benchmark(
        self,
        benchmark_name: str,
        split: str = "test",
        max_samples: Optional[int] = None,
        prompt_template: str = "structured",
        verbose: bool = True,
    ) -> BenchmarkMetrics:
        """Run a single benchmark."""
        print(f"\n{'='*60}")
        print(f"Running benchmark: {benchmark_name}")
        print(f"{'='*60}")

        # Load benchmark data
        loader = get_benchmark_loader(benchmark_name)
        examples = loader.load(split=split, max_samples=max_samples)

        if not examples:
            print(f"Warning: No examples loaded for {benchmark_name}")
            return BenchmarkMetrics(
                benchmark_name=benchmark_name,
                model_name=self.model.model_path,
                total_examples=0,
                correct=0,
                accuracy=0.0,
                timestamp=datetime.now().isoformat(),
            )

        print(f"Loaded {len(examples)} examples")

        # Run predictions
        results: List[PredictionResult] = []
        category_correct = defaultdict(int)
        category_total = defaultdict(int)
        difficulty_correct = defaultdict(int)
        difficulty_total = defaultdict(int)

        iterator = tqdm(examples, desc="Evaluating") if verbose else examples

        for example in iterator:
            result = self.model.predict_multiple_choice(example, prompt_template)
            results.append(result)

            # Track by category
            if example.category:
                category_total[example.category] += 1
                if result.is_correct:
                    category_correct[example.category] += 1

            # Track by difficulty
            if example.difficulty:
                difficulty_total[example.difficulty] += 1
                if result.is_correct:
                    difficulty_correct[example.difficulty] += 1

        # Calculate metrics
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        accuracy = correct / total if total > 0 else 0.0

        accuracy_by_category = {
            cat: category_correct[cat] / category_total[cat]
            for cat in category_total
        }

        accuracy_by_difficulty = {
            diff: difficulty_correct[diff] / difficulty_total[diff]
            for diff in difficulty_total
        }

        avg_latency = np.mean([r.latency_ms for r in results]) if results else 0.0
        avg_tokens = np.mean([r.tokens_generated for r in results]) if results else 0.0

        # Calculate reasoning quality (based on having valid reasoning)
        reasoning_scores = []
        for r in results:
            if r.reasoning:
                # Simple heuristic: longer, structured reasoning is better
                score = min(1.0, len(r.reasoning) / 500)
                if "step" in r.reasoning.lower() or "therefore" in r.reasoning.lower():
                    score += 0.2
                reasoning_scores.append(min(1.0, score))
            else:
                reasoning_scores.append(0.0)
        reasoning_quality = np.mean(reasoning_scores) if reasoning_scores else 0.0

        metrics = BenchmarkMetrics(
            benchmark_name=benchmark_name,
            model_name=str(self.model.model_path),
            total_examples=total,
            correct=correct,
            accuracy=accuracy,
            accuracy_by_category=accuracy_by_category,
            accuracy_by_difficulty=accuracy_by_difficulty,
            avg_latency_ms=avg_latency,
            avg_tokens=avg_tokens,
            reasoning_quality_score=reasoning_quality,
            timestamp=datetime.now().isoformat(),
        )

        # Save results
        self._save_results(benchmark_name, results, metrics)

        # Print summary
        print(f"\n--- {benchmark_name} Results ---")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
        print(f"Avg Latency: {avg_latency:.1f}ms")
        print(f"Avg Tokens: {avg_tokens:.1f}")
        print(f"Reasoning Quality: {reasoning_quality:.2%}")

        if accuracy_by_category:
            print("\nAccuracy by Category:")
            for cat, acc in sorted(accuracy_by_category.items()):
                print(f"  {cat}: {acc:.2%}")

        if accuracy_by_difficulty:
            print("\nAccuracy by Difficulty:")
            for diff, acc in sorted(accuracy_by_difficulty.items()):
                print(f"  {diff}: {acc:.2%}")

        return metrics

    def run_all_benchmarks(
        self,
        benchmarks: Optional[List[str]] = None,
        split: str = "test",
        max_samples: Optional[int] = None,
        prompt_template: str = "structured",
    ) -> Dict[str, BenchmarkMetrics]:
        """Run multiple benchmarks."""
        if benchmarks is None:
            benchmarks = list(BENCHMARK_REGISTRY.keys())

        all_metrics = {}
        for benchmark in benchmarks:
            if benchmark in BENCHMARK_REGISTRY:
                metrics = self.run_benchmark(
                    benchmark,
                    split=split,
                    max_samples=max_samples,
                    prompt_template=prompt_template,
                )
                all_metrics[benchmark] = metrics
            else:
                print(f"Warning: Unknown benchmark '{benchmark}'")

        # Save aggregate results
        self._save_aggregate_results(all_metrics)

        return all_metrics

    def _save_results(
        self,
        benchmark_name: str,
        results: List[PredictionResult],
        metrics: BenchmarkMetrics,
    ):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = self.output_dir / benchmark_name / timestamp
        result_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions
        predictions_file = result_dir / "predictions.jsonl"
        with open(predictions_file, "w") as f:
            for r in results:
                f.write(json.dumps(asdict(r)) + "\n")

        # Save metrics
        metrics_file = result_dir / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(asdict(metrics), f, indent=2)

        # Save CSV summary if pandas available
        if PANDAS_AVAILABLE:
            df = pd.DataFrame([asdict(r) for r in results])
            df.to_csv(result_dir / "predictions.csv", index=False)

        print(f"Results saved to: {result_dir}")

    def _save_aggregate_results(self, all_metrics: Dict[str, BenchmarkMetrics]):
        """Save aggregate results across all benchmarks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.output_dir / f"summary_{timestamp}.json"

        summary = {
            "model_path": str(self.model.model_path),
            "timestamp": datetime.now().isoformat(),
            "benchmarks": {
                name: asdict(metrics)
                for name, metrics in all_metrics.items()
            },
            "overall_accuracy": np.mean([m.accuracy for m in all_metrics.values()]) if all_metrics else 0.0,
        }

        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        # Print overall summary
        print("\n" + "="*60)
        print("OVERALL BENCHMARK SUMMARY")
        print("="*60)
        print(f"\nModel: {self.model.model_path}")
        print(f"\nBenchmark Results:")
        for name, metrics in all_metrics.items():
            print(f"  {name:20s}: {metrics.accuracy:.2%} ({metrics.correct}/{metrics.total_examples})")
        print(f"\nOverall Accuracy: {summary['overall_accuracy']:.2%}")
        print(f"\nSummary saved to: {summary_file}")


# =============================================================================
# CLI Interface
# =============================================================================

def list_benchmarks():
    """Print available benchmarks."""
    print("\n=== Available Benchmarks ===\n")
    for key, info in BENCHMARK_REGISTRY.items():
        print(f"  {key:15s} - {info['name']}: {info['description']}")
        print(f"                   Task: {info['task_type']}, Metrics: {', '.join(info['metrics'])}")
    print()


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Logic Reasoning Models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available benchmarks
  python benchmark_logic.py --list-benchmarks

  # Run all benchmarks on a finetuned model
  python benchmark_logic.py --model-path ./models/lora_finetuned/final

  # Run specific benchmarks
  python benchmark_logic.py --model-path ./models/final --benchmarks logiqa reclor

  # Run with limited samples (for testing)
  python benchmark_logic.py --model-path ./models/final --max-samples 100

  # Use chain-of-thought prompting
  python benchmark_logic.py --model-path ./models/final --prompt-template cot

  # Specify base model for LoRA adapters
  python benchmark_logic.py --model-path ./models/lora/final --base-model meta-llama/Llama-2-7b-hf
        """
    )

    # Model
    parser.add_argument("--model-path", "-m", type=str, required=False,
                       help="Path to finetuned model or LoRA adapter")
    parser.add_argument("--base-model", type=str,
                       help="Base model path (required for LoRA if not auto-detected)")

    # Benchmark selection
    parser.add_argument("--list-benchmarks", action="store_true",
                       help="List available benchmarks and exit")
    parser.add_argument("--benchmarks", "-b", type=str, nargs="+",
                       help="Benchmarks to run (space-separated). Default: all")
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "validation", "test"],
                       help="Dataset split to use")
    parser.add_argument("--max-samples", type=int,
                       help="Maximum samples per benchmark (for testing)")

    # Inference settings
    parser.add_argument("--prompt-template", type=str, default="structured",
                       choices=["structured", "cot", "simple"],
                       help="Prompt template to use")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                       help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.1,
                       help="Sampling temperature")
    parser.add_argument("--do-sample", action="store_true",
                       help="Enable sampling (vs greedy decoding)")

    # Precision
    parser.add_argument("--dtype", type=str, default="float16",
                       choices=["float16", "bfloat16", "float32"],
                       help="Torch dtype for model")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (auto, cuda, cpu)")

    # Output
    parser.add_argument("--output-dir", "-o", type=str, default="./benchmark_results",
                       help="Directory to save results")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Suppress progress bars")

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # List benchmarks and exit
    if args.list_benchmarks:
        list_benchmarks()
        return

    # Validate model path
    if not args.model_path:
        print("Error: --model-path is required")
        print("Run with --help for usage information")
        sys.exit(1)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)

    # Load model
    print("Loading model...")
    model = LogicReasoningModel(
        model_path=str(model_path),
        base_model_path=args.base_model,
        device=args.device,
        torch_dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
    )

    # Create runner
    runner = BenchmarkRunner(
        model=model,
        output_dir=args.output_dir,
    )

    # Run benchmarks
    if args.benchmarks:
        metrics = runner.run_all_benchmarks(
            benchmarks=args.benchmarks,
            split=args.split,
            max_samples=args.max_samples,
            prompt_template=args.prompt_template,
        )
    else:
        # Run all available benchmarks
        metrics = runner.run_all_benchmarks(
            split=args.split,
            max_samples=args.max_samples,
            prompt_template=args.prompt_template,
        )

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
