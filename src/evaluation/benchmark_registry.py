"""Benchmark registry with metadata, prompt templates, and evaluation configs."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum


class BenchmarkType(Enum):
    MULTIPLE_CHOICE = "multiple_choice"
    GENERATION = "generation"
    ENTAILMENT = "entailment"


class BenchmarkCategory(Enum):
    STANDARD = "standard"
    LOGIC = "logic"


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    name: str
    key: str
    hf_dataset: str
    hf_subset: Optional[str] = None
    split: str = "test"
    few_shot_split: str = "validation"
    category: BenchmarkCategory = BenchmarkCategory.STANDARD
    benchmark_type: BenchmarkType = BenchmarkType.MULTIPLE_CHOICE
    num_few_shot: int = 5
    num_choices: int = 4
    description: str = ""
    random_baseline: float = 0.25
    metrics: List[str] = field(default_factory=lambda: ["accuracy"])


BENCHMARK_REGISTRY: Dict[str, BenchmarkConfig] = {
    # -------------------------------------------------------------------------
    # Standard benchmarks
    # -------------------------------------------------------------------------
    "mmlu": BenchmarkConfig(
        name="MMLU",
        key="mmlu",
        hf_dataset="cais/mmlu",
        hf_subset="all",
        split="test",
        few_shot_split="dev",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=4,
        description="Massive Multitask Language Understanding",
        random_baseline=0.25,
    ),
    "arc_challenge": BenchmarkConfig(
        name="ARC-Challenge",
        key="arc_challenge",
        hf_dataset="allenai/ai2_arc",
        hf_subset="ARC-Challenge",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=25,
        num_choices=4,
        description="AI2 Reasoning Challenge (Challenge set)",
        random_baseline=0.25,
    ),
    "hellaswag": BenchmarkConfig(
        name="HellaSwag",
        key="hellaswag",
        hf_dataset="Rowan/hellaswag",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=10,
        num_choices=4,
        description="Grounded Commonsense Inference",
        random_baseline=0.25,
    ),
    "winogrande": BenchmarkConfig(
        name="Winogrande",
        key="winogrande",
        hf_dataset="allenai/winogrande",
        hf_subset="winogrande_xl",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=2,
        description="Winograd Schema Challenge (large)",
        random_baseline=0.5,
    ),
    "gsm8k": BenchmarkConfig(
        name="GSM8K",
        key="gsm8k",
        hf_dataset="openai/gsm8k",
        hf_subset="main",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.GENERATION,
        num_few_shot=8,
        num_choices=0,
        description="Grade School Math",
        random_baseline=0.0,
    ),
    "truthfulqa": BenchmarkConfig(
        name="TruthfulQA",
        key="truthfulqa",
        hf_dataset="truthfulqa/truthful_qa",
        hf_subset="multiple_choice",
        split="validation",
        few_shot_split="validation",
        category=BenchmarkCategory.STANDARD,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=6,
        num_choices=4,
        description="TruthfulQA MC2",
        random_baseline=0.25,
    ),
    # -------------------------------------------------------------------------
    # Logic benchmarks
    # -------------------------------------------------------------------------
    "logiqa": BenchmarkConfig(
        name="LogiQA",
        key="logiqa",
        hf_dataset="datatune/LogiQA2.0",
        split="test",
        few_shot_split="train",
        category=BenchmarkCategory.LOGIC,
        benchmark_type=BenchmarkType.MULTIPLE_CHOICE,
        num_few_shot=5,
        num_choices=4,
        description="Logic Question Answering",
        random_baseline=0.25,
    ),
    "folio": BenchmarkConfig(
        name="FOLIO",
        key="folio",
        hf_dataset="tasksource/folio",
        split="validation",
        few_shot_split="train",
        category=BenchmarkCategory.LOGIC,
        benchmark_type=BenchmarkType.ENTAILMENT,
        num_few_shot=5,
        num_choices=3,
        description="First-Order Logic Reasoning",
        random_baseline=1.0 / 3.0,
    ),
}


def get_benchmark_config(name: str) -> BenchmarkConfig:
    """Get benchmark configuration by key."""
    if name not in BENCHMARK_REGISTRY:
        available = ", ".join(BENCHMARK_REGISTRY.keys())
        raise ValueError(f"Unknown benchmark: {name}. Available: {available}")
    return BENCHMARK_REGISTRY[name]


def list_benchmarks() -> Dict[str, str]:
    """List all available benchmarks with descriptions."""
    return {k: f"{v.name} - {v.description}" for k, v in BENCHMARK_REGISTRY.items()}
