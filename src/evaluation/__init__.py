"""Evaluation framework for model outputs."""

from .evaluator import ModelEvaluator, EvaluationMetrics, EvaluationConfig
from .benchmark_registry import (
    BenchmarkConfig,
    BenchmarkTier,
    BenchmarkType,
    BenchmarkCategory,
    BENCHMARK_REGISTRY,
    get_benchmark_config,
    get_benchmarks_by_tier,
    list_benchmarks,
)
from .benchmark_evaluator import (
    BenchmarkEvaluator,
    ComparisonResult,
    BenchmarkResult,
    StagedEvalConfig,
    CheckpointManager,
)
from .benchmark_loaders import get_loader, BenchmarkItem
from .report_generator import ReportGenerator

__all__ = [
    # Existing
    "ModelEvaluator",
    "EvaluationMetrics",
    "EvaluationConfig",
    # Benchmark evaluation
    "BenchmarkConfig",
    "BenchmarkTier",
    "BenchmarkType",
    "BenchmarkCategory",
    "BENCHMARK_REGISTRY",
    "get_benchmark_config",
    "get_benchmarks_by_tier",
    "list_benchmarks",
    "BenchmarkEvaluator",
    "ComparisonResult",
    "BenchmarkResult",
    "StagedEvalConfig",
    "CheckpointManager",
    "ReportGenerator",
    # Loaders
    "get_loader",
    "BenchmarkItem",
]
