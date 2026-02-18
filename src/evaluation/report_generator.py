"""Generate human-readable benchmark comparison reports."""

import json
from typing import List, Dict, Any
from datetime import datetime
from dataclasses import asdict

from .benchmark_evaluator import ComparisonResult, BenchmarkResult
from .statistics import compute_delta_significance


class ReportGenerator:
    """Generates text and JSON reports from benchmark comparison results."""

    def __init__(
        self,
        base_model_name: str,
        lora_adapter_path: str,
        comparisons: List[ComparisonResult],
    ):
        self.base_model_name = base_model_name
        self.lora_adapter_path = lora_adapter_path
        self.comparisons = comparisons

    def generate_text_report(self) -> str:
        """Generate a comprehensive text comparison report."""
        lines = []
        w = 80  # report width

        # Header
        lines.append("=" * w)
        lines.append("BENCHMARK EVALUATION: BASE vs FINETUNED MODEL COMPARISON")
        lines.append("=" * w)
        lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Base model: {self.base_model_name}")
        lines.append(f"LoRA adapter: {self.lora_adapter_path}")
        lines.append(f"Benchmarks evaluated: {len(self.comparisons)}")
        lines.append("")

        # Summary table
        lines.append("-" * w)
        lines.append("SUMMARY")
        lines.append("-" * w)
        header = (
            f"{'Benchmark':<18} {'Base':>10} {'Finetuned':>10} "
            f"{'Delta':>10} {'Direction':>12}"
        )
        lines.append(header)
        lines.append("-" * w)

        for comp in self.comparisons:
            direction = self._direction_label(comp)
            lines.append(
                f"{comp.benchmark_name:<18} "
                f"{comp.base_result.accuracy:>9.1%} "
                f"{comp.finetuned_result.accuracy:>9.1%} "
                f"{comp.accuracy_delta:>+9.1%} "
                f"{direction:>12}"
            )

        lines.append("-" * w)

        # Averages
        if self.comparisons:
            avg_base = sum(c.base_result.accuracy for c in self.comparisons) / len(
                self.comparisons
            )
            avg_ft = sum(
                c.finetuned_result.accuracy for c in self.comparisons
            ) / len(self.comparisons)
            lines.append(
                f"{'AVERAGE':<18} {avg_base:>9.1%} {avg_ft:>9.1%} "
                f"{avg_ft - avg_base:>+9.1%}"
            )

            # Category averages
            standard = [c for c in self.comparisons if c.category == "standard"]
            logic = [c for c in self.comparisons if c.category == "logic"]

            if standard:
                avg_s_base = sum(c.base_result.accuracy for c in standard) / len(
                    standard
                )
                avg_s_ft = sum(c.finetuned_result.accuracy for c in standard) / len(
                    standard
                )
                lines.append(
                    f"{'  Standard Avg':<18} {avg_s_base:>9.1%} {avg_s_ft:>9.1%} "
                    f"{avg_s_ft - avg_s_base:>+9.1%}"
                )

            if logic:
                avg_l_base = sum(c.base_result.accuracy for c in logic) / len(logic)
                avg_l_ft = sum(c.finetuned_result.accuracy for c in logic) / len(logic)
                lines.append(
                    f"{'  Logic Avg':<18} {avg_l_base:>9.1%} {avg_l_ft:>9.1%} "
                    f"{avg_l_ft - avg_l_base:>+9.1%}"
                )

        lines.append("")

        # Detailed per-benchmark sections
        for comp in self.comparisons:
            lines.extend(self._format_benchmark_detail(comp, w))

        # Timing summary
        lines.append("")
        lines.append("-" * w)
        lines.append("TIMING AND THROUGHPUT")
        lines.append("-" * w)
        for comp in self.comparisons:
            b = comp.base_result
            f = comp.finetuned_result
            lines.append(f"{comp.benchmark_name}:")
            lines.append(
                f"  Base:      {b.total_time_seconds:>7.1f}s  "
                f"({b.throughput_items_per_sec:.1f} items/sec)  "
                f"avg latency: {b.avg_latency_seconds * 1000:.0f}ms"
            )
            lines.append(
                f"  Finetuned: {f.total_time_seconds:>7.1f}s  "
                f"({f.throughput_items_per_sec:.1f} items/sec)  "
                f"avg latency: {f.avg_latency_seconds * 1000:.0f}ms"
            )

        lines.append("")
        lines.append("=" * w)
        lines.append("END OF REPORT")
        lines.append("=" * w)

        return "\n".join(lines)

    def generate_json_report(self) -> Dict[str, Any]:
        """Generate a JSON-serializable report dict."""
        benchmarks = {}
        for comp in self.comparisons:
            # Strip individual_results for JSON to keep size manageable
            base_dict = self._result_to_dict(comp.base_result)
            ft_dict = self._result_to_dict(comp.finetuned_result)

            benchmarks[comp.benchmark_key] = {
                "benchmark_name": comp.benchmark_name,
                "category": comp.category,
                "base": base_dict,
                "finetuned": ft_dict,
                "accuracy_delta": comp.accuracy_delta,
                "is_improvement": comp.is_improvement,
            }

        # Averages
        avg_base = (
            sum(c.base_result.accuracy for c in self.comparisons)
            / len(self.comparisons)
            if self.comparisons
            else 0.0
        )
        avg_ft = (
            sum(c.finetuned_result.accuracy for c in self.comparisons)
            / len(self.comparisons)
            if self.comparisons
            else 0.0
        )

        return {
            "metadata": {
                "base_model": self.base_model_name,
                "lora_adapter": self.lora_adapter_path,
                "timestamp": datetime.now().isoformat(),
                "num_benchmarks": len(self.comparisons),
            },
            "summary": {
                "average_base_accuracy": avg_base,
                "average_finetuned_accuracy": avg_ft,
                "average_delta": avg_ft - avg_base,
            },
            "benchmarks": benchmarks,
        }

    def save_text(self, output_path: str):
        """Save text report to file."""
        report = self.generate_text_report()
        with open(output_path, "w") as f:
            f.write(report)

    def save_json(self, output_path: str):
        """Save JSON report to file."""
        report = self.generate_json_report()
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _direction_label(comp: ComparisonResult) -> str:
        if comp.accuracy_delta > 0.001:
            return "IMPROVED"
        elif comp.accuracy_delta < -0.001:
            return "REGRESSED"
        return "SAME"

    def _format_benchmark_detail(
        self, comp: ComparisonResult, w: int
    ) -> List[str]:
        """Format detailed results for one benchmark."""
        lines = []
        lines.append("=" * w)
        lines.append(f"BENCHMARK: {comp.benchmark_name} ({comp.category})")
        lines.append("=" * w)

        # Delta significance
        base_correct = [r.correct for r in comp.base_result.individual_results]
        ft_correct = [r.correct for r in comp.finetuned_result.individual_results]
        if base_correct and ft_correct and len(base_correct) == len(ft_correct):
            delta, p_val, ci_lo, ci_hi = compute_delta_significance(
                base_correct, ft_correct
            )
            sig = "significant" if p_val < 0.05 else "not significant"
            lines.append(
                f"  Delta: {delta:+.2%} (p={p_val:.3f}, {sig})"
            )
            lines.append(f"  Delta 95% CI: [{ci_lo:+.2%}, {ci_hi:+.2%}]")

        for result in [comp.base_result, comp.finetuned_result]:
            lines.append(f"\n  [{result.model_label.upper()}]")
            lines.append(
                f"  Accuracy: {result.accuracy:.2%} "
                f"(95% CI: [{result.accuracy_ci_lower:.2%}, "
                f"{result.accuracy_ci_upper:.2%}])"
            )
            lines.append(f"  Normalized Score: {result.normalized_score:.2%}")
            lines.append(f"  Correct: {result.num_correct}/{result.num_total}")
            lines.append(f"  Parse Failures: {result.num_parse_failures}")

            # Per-subject breakdown (skip if only one category)
            if result.per_subject_accuracy and len(result.per_subject_accuracy) > 1:
                lines.append(f"  Per-category breakdown:")
                sorted_subjects = sorted(
                    result.per_subject_accuracy.keys(),
                    key=lambda s: result.per_subject_accuracy[s],
                    reverse=True,
                )
                for subj in sorted_subjects[:20]:  # Limit to top 20
                    acc = result.per_subject_accuracy[subj]
                    cnt = result.per_subject_count[subj]
                    lines.append(f"    {subj:<35} {acc:.2%} ({cnt} items)")
                if len(sorted_subjects) > 20:
                    lines.append(
                        f"    ... and {len(sorted_subjects) - 20} more categories"
                    )

        lines.append("")
        return lines

    @staticmethod
    def _result_to_dict(result: BenchmarkResult) -> Dict[str, Any]:
        """Convert BenchmarkResult to JSON-safe dict without individual_results."""
        return {
            "accuracy": result.accuracy,
            "accuracy_ci_lower": result.accuracy_ci_lower,
            "accuracy_ci_upper": result.accuracy_ci_upper,
            "normalized_score": result.normalized_score,
            "num_total": result.num_total,
            "num_correct": result.num_correct,
            "num_parse_failures": result.num_parse_failures,
            "per_subject_accuracy": result.per_subject_accuracy,
            "per_subject_count": result.per_subject_count,
            "avg_latency_seconds": result.avg_latency_seconds,
            "total_time_seconds": result.total_time_seconds,
            "throughput_items_per_sec": result.throughput_items_per_sec,
            "timestamp": result.timestamp,
        }
