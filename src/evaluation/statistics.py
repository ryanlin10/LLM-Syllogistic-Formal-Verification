"""Statistical utilities for benchmark evaluation."""

import numpy as np
from typing import List, Tuple


def bootstrap_confidence_interval(
    correctness: List[bool],
    confidence_level: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for accuracy.

    Args:
        correctness: List of per-item correctness booleans.
        confidence_level: Confidence level (default 0.95 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for reproducibility.

    Returns:
        (lower, upper) bounds of the confidence interval.
    """
    if len(correctness) == 0:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    arr = np.array(correctness, dtype=float)
    n = len(arr)

    bootstrap_means = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        bootstrap_means[i] = np.mean(sample)

    alpha = 1 - confidence_level
    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))
    return lower, upper


def compute_normalized_score(accuracy: float, random_baseline: float) -> float:
    """Compute normalized score adjusting for random-chance baseline.

    normalized = (accuracy - baseline) / (1.0 - baseline)

    Args:
        accuracy: Raw accuracy (0-1).
        random_baseline: Expected accuracy from random guessing.

    Returns:
        Normalized score, clipped to 0 if accuracy <= baseline.
    """
    if random_baseline >= 1.0:
        return accuracy
    if accuracy <= random_baseline:
        return 0.0
    return (accuracy - random_baseline) / (1.0 - random_baseline)


def compute_delta_significance(
    base_correctness: List[bool],
    finetuned_correctness: List[bool],
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Tuple[float, float, float, float]:
    """Compute significance of accuracy delta via paired bootstrap.

    Args:
        base_correctness: Per-item correctness for the base model.
        finetuned_correctness: Per-item correctness for the finetuned model.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        (delta, p_value, ci_lower, ci_upper) for the accuracy difference
        (finetuned - base).
    """
    base = np.array(base_correctness, dtype=float)
    finetuned = np.array(finetuned_correctness, dtype=float)
    n = len(base)

    observed_delta = float(np.mean(finetuned) - np.mean(base))

    rng = np.random.RandomState(seed)
    bootstrap_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        bootstrap_deltas[i] = np.mean(finetuned[idx]) - np.mean(base[idx])

    ci_lower = float(np.percentile(bootstrap_deltas, 2.5))
    ci_upper = float(np.percentile(bootstrap_deltas, 97.5))

    # Two-sided p-value: fraction of bootstrap deltas on opposite side of zero
    if observed_delta >= 0:
        p_value = float(np.mean(bootstrap_deltas <= 0))
    else:
        p_value = float(np.mean(bootstrap_deltas >= 0))

    return observed_delta, p_value, ci_lower, ci_upper
