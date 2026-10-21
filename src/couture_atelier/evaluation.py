"""Uncertainty for model comparisons: paired bootstrap over clients."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Metric = Callable[[np.ndarray, np.ndarray], float]


def paired_bootstrap(
    metric: Metric,
    y: np.ndarray,
    model: np.ndarray,
    baseline: np.ndarray,
    n_boot: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Difference metric(model) - metric(baseline) with a 95% percentile interval.

    Both scores are evaluated on the same resampled clients each time, so the
    interval reflects the comparison rather than the noise in each metric.
    """
    y, model, baseline = (np.asarray(v) for v in (y, model, baseline))
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = metric(y[idx], model[idx]) - metric(y[idx], baseline[idx])
    point = metric(y, model) - metric(y, baseline)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": round(float(point), 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
            "significant": bool(low > 0 or high < 0)}
