"""Demand recovery when stock runs out, and helpers for demand forecasting.

When a store sells out, recorded sales are a lower bound on what clients
wanted. Given a model's expected demand mu for that week, the recovered demand
is the expected demand conditional on reaching the stock level:

    E[D | D >= c] = mu * P(D >= c - 1) / P(D >= c)     for D ~ Poisson(mu)

which follows from E[D * 1{D >= c}] = mu * P(D >= c - 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import poisson


def conditional_demand(mu: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Expected Poisson demand given that it was at least ``cap`` (the units that sold)."""
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-9)
    cap = np.asarray(cap, dtype=float)
    upper = poisson.sf(cap - 2, mu)  # P(D >= c - 1)
    lower = poisson.sf(cap - 1, mu)  # P(D >= c)
    with np.errstate(divide="ignore", invalid="ignore"):
        value = mu * upper / lower
    # Far in the tail the ratio is numerically unstable; demand is then just above the cap.
    value = np.where((lower < 1e-12) | ~np.isfinite(value), cap + 1, value)
    return np.maximum(value, cap)


def recover(sold: np.ndarray, censored: np.ndarray, mu: np.ndarray, method: str = "conditional") -> np.ndarray:
    """Demand estimate per week: sales where stock lasted, an estimate where it ran out."""
    sold = np.asarray(sold, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if method == "naive":
        return sold
    if method == "max":
        estimate = np.maximum(sold, mu)
    elif method == "conditional":
        estimate = conditional_demand(mu, sold)
    else:
        raise ValueError(f"Unknown method {method}")
    return np.where(censored, estimate, sold)


def artificial_censoring(
    true_demand: np.ndarray, expected: np.ndarray, share: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Cap a random share of weeks at a stock level set before the week.

    Like real stock, the cap depends only on expected demand (between 0 and
    twice the expectation), not on what clients actually wanted that week.
    Returns observed sales and the censored flag, so a recovery method can be
    scored against the demand it is trying to recover.
    """
    rng = np.random.default_rng(seed)
    demand = np.asarray(true_demand, dtype=float)
    capped = rng.random(len(demand)) < share
    stock = np.floor(np.asarray(expected, dtype=float) * rng.uniform(0, 2, len(demand)))
    observed = np.where(capped, np.minimum(demand, stock), demand)
    censored = capped & (demand >= stock)
    return observed, censored


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = np.asarray(y, dtype=float) - np.asarray(pred, dtype=float)
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def wape(y: np.ndarray, pred: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(np.abs(y - np.asarray(pred, dtype=float)).sum() / y.sum())


def coverage(y: np.ndarray, low: np.ndarray, high: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    return float(((y >= low) & (y <= high)).mean())


def category_codes(frame: pd.DataFrame, columns: list[str], vocab: dict[str, list] | None = None):
    """Integer codes for categorical columns, with a fixed vocabulary learned on training rows."""
    vocab = vocab or {c: sorted(frame[c].dropna().unique()) for c in columns}
    coded = frame.copy()
    for c in columns:
        coded[f"{c}_code"] = pd.Categorical(frame[c], categories=vocab[c]).codes
    return coded, vocab
