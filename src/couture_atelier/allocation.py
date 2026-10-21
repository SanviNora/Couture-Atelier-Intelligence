"""Allocating scarce pieces across boutiques, and the economics of unsold stock.

Store demand for a piece over the season is modelled as a Gamma-Poisson
(negative binomial): Poisson sales noise around a mean whose own uncertainty
comes from the forecast's 10th-90th percentile spread. Expected sales from
x units are E[min(D, x)], which is concave in x, so giving each next unit to
the store with the highest chance of selling it, P(D >= x + 1), is optimal.
"""

from __future__ import annotations

import heapq

import numpy as np
from scipy.stats import nbinom

Z90 = 1.2816  # standard normal 90th percentile


def demand_distribution(mean: np.ndarray, q10: np.ndarray, q90: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Negative-binomial (n, p) per store from a forecast mean and its 10-90 spread.

    The spread gives a coefficient of variation for the mean (forecast
    uncertainty); Poisson noise is added on top.
    """
    mean = np.maximum(np.asarray(mean, dtype=float), 1e-6)
    cv = np.clip((np.asarray(q90) - np.asarray(q10)) / (2 * Z90 * mean), 0.05, 3.0)
    shape = 1.0 / cv**2               # Gamma shape of the uncertain mean
    p = shape / (shape + mean)        # scipy's nbinom: mean = n (1 - p) / p
    return shape, p


def expected_sales(units: np.ndarray, n: np.ndarray, p: np.ndarray) -> np.ndarray:
    """E[min(D, x)] = sum over k < x of P(D > k)."""
    units = np.asarray(units, dtype=int)
    out = np.zeros(len(units))
    for i, (x, ni, pi) in enumerate(zip(units, n, p, strict=True)):
        if x > 0:
            out[i] = nbinom.sf(np.arange(x), ni, pi).sum()
    return out


def greedy_allocation(total: int, n: np.ndarray, p: np.ndarray, cap: np.ndarray,
                      minimum: np.ndarray | None = None) -> np.ndarray:
    """Optimal allocation of ``total`` units: each next unit to the highest P(D >= x + 1).

    ``minimum`` units are placed first (for example one per flagship); ``cap``
    bounds each store.
    """
    k = len(n)
    units = np.zeros(k, dtype=int) if minimum is None else np.minimum(np.asarray(minimum, dtype=int), cap)
    remaining = int(total - units.sum())
    heap = [(-nbinom.sf(units[i], n[i], p[i]), i) for i in range(k) if units[i] < cap[i]]
    heapq.heapify(heap)
    while remaining > 0 and heap:
        _, i = heapq.heappop(heap)
        units[i] += 1
        remaining -= 1
        if units[i] < cap[i]:
            heapq.heappush(heap, (-nbinom.sf(units[i], n[i], p[i]), i))
    return units


def proportional_allocation(total: int, weights: np.ndarray, cap: np.ndarray) -> np.ndarray:
    """Split ``total`` in proportion to ``weights`` (largest remainders), respecting caps."""
    weights = np.maximum(np.asarray(weights, dtype=float), 0)
    if weights.sum() == 0:
        weights = np.ones_like(weights)
    raw = total * weights / weights.sum()
    units = np.minimum(np.floor(raw).astype(int), cap)
    remainder = raw - units
    while units.sum() < total and (units < cap).any():
        order = np.argsort(-np.where(units < cap, remainder, -np.inf))
        for i in order[: total - units.sum()]:
            if units[i] < cap[i]:
                units[i] += 1
        remainder = np.zeros_like(remainder) + (units < cap)
    return units


def critical_ratio(price: float, cost: float, salvage: float) -> float:
    """Newsvendor service level: (underage cost) / (underage + overage cost)."""
    underage = price - cost
    overage = cost - salvage
    return underage / (underage + overage)


def leftover_value(cfg: dict, price: float, cost: float, route: str) -> float:
    """Net value of one unsold unit sent down ``route``, in euros."""
    routes = cfg["leftover_routes"]
    if route == "destroy":
        return routes["destroy"]["value"] * price
    if route == "carry_over":
        r = routes["carry_over"]
        return (r["sell_through_next_season"] * price * (1 - r["desirability_decay"])
                - (r["holding_cost"] + r["brand_dilution_cost"]) * price)
    if route == "certified_pre_owned":
        return routes["certified_pre_owned"]["value"] * price
    if route == "recycle":
        return routes["recycle"]["value"] * price
    raise ValueError(route)
