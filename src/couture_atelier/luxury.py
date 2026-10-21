"""Pure functions that turn observed fashion retail data into a calibrated luxury layer.

Every transform keeps the observed ordering (prices, client activity) and
changes only scale, so the real behaviour underneath stays intact.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from couture_atelier.paths import CONFIGS


def load_calibration(path: Path = CONFIGS / "calibration.yml") -> dict:
    return yaml.safe_load(path.read_text())


def stable_uniform(keys: pd.Series, seed: int) -> np.ndarray:
    """Deterministic pseudo-random numbers in [0, 1) keyed by id and seed."""
    def one(value: object) -> float:
        digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
        return int(digest[:13], 16) / 16**13

    return keys.map(one).to_numpy()


def map_price_ladder(price_index: pd.Series, low: float, high: float) -> pd.Series:
    """Map a price index onto [low, high] euros by percentile on a log scale."""
    pct = price_index.rank(pct=True, method="average")
    return np.exp(np.log(low) + pct * (np.log(high) - np.log(low))).round(-1)


def price_tier(price_eur: pd.Series, cuts: dict[str, float]) -> pd.Series:
    pct = price_eur.rank(pct=True, method="average")
    tier = pd.Series("exceptional", index=price_eur.index)
    tier[pct <= cuts["high"]] = "high"
    tier[pct <= cuts["core"]] = "core"
    tier[pct <= cuts["entry"]] = "entry"
    return tier


def top_share(values: np.ndarray, fraction: float) -> float:
    """Share of the total held by the top ``fraction`` of entries."""
    ordered = np.sort(values)[::-1]
    k = max(1, round(len(ordered) * fraction))
    return float(ordered[:k].sum() / ordered.sum())


def calibrate_trade_up(
    base: np.ndarray,
    prior_visits: np.ndarray,
    client_codes: np.ndarray,
    target_share: float,
    fraction: float = 0.001,
) -> tuple[float, np.ndarray]:
    """Find beta so that receipt values ``base * (1 + prior_visits) ** beta``
    give the top ``fraction`` of clients ``target_share`` of total spend.

    Each receipt's value depends only on visits that happened before it, so
    the layer never encodes a client's future. One global beta is calibrated,
    and the result is rescaled to keep total spend unchanged.
    """
    base = np.asarray(base, dtype=float)
    prior_visits = np.asarray(prior_visits, dtype=float)
    if (base <= 0).any():
        raise ValueError("Base spend must be positive")
    if (prior_visits < 0).any():
        raise ValueError("Prior visits must be non-negative")

    def shaped(beta: float) -> np.ndarray:
        values = base * (1 + prior_visits) ** beta
        return values * base.sum() / values.sum()

    def share(beta: float) -> float:
        return top_share(np.bincount(client_codes, weights=shaped(beta)), fraction)

    if share(0.0) >= target_share:
        return 0.0, base.copy()
    low, high = 0.0, 1.0
    while share(high) < target_share:
        high *= 2
        if high > 16:
            raise ValueError("Target concentration is unreachable")
    for _ in range(50):
        mid = (low + high) / 2
        if share(mid) < target_share:
            low = mid
        else:
            high = mid
    return high, shaped(high)


def client_tier(spend: pd.Series, cuts: dict[str, float]) -> pd.Series:
    rank = spend.rank(ascending=False, method="first") / len(spend)
    tier = pd.Series("aspirational", index=spend.index)
    tier[rank <= cuts["core"]] = "core"
    tier[rank <= cuts["vic"]] = "vic"
    tier[rank <= cuts["top_tier"]] = "top_tier"
    return tier


def consent_status(client_ids: pd.Series, rate: float, seed: int) -> pd.Series:
    """Opt-in status drawn independently of behaviour, so it cannot leak future visits."""
    draw = stable_uniform(client_ids, seed)
    return pd.Series(np.where(draw < rate, "consented", "not_consented"), index=client_ids.index)
