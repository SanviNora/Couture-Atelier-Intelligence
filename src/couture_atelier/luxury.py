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


def calibrate_concentration(
    base: np.ndarray, target_share: float, fraction: float = 0.001
) -> tuple[float, np.ndarray]:
    """Find alpha so that base**alpha (rescaled to the same total) gives the
    top ``fraction`` of clients ``target_share`` of spend.

    The transform is monotone, so client ranking is unchanged, and total
    spend is preserved.
    """
    base = np.asarray(base, dtype=float)
    if (base <= 0).any():
        raise ValueError("Base spend must be positive")
    scaled = base / base.mean()

    def shaped(alpha: float) -> np.ndarray:
        powered = scaled**alpha
        return powered * base.sum() / powered.sum()

    low, high = 1.0, 1.0
    if top_share(base, fraction) >= target_share:
        return 1.0, base.copy()
    while top_share(shaped(high), fraction) < target_share:
        high *= 2
        if high > 64:
            raise ValueError("Target concentration is unreachable")
    for _ in range(60):
        mid = (low + high) / 2
        if top_share(shaped(mid), fraction) < target_share:
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


def consent_status(client_ids: pd.Series, visits: pd.Series, base: float, cap: float, seed: int) -> pd.Series:
    probability = base + (cap - base) * visits.clip(upper=10).sub(1).clip(lower=0) / 9
    draw = stable_uniform(client_ids, seed)
    return pd.Series(np.where(draw < probability, "consented", "not_consented"), index=client_ids.index)
