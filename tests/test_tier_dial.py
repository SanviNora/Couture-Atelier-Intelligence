import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scripts.build_tier_dial import WEEKS, lift_curve, pooled_buy, simulate

from couture_atelier.allocation import demand_distribution

EFFECTS = {"markdown_lift": {"by_depth": [
    {"mean_depth": 0.14, "lift": 1.6, "lift_weeks_3_4": 1.1, "events": 5000},
    {"mean_depth": 0.28, "lift": 3.3, "lift_weeks_3_4": 2.7, "events": 4000},
    {"mean_depth": 0.47, "lift": 3.1, "lift_weeks_3_4": 1.9, "events": 1000},
    {"mean_depth": 0.64, "lift": 2.9, "lift_weeks_3_4": 1.3, "events": 70},
]}}


def test_lift_is_one_without_discount_never_falls_with_depth_and_fades() -> None:
    lift = lift_curve(EFFECTS, "central")
    depths = np.linspace(0, 0.6, 13)
    early = lift(depths, np.zeros(13))

    assert early[0] == 1.0
    assert (np.diff(early) >= 0).all()
    assert (lift(depths, np.full(13, 3)) <= early).all()


def test_conservative_lift_is_never_above_central() -> None:
    depths = np.linspace(0, 0.6, 13)
    ages = np.zeros(13)

    assert (lift_curve(EFFECTS, "conservative")(depths, ages) <= lift_curve(EFFECTS, "central")(depths, ages)).all()


def test_pooled_buy_needs_fewer_units_than_buying_store_by_store() -> None:
    data = pd.DataFrame({"product_id": [1] * 20})
    mean = np.full(20, 2.0)
    n, p = demand_distribution(mean, mean * 0.3, mean * 2.0)

    pooled = pooled_buy(data, n, p, ratio=0.8)
    store_by_store = nbinom.ppf(0.8, n, p).sum()

    assert pooled.sum() < store_by_store
    assert pooled.sum() > mean.sum()


def test_simulation_conserves_stock() -> None:
    rng = np.random.default_rng(0)
    rate = rng.uniform(0.5, 3, 500)
    stock = rng.integers(0, 30, 500).astype(float)
    schedule = np.zeros((500, WEEKS))
    schedule[:, 8:] = 0.2

    out = simulate(rate, np.ones(WEEKS), schedule, stock, lift_curve(EFFECTS, "central"), rng)

    assert np.allclose(out["sold_full"] + out["sold_disc"] + out["left"], stock)
    assert (out["left"] >= 0).all()
