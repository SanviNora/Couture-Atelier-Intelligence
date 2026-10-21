import itertools

import numpy as np
import pandas as pd
import pytest
from scipy.stats import nbinom
from scripts.build_client_allocation import solve_store

from couture_atelier.allocation import (
    critical_ratio,
    demand_distribution,
    expected_sales,
    greedy_allocation,
    leftover_value,
    proportional_allocation,
)

ECON = {"leftover_routes": {
    "destroy": {"value": -0.02},
    "carry_over": {"sell_through_next_season": 0.4, "desirability_decay": 0.3, "holding_cost": 0.05,
                   "brand_dilution_cost": 0.10},
    "certified_pre_owned": {"value": 0.30},
    "recycle": {"value": -0.03},
}}


def test_demand_distribution_keeps_the_forecast_mean() -> None:
    mean = np.array([3.0, 6.0, 1.0])
    n, p = demand_distribution(mean, mean * 0.4, mean * 1.8)

    assert np.allclose(nbinom.mean(n, p), mean)


def test_greedy_allocation_matches_brute_force() -> None:
    n, p = demand_distribution(np.array([3.0, 6.0, 1.0, 2.5]), np.array([1, 3, 0.2, 0.5]), np.array([6, 10, 2.5, 6]))
    cap = np.array([6, 6, 6, 6])
    greedy = greedy_allocation(8, n, p, cap)
    best = max((a for a in itertools.product(range(7), repeat=4) if sum(a) == 8),
               key=lambda a: expected_sales(np.array(a), n, p).sum())

    assert expected_sales(greedy, n, p).sum() == pytest.approx(expected_sales(np.array(best), n, p).sum())


def test_greedy_respects_minimums_and_caps() -> None:
    n, p = demand_distribution(np.array([10.0, 0.1]), np.array([8.0, 0.05]), np.array([12.0, 0.2]))

    # Store B barely sells but must show the piece; store A takes the rest up to its cap.
    units = greedy_allocation(5, n, p, cap=np.array([4, 4]), minimum=np.array([0, 1]))

    assert units.tolist() == [4, 1]


def test_proportional_allocation_sums_and_caps() -> None:
    units = proportional_allocation(10, np.array([5.0, 3.0, 2.0, 0.0]), cap=np.array([4, 10, 10, 10]))

    assert units.sum() == 10
    assert units[0] == 4


def test_newsvendor_and_leftover_routes() -> None:
    assert critical_ratio(1.0, 0.3, -0.02) == pytest.approx(0.7 / 1.02)
    assert critical_ratio(1.0, 0.3, 0.3) == 1.0
    assert leftover_value(ECON, 1.0, 0.3, "carry_over") == pytest.approx(0.4 * 0.7 - 0.15)
    assert leftover_value(ECON, 1.0, 0.3, "destroy") < 0


def test_client_allocation_rules_hold() -> None:
    cands = pd.DataFrame({
        "client_id": [1, 1, 2, 3, 4, 5],
        "product_id": [10, 11, 10, 10, 11, 11],
        "is_rising": [False, False, False, True, False, True],
        "total_value": [9.0, 8.0, 7.0, 1.0, 6.0, 0.5],
    })

    chosen = solve_store(cands, {10: 2, 11: 2}, reserve=0.5, objective="total_value")
    picked = cands[chosen]

    assert picked.groupby("client_id").size().max() == 1
    assert (picked.groupby("product_id").size() <= 2).all()
    assert picked["is_rising"].sum() >= 2
    assert len(picked) == 4
