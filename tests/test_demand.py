import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import PoissonRegressor

from couture_atelier.demand import (
    artificial_censoring,
    conditional_demand,
    coverage,
    em_recover,
    pinball_loss,
    recover,
    wape,
)


@pytest.mark.parametrize(("mu", "cap"), [(2.0, 3), (5.0, 2), (1.0, 0), (3.0, 8)])
def test_conditional_demand_matches_simulation(mu: float, cap: int) -> None:
    draws = np.random.default_rng(0).poisson(mu, 1_000_000)

    expected = conditional_demand(np.array([mu]), np.array([cap]))[0]

    assert expected == pytest.approx(draws[draws >= cap].mean(), rel=0.01)


def test_conditional_demand_never_below_sales_even_in_the_far_tail() -> None:
    result = conditional_demand(np.array([0.5, 1.0]), np.array([40, 3]))

    assert (result >= np.array([40, 3])).all()
    assert np.isfinite(result).all()


def test_recovery_only_changes_censored_weeks() -> None:
    sold = np.array([2.0, 3.0, 0.0])
    censored = np.array([False, True, True])
    mu = np.array([5.0, 5.0, 1.5])

    out = recover(sold, censored, mu, "conditional")

    assert out[0] == 2.0
    assert out[1] > 3.0
    assert out[2] == pytest.approx(1.5)
    assert recover(sold, censored, mu, "naive").tolist() == sold.tolist()


def test_conditional_recovery_is_unbiased_when_stock_is_set_before_the_week() -> None:
    rng = np.random.default_rng(1)
    mu = rng.uniform(0.5, 4.0, 200_000)
    demand = rng.poisson(mu)

    observed, censored = artificial_censoring(demand, mu, share=0.5, seed=2)
    recovered = recover(observed, censored, mu, "conditional")

    assert recovered[censored].sum() / demand[censored].sum() == pytest.approx(1.0, abs=0.01)
    assert observed[censored].sum() < demand[censored].sum()


def test_forecast_metrics() -> None:
    y = np.array([1.0, 2.0, 3.0])

    assert wape(y, y) == 0
    assert pinball_loss(y, y, 0.9) == 0
    assert pinball_loss(y, y - 1, 0.9) > pinball_loss(y, y + 1, 0.9)
    assert coverage(y, y - 0.5, y + 0.5) == 1.0


def test_em_recovers_more_than_a_model_trained_only_on_uncensored_weeks() -> None:
    rng = np.random.default_rng(3)
    n = 40_000
    x = rng.uniform(0, 2, n)
    demand = rng.poisson(np.exp(0.3 + 0.8 * x))
    stock = np.floor(np.exp(0.3 + 0.8 * x) * rng.uniform(0.3, 1.2, n))
    frame = pd.DataFrame({"x": x, "units_sold": np.minimum(demand, stock), "censored": demand >= stock})

    def fit(X, y):
        return PoissonRegressor(alpha=0).fit(X, y)

    one_pass = fit(frame.loc[~frame.censored, ["x"]], frame.loc[~frame.censored, "units_sold"])
    first = recover(frame.units_sold, frame.censored, one_pass.predict(frame[["x"]]))
    em, path = em_recover(frame, ["x"], fit, iterations=8)

    truth = demand[frame.censored].sum()
    assert abs(em[frame.censored].sum() / truth - 1) < abs(first[frame.censored].sum() / truth - 1)
    assert path == sorted(path)  # recovered demand rises monotonically towards the truth
