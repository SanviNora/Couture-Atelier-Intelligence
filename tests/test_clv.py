import numpy as np
import pandas as pd
import pytest

from couture_atelier.client_frames import spend_metrics
from couture_atelier.clv import BGNBD, MBGNBD, fit_bgnbd, fit_gamma_gamma, fit_mbgnbd


def simulate_bgnbd(n: int, r: float, alpha: float, a: float, b: float, weeks: float, seed: int = 1):
    """Draw visit histories from the BG/NBD generative story."""
    rng = np.random.default_rng(seed)
    rate = rng.gamma(r, 1 / alpha, n)
    churn = rng.beta(a, b, n)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        t = 0.0
        while True:
            t += rng.exponential(1 / rate[i])
            if t > weeks:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < churn[i]:
                break
    return x, t_x, np.full(n, weeks)


def test_bgnbd_recovers_simulated_parameters() -> None:
    x, t_x, T = simulate_bgnbd(8_000, r=0.6, alpha=4.0, a=1.2, b=2.5, weeks=52)

    fitted = fit_bgnbd(x, t_x, T)

    for got, want in zip((fitted.r, fitted.alpha, fitted.a, fitted.b), (0.6, 4.0, 1.2, 2.5), strict=True):
        assert got == pytest.approx(want, rel=0.2)


def test_exact_expected_visits_match_closed_form_when_valid() -> None:
    model = BGNBD(0.6, 4.0, 1.4, 2.5)
    x = np.array([0, 1, 3, 10, 0, 5.0])
    t_x = np.array([0, 10, 30, 50, 0, 20.0])
    T = np.array([52, 52, 60, 60, 10, 104.0])

    exact = model.expected_visits(52, x, t_x, T)

    assert np.allclose(exact, model.expected_visits_closed_form(52, x, t_x, T), rtol=1e-6)


def test_exact_expected_visits_work_when_closed_form_does_not() -> None:
    model = BGNBD(0.45, 18.0, 0.3, 1.9)
    x = np.array([0, 2, 6.0])

    visits = model.expected_visits(52, x, np.array([0, 20, 40.0]), np.array([52, 52, 52.0]))

    assert np.isfinite(visits).all() and (visits > 0).all()
    with pytest.raises(ValueError, match="a > 1"):
        model.expected_visits_closed_form(52, x, np.array([0, 20, 40.0]), np.array([52, 52, 52.0]))


def test_p_alive_is_one_for_single_visit_clients() -> None:
    # A known BG/NBD property, and the reason a separate return model is used.
    model = BGNBD(0.45, 18.0, 0.3, 1.9)

    assert model.p_alive(np.array([0.0]), np.array([0.0]), np.array([100.0]))[0] == 1.0


def test_gamma_gamma_recovers_population_mean_spend() -> None:
    rng = np.random.default_rng(3)
    p, q, gamma = 3.0, 4.0, 2_000.0
    nu = rng.gamma(q, 1 / gamma, 6_000)
    x = rng.integers(1, 12, 6_000).astype(float)
    monetary = np.array([rng.gamma(p, 1 / v, int(k)).mean() for v, k in zip(nu, x, strict=True)])

    fitted = fit_gamma_gamma(x, monetary)

    true_mean = p * gamma / (q - 1)
    assert fitted.p * fitted.gamma / (fitted.q - 1) == pytest.approx(true_mean, rel=0.1)


def test_spend_metrics_reward_perfect_ranking() -> None:
    actual = pd.Series(np.arange(1, 1_001, dtype=float))

    perfect = spend_metrics(actual, actual)

    assert perfect["spearman"] == 1.0
    assert perfect["top1pct_precision"] == 1.0
    assert perfect["total_bias_pct"] == 0.0


def simulate_mbgnbd(n: int, r: float, alpha: float, a: float, b: float, weeks: float, seed: int = 2):
    """As BG/NBD, but a client can also drop out right after the first purchase."""
    rng = np.random.default_rng(seed)
    rate = rng.gamma(r, 1 / alpha, n)
    churn = rng.beta(a, b, n)
    x, t_x = np.zeros(n), np.zeros(n)
    for i in range(n):
        if rng.random() < churn[i]:
            continue
        t = 0.0
        while True:
            t += rng.exponential(1 / rate[i])
            if t > weeks:
                break
            x[i] += 1
            t_x[i] = t
            if rng.random() < churn[i]:
                break
    return x, t_x, np.full(n, weeks)


def test_mbgnbd_recovers_parameters_and_fixes_one_time_buyers() -> None:
    x, t_x, T = simulate_mbgnbd(8_000, r=0.6, alpha=4.0, a=1.2, b=2.5, weeks=52)

    fitted = fit_mbgnbd(x, t_x, T)

    for got, want in zip((fitted.r, fitted.alpha, fitted.a, fitted.b), (0.6, 4.0, 1.2, 2.5), strict=True):
        assert got == pytest.approx(want, rel=0.2)
    # Unlike BG/NBD, a long-silent one-time buyer is no longer assumed to be active.
    assert fitted.p_alive(np.array([0.0]), np.array([0.0]), np.array([100.0]))[0] < 0.5


def test_mbgnbd_expected_visits_match_simulation() -> None:
    model = MBGNBD(0.6, 4.0, 1.2, 2.5)
    rng = np.random.default_rng(5)
    n, horizon = 40_000, 26.0
    rate, churn = rng.gamma(0.6, 1 / 4.0, n), rng.beta(1.2, 2.5, n)
    counts = np.zeros(n)
    for i in range(n):
        if rng.random() < churn[i]:
            continue
        t = 0.0
        while True:
            t += rng.exponential(1 / rate[i])
            if t > horizon:
                break
            counts[i] += 1
            if rng.random() < churn[i]:
                break
    # For a brand-new client (x = 0, T = 0) the prediction is the population mean.
    predicted = model.expected_visits(horizon, np.array([0.0]), np.array([0.0]), np.array([1e-9]))[0]
    assert predicted == pytest.approx(counts.mean(), rel=0.05)
