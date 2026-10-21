import numpy as np
import pandas as pd
import pytest

from couture_atelier.luxury import (
    calibrate_trade_up,
    client_tier,
    consent_status,
    load_calibration,
    map_price_ladder,
    price_tier,
    top_share,
)


def receipts_fixture(n_clients: int = 5_000, seed: int = 0):
    rng = np.random.default_rng(seed)
    visits = rng.geometric(0.35, size=n_clients)
    codes = np.repeat(np.arange(n_clients), visits)
    prior = np.concatenate([np.arange(v) for v in visits])
    base = rng.lognormal(mean=8, sigma=0.5, size=len(codes))
    return base, prior, codes


def test_trade_up_hits_target() -> None:
    base, prior, codes = receipts_fixture()

    beta, values = calibrate_trade_up(base, prior, codes, target_share=0.12, fraction=0.01)

    assert beta > 0
    assert top_share(np.bincount(codes, weights=values), 0.01) == pytest.approx(0.12, abs=0.005)


def test_trade_up_uses_only_past_visits_and_levels_off() -> None:
    base, prior, codes = receipts_fixture()
    beta, values = calibrate_trade_up(base, prior, codes, target_share=0.12, fraction=0.01, cap=3)
    lift = values / base

    # A first purchase keeps its list value; later ones depend only on visits before them.
    assert np.allclose(lift[prior == 0], 1.0)
    assert np.allclose(lift[prior == 2], 3**beta)
    assert np.allclose(lift[prior >= 3], 4**beta)


def test_trade_up_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="positive"):
        calibrate_trade_up(np.array([10.0, 0.0]), np.array([0, 1]), np.array([0, 0]), 0.5)
    with pytest.raises(ValueError, match="non-negative"):
        calibrate_trade_up(np.array([10.0, 5.0]), np.array([0, -1]), np.array([0, 0]), 0.5)


def test_price_ladder_is_monotone_and_bounded() -> None:
    index = pd.Series([0.01, 0.05, 0.05, 0.2, 0.57])

    prices = map_price_ladder(index, 650, 9500)

    assert prices.is_monotonic_increasing
    assert prices.between(650, 9500).all()


def test_price_and_client_tiers_cover_everyone() -> None:
    cfg = load_calibration()["assumptions"]
    prices = pd.Series(np.linspace(650, 9500, 100))
    spend = pd.Series(np.arange(1, 10_001, dtype=float))

    assert set(price_tier(prices, cfg["price_tiers"])) == {"entry", "core", "high", "exceptional"}
    tiers = client_tier(spend, cfg["client_tiers"])
    assert tiers.value_counts()["top_tier"] == 10
    assert tiers[spend.idxmax()] == "top_tier"


def test_consent_is_deterministic_and_matches_rate() -> None:
    ids = pd.Series(range(20_000))
    first = consent_status(ids, rate=0.65, seed=42)

    assert first.equals(consent_status(ids, rate=0.65, seed=42))
    assert first.eq("consented").mean() == pytest.approx(0.65, abs=0.02)


def test_calibration_targets_cite_sources() -> None:
    cfg = load_calibration()

    for target in cfg["targets"].values():
        assert target["source"] and target["url"].startswith("https://")
