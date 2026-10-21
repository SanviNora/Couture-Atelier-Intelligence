import numpy as np
import pandas as pd
import pytest

from couture_atelier.luxury import (
    calibrate_concentration,
    client_tier,
    consent_status,
    load_calibration,
    map_price_ladder,
    price_tier,
    top_share,
)


def test_calibration_hits_target_and_keeps_order_and_total() -> None:
    rng = np.random.default_rng(0)
    base = rng.lognormal(mean=6, sigma=0.8, size=20_000)

    alpha, shaped = calibrate_concentration(base, target_share=0.23)

    assert alpha > 1
    assert top_share(shaped, 0.001) == pytest.approx(0.23, abs=0.005)
    assert shaped.sum() == pytest.approx(base.sum())
    assert (np.argsort(shaped) == np.argsort(base)).all()


def test_calibration_rejects_non_positive_spend() -> None:
    with pytest.raises(ValueError, match="positive"):
        calibrate_concentration(np.array([10.0, 0.0]), target_share=0.5)


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


def test_consent_is_deterministic_and_rises_with_visits() -> None:
    ids = pd.Series(range(20_000))
    one_visit = consent_status(ids, pd.Series(1, index=ids.index), 0.55, 0.85, seed=42)
    ten_visits = consent_status(ids, pd.Series(10, index=ids.index), 0.55, 0.85, seed=42)

    assert one_visit.equals(consent_status(ids, pd.Series(1, index=ids.index), 0.55, 0.85, seed=42))
    assert one_visit.eq("consented").mean() == pytest.approx(0.55, abs=0.02)
    assert ten_visits.eq("consented").mean() == pytest.approx(0.85, abs=0.02)


def test_calibration_targets_cite_sources() -> None:
    cfg = load_calibration()

    for target in cfg["targets"].values():
        assert target["source"] and target["url"].startswith("https://")
