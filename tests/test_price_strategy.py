import numpy as np
import pandas as pd
from scripts.price_strategy import cut_economics, within_store_month_gap

EFFECTS = {"markdown_lift": {"by_depth": [
    {"depth": "up to 20%", "mean_depth": 0.135, "lift": 1.565, "lift_weeks_3_4": 1.076, "events": 5000},
]}}
LUXURY = {"revenue_eur_m": 800.0, "sell_through": 0.73}
CFG = {"cut": {"depths": [0.05, 0.10]}}


def test_breakeven_volume_matches_the_closed_form() -> None:
    result = cut_economics(CFG, EFFECTS, LUXURY, leftover_value=0.13)
    ten = next(row for row in result["cuts"] if row["cut"] == 0.10)
    assert abs(ten["breakeven_volume_gain"] - (0.87 / 0.77 - 1)) < 1e-4
    # The sustained response is far below break-even, so a permanent cut loses money.
    assert ten["volume_gain_sustained"] < ten["breakeven_volume_gain"]
    assert ten["profit_change_sustained_eur_m"] < 0


def test_volume_cannot_exceed_the_buy() -> None:
    result = cut_economics({"cut": {"depths": [0.15]}}, EFFECTS, LUXURY, leftover_value=0.13)
    assert result["cuts"][0]["volume_gain_first_two_weeks"] <= 1 / 0.73 - 1 + 1e-4


def test_gap_is_zero_when_entry_clients_behave_the_same() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    frame = pd.DataFrame({
        "store_id": rng.integers(0, 20, n), "first_month": rng.integers(0, 6, n),
        "first_tier": rng.choice(["entry", "core"], n),
    })
    # Return rate depends on the boutique only, so any raw gap is confounding.
    frame["returned_2y"] = rng.random(n) < 0.3 + 0.02 * frame["store_id"]
    gap = within_store_month_gap(frame, "returned_2y", draws=100, seed=1)
    assert gap["ci_low"] < 0 < gap["ci_high"]
