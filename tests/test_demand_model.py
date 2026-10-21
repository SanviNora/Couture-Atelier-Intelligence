import numpy as np
import pandas as pd
import pytest
from scripts.build_demand_model import (
    FEATURES,
    add_forecast_features,
    evaluate_model,
    prepare_model_data,
)


def test_features_are_lagged_and_chronological() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = pd.DataFrame(
        {
            "transaction_date": dates,
            "product_id": ["A"] * len(dates),
            "country": ["UK"] * len(dates),
            "gross_units": range(1, len(dates) + 1),
        }
    )

    features = add_forecast_features(panel)

    assert features.loc[7, "lag_7"] == 1
    assert features.loc[7, "rolling_7_mean"] == 4
    assert features.loc[7, "days_since_last_sale"] == 1
    assert features["transaction_date"].is_monotonic_increasing
    assert set(FEATURES).issubset(features.columns)


def test_prepare_model_data_respects_cutoff() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = pd.DataFrame(
        {
            "transaction_date": dates,
            "product_id": ["A"] * len(dates),
            "country": ["UK"] * len(dates),
            "gross_units": range(1, len(dates) + 1),
        }
    )
    features = add_forecast_features(panel)
    train, test = prepare_model_data(features, pd.Timestamp("2024-01-21"))

    assert train["transaction_date"].max() < pd.Timestamp("2024-01-21")
    assert test["transaction_date"].min() >= pd.Timestamp("2024-01-21")


def test_lags_do_not_cross_product_country_groups() -> None:
    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    panel = pd.concat(
        [
            pd.DataFrame(
                {
                    "transaction_date": dates,
                    "product_id": "A",
                    "country": "UK",
                    "gross_units": 10,
                }
            ),
            pd.DataFrame(
                {
                    "transaction_date": dates,
                    "product_id": "B",
                    "country": "FR",
                    "gross_units": 100,
                }
            ),
        ],
        ignore_index=True,
    )

    features = add_forecast_features(panel)
    row = features.loc[
        (features["product_id"] == "B") & (features["transaction_date"] == dates[1])
    ].iloc[0]

    assert row["lag_1"] == 100


def test_zero_demand_days_keep_prior_sale_recency() -> None:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    panel = pd.DataFrame(
        {
            "transaction_date": dates,
            "product_id": "A",
            "country": "UK",
            "gross_units": [3, 0, 0, 2],
        }
    )

    features = add_forecast_features(panel)

    assert features.loc[1, "days_since_last_sale"] == 1
    assert features.loc[2, "days_since_last_sale"] == 2
    assert features.loc[3, "days_since_last_sale"] == 3


def test_duplicate_keys_and_missing_columns_fail_fast() -> None:
    panel = pd.DataFrame(
        {
            "transaction_date": ["2024-01-01", "2024-01-01"],
            "product_id": ["A", "A"],
            "country": ["UK", "UK"],
            "gross_units": [1, 2],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        add_forecast_features(panel)
    with pytest.raises(ValueError, match="missing required"):
        add_forecast_features(panel.drop(columns="country"))


def test_model_predictions_are_deterministic_and_nonnegative() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    panel = pd.DataFrame(
        {
            "transaction_date": dates,
            "product_id": "A",
            "country": "UK",
            "gross_units": np.where(np.arange(40) % 3 == 0, 5, 0),
        }
    )
    features = add_forecast_features(panel)
    train, test = prepare_model_data(features, pd.Timestamp("2024-01-21"))

    first = evaluate_model(train, test)
    second = evaluate_model(train, test)

    assert first == second
    assert first["mae_units"] >= 0
    assert first["rmse_units"] >= 0
