"""Train and evaluate a leakage-safe feature-based demand model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

INPUT = Path("data/processed/demand_panel.parquet")
OUTPUT = Path("data/processed/demand_model_report.json")
RANDOM_SEED = 42
MAX_TRAIN_ROWS = 750_000
HOLDOUT_DAYS = 28
GROUP_COLUMNS = ["product_id", "country"]
TARGET = "gross_units"
FEATURES = [
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_7_mean",
    "rolling_28_mean",
    "day_of_week",
    "month",
    "day_of_month",
    "active_age_days",
    "days_since_last_sale",
]


def add_forecast_features(panel: pd.DataFrame) -> pd.DataFrame:
    required = {"transaction_date", *GROUP_COLUMNS, TARGET}
    missing = required.difference(panel.columns)
    if missing:
        raise ValueError(f"Demand panel is missing required columns: {sorted(missing)}")

    features = panel[["transaction_date", *GROUP_COLUMNS, TARGET]].copy()
    features["transaction_date"] = pd.to_datetime(features["transaction_date"])
    if features["transaction_date"].isna().any():
        raise ValueError("Demand panel contains invalid transaction dates")
    if features.duplicated(["transaction_date", *GROUP_COLUMNS]).any():
        raise ValueError("Demand panel contains duplicate product-country-date keys")
    features = features.sort_values([*GROUP_COLUMNS, "transaction_date"]).reset_index(drop=True)
    grouped_target = features.groupby(GROUP_COLUMNS, sort=False)[TARGET]
    for lag in [1, 7, 14, 28]:
        features[f"lag_{lag}"] = grouped_target.shift(lag)

    lagged = grouped_target.shift(1)
    rolling_groups = lagged.groupby(
        [features[column] for column in GROUP_COLUMNS], sort=False
    )
    features["rolling_7_mean"] = rolling_groups.transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    features["rolling_28_mean"] = rolling_groups.transform(
        lambda values: values.rolling(28, min_periods=1).mean()
    )
    features["day_of_week"] = features["transaction_date"].dt.dayofweek
    features["month"] = features["transaction_date"].dt.month
    features["day_of_month"] = features["transaction_date"].dt.day
    first_date = features.groupby(GROUP_COLUMNS, sort=False)["transaction_date"].transform("min")
    features["active_age_days"] = (
        features["transaction_date"] - first_date
    ).dt.days
    sale_dates = features["transaction_date"].where(features[TARGET] > 0)
    last_sale_including_current = sale_dates.groupby(
        [features[column] for column in GROUP_COLUMNS], sort=False
    ).ffill()
    last_sale = last_sale_including_current.groupby(
        [features[column] for column in GROUP_COLUMNS], sort=False
    ).shift(1)
    features["days_since_last_sale"] = (
        features["transaction_date"] - last_sale
    ).dt.days.fillna(features["active_age_days"])
    return features


def prepare_model_data(features: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_data = features.copy()
    model_data[FEATURES] = model_data[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0)
    train = model_data[model_data["transaction_date"] < cutoff].copy()
    test = model_data[model_data["transaction_date"] >= cutoff].copy()
    return train, test


def evaluate_model(train: pd.DataFrame, test: pd.DataFrame) -> dict[str, float | int | str]:
    if train.empty or test.empty:
        raise ValueError("Both train and test windows must contain rows")
    train_sample = train.sample(
        n=min(MAX_TRAIN_ROWS, len(train)), random_state=RANDOM_SEED
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=160,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=RANDOM_SEED,
    )
    model.fit(train_sample[FEATURES], np.log1p(train_sample[TARGET]))
    predictions = np.expm1(model.predict(test[FEATURES])).clip(min=0)
    actual = test[TARGET].to_numpy()
    errors = actual - predictions
    return {
        "method": "hist_gradient_boosting_log1p",
        "train_rows": len(train_sample),
        "test_rows": len(test),
        "mae_units": float(mean_absolute_error(actual, predictions)),
        "rmse_units": float(np.sqrt(mean_squared_error(actual, predictions))),
        "bias_units": float(errors.mean()),
        "feature_count": len(FEATURES),
        "data_provenance": "derived",
    }


def build_demand_model(input_path: Path = INPUT, output_path: Path = OUTPUT) -> dict[str, object]:
    panel = pd.read_parquet(input_path)
    features = add_forecast_features(panel)
    cutoff = features["transaction_date"].max() - pd.Timedelta(days=HOLDOUT_DAYS - 1)
    train, test = prepare_model_data(features, cutoff)
    report = {
        "cutoff": str(cutoff.date()),
        "holdout_end": str(features["transaction_date"].max().date()),
        "features": FEATURES,
        "model": evaluate_model(train, test),
    }
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_demand_model(), indent=2))
