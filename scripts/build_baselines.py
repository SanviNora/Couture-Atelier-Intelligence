"""Build customer RFM/CLV baselines and evaluate a seasonal demand baseline."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
CUSTOMER_INPUT = PROCESSED_DIR / "customer_360.parquet"
DEMAND_INPUT = PROCESSED_DIR / "demand_panel.parquet"
SEGMENT_OUTPUT = PROCESSED_DIR / "customer_segments.parquet"
REPORT_OUTPUT = PROCESSED_DIR / "baseline_report.json"


def score_percentile(values: pd.Series, higher_is_better: bool = True) -> pd.Series:
    percentile = values.rank(method="average", pct=True)
    if not higher_is_better:
        percentile = 1 - percentile + (1 / max(len(values), 1))
    return (percentile.mul(5).clip(upper=5).round().astype("int64")).clip(lower=1)


def build_customer_segments(
    customers: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
) -> pd.DataFrame:
    segments = customers.copy()
    segments["last_purchase_at"] = pd.to_datetime(segments["last_purchase_at"])
    reference_date = pd.Timestamp(as_of_date).normalize()
    segments = segments[segments["last_purchase_at"] <= reference_date].copy()
    segments["recency_days"] = (
        reference_date - segments["last_purchase_at"].dt.normalize()
    ).dt.days
    segments["frequency"] = segments["order_count"]
    segments["monetary_value"] = segments["net_revenue"].clip(lower=0)
    lifetime_days = (
        segments["last_purchase_at"] - pd.to_datetime(segments["first_purchase_at"])
    ).dt.days.clip(lower=30)
    segments["average_order_value"] = segments["monetary_value"] / segments["frequency"].clip(lower=1)
    segments["annualized_value"] = segments["monetary_value"] / lifetime_days * 365
    segments["retention_factor"] = np.exp(-segments["recency_days"] / 180)
    segments["baseline_clv_12m"] = segments["annualized_value"] * segments["retention_factor"]

    segments["recency_score"] = score_percentile(segments["recency_days"], higher_is_better=False)
    segments["frequency_score"] = score_percentile(segments["frequency"])
    segments["monetary_score"] = score_percentile(segments["monetary_value"])
    segments["segment"] = "Other"
    segments.loc[
        (segments["recency_score"] >= 4)
        & (segments["frequency_score"] >= 4)
        & (segments["monetary_score"] >= 4),
        "segment",
    ] = "Champions"
    segments.loc[
        (segments["segment"] == "Other")
        & (segments["frequency_score"] >= 4)
        & (segments["monetary_score"] >= 3),
        "segment",
    ] = "Loyal"
    segments.loc[
        (segments["segment"] == "Other") & (segments["monetary_score"] >= 4),
        "segment",
    ] = "High Value"
    segments.loc[
        (segments["segment"] == "Other")
        & (segments["recency_score"] <= 2)
        & (segments["frequency_score"] >= 3),
        "segment",
    ] = "At Risk"
    segments.loc[
        (segments["segment"] == "Other")
        & (segments["recency_score"] >= 4)
        & (segments["frequency_score"] <= 2),
        "segment",
    ] = "Emerging"
    segments["data_provenance"] = "derived"
    return segments


def evaluate_seasonal_naive(daily_sales: pd.DataFrame) -> dict[str, object]:
    daily = daily_sales.copy()
    daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])
    max_date = daily["transaction_date"].max()
    test_start = max_date - pd.Timedelta(days=27)
    test = daily[daily["transaction_date"] >= test_start].copy()
    test["forecast_date"] = test["transaction_date"]
    history = daily[daily["transaction_date"] < test_start].copy()
    history["forecast_date"] = history["transaction_date"] + pd.Timedelta(days=7)
    baseline = history[["forecast_date", "product_id", "country", "gross_units"]].rename(
        columns={"gross_units": "forecast_units"}
    )
    predictions = test.merge(
        baseline, on=["forecast_date", "product_id", "country"], how="left"
    )
    predictions["lag_available"] = predictions["forecast_units"].notna()
    predictions["forecast_units"] = predictions["forecast_units"].fillna(0)
    errors = predictions["gross_units"] - predictions["forecast_units"]
    return {
        "method": "7-day seasonal naive",
        "test_start": str(test_start.date()),
        "test_end": str(max_date.date()),
        "test_rows": len(predictions),
        "mae_units": float(errors.abs().mean()),
        "rmse_units": float(np.sqrt(np.mean(np.square(errors)))),
        "bias_units": float(errors.mean()),
        "lag_available_rate": float(predictions["lag_available"].mean()),
        "zero_forecast_rate": float(predictions["forecast_units"].eq(0).mean()),
        "data_provenance": "derived",
    }


def build_baselines() -> dict[str, object]:
    customers = pd.read_parquet(CUSTOMER_INPUT)
    demand_panel = pd.read_parquet(DEMAND_INPUT)
    as_of_date = pd.to_datetime(customers["last_purchase_at"]).max()
    segments = build_customer_segments(customers, as_of_date=as_of_date)
    segments.to_parquet(SEGMENT_OUTPUT, index=False)
    report = {
        "customer_count": len(segments),
        "segment_counts": segments["segment"].value_counts().to_dict(),
        "forecast": evaluate_seasonal_naive(demand_panel),
    }
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_baselines(), indent=2))
