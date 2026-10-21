"""Build an as-of customer value snapshot and evaluate a simple CLV model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/staging/uci_online_retail_ii.parquet")
OUTPUT = Path("data/processed/client_clv.parquet")
REPORT = Path("data/processed/client_clv_report.json")
PREDICTION_HORIZON_DAYS = 90
PRIOR_ORDERS = 1.0
PRIOR_REVENUE = 25.0


def build_customer_snapshot(
    staged: pd.DataFrame,
    as_of_date: pd.Timestamp | str,
    horizon_days: int = PREDICTION_HORIZON_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {
        "customer_id",
        "invoice_id",
        "transaction_at",
        "line_revenue",
        "is_cancelled",
        "is_adjustment",
    }
    missing = required.difference(staged.columns)
    if missing:
        raise ValueError(f"Staged data is missing required columns: {sorted(missing)}")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")

    data = staged.copy()
    data["transaction_at"] = pd.to_datetime(data["transaction_at"])
    cutoff = pd.Timestamp(as_of_date)
    if data["transaction_at"].isna().any():
        raise ValueError("Staged data contains invalid transaction dates")
    customer_rows = data[data["customer_id"].notna()]
    purchases = customer_rows[
        (customer_rows["transaction_at"] <= cutoff)
        & ~customer_rows["is_cancelled"]
        & ~customer_rows["is_adjustment"]
    ].copy()
    returns = customer_rows[
        (customer_rows["transaction_at"] <= cutoff) & customer_rows["is_cancelled"]
    ].copy()
    future = customer_rows[
        (customer_rows["transaction_at"] > cutoff)
        & (customer_rows["transaction_at"] <= cutoff + pd.Timedelta(days=horizon_days))
        & ~customer_rows["is_cancelled"]
        & ~customer_rows["is_adjustment"]
    ].copy()
    if purchases.empty:
        raise ValueError("No eligible purchases exist before as_of_date")

    purchases["customer_id"] = purchases["customer_id"].astype("string")
    returns["customer_id"] = returns["customer_id"].astype("string")
    future["customer_id"] = future["customer_id"].astype("string")
    snapshot = purchases.groupby("customer_id", as_index=False).agg(
        first_purchase_at=("transaction_at", "min"),
        last_purchase_at=("transaction_at", "max"),
        order_count=("invoice_id", "nunique"),
        historical_revenue=("line_revenue", "sum"),
        distinct_products=("product_id", "nunique"),
    )
    snapshot["active_days"] = (
        cutoff - snapshot["first_purchase_at"].dt.normalize()
    ).dt.days.clip(lower=30)
    snapshot["recency_days"] = (
        cutoff - snapshot["last_purchase_at"].dt.normalize()
    ).dt.days
    snapshot["average_order_value"] = snapshot["historical_revenue"] / snapshot["order_count"]
    snapshot["orders_per_day"] = snapshot["order_count"] / snapshot["active_days"]
    snapshot["return_line_count"] = returns.groupby("customer_id").size()
    snapshot["return_line_count"] = snapshot["return_line_count"].fillna(0)
    snapshot["return_rate"] = (
        snapshot["return_line_count"] / snapshot["order_count"].clip(lower=1)
    ).clip(upper=1)
    snapshot["recency_factor"] = np.exp(-snapshot["recency_days"] / 180)
    snapshot["expected_orders_90d"] = (
        (snapshot["orders_per_day"] * horizon_days + PRIOR_ORDERS)
        / (1 + PRIOR_ORDERS)
        * snapshot["recency_factor"]
    )
    snapshot["smoothed_order_value"] = (
        snapshot["historical_revenue"] + PRIOR_REVENUE
    ) / (snapshot["order_count"] + 1)
    snapshot["predicted_clv_90d"] = (
        snapshot["expected_orders_90d"]
        * snapshot["smoothed_order_value"]
        * (1 - snapshot["return_rate"])
    ).clip(lower=0)
    snapshot["as_of_date"] = cutoff
    snapshot["prediction_horizon_days"] = horizon_days
    snapshot["data_provenance"] = "derived"

    realized = future.groupby("customer_id", as_index=False).agg(
        realized_revenue_90d=("line_revenue", "sum"),
        realized_orders_90d=("invoice_id", "nunique"),
    )
    snapshot = snapshot.merge(realized, on="customer_id", how="left")
    snapshot[["realized_revenue_90d", "realized_orders_90d"]] = snapshot[
        ["realized_revenue_90d", "realized_orders_90d"]
    ].fillna(0)
    return snapshot, future


def evaluate_clv(snapshot: pd.DataFrame) -> dict[str, float | int | str]:
    error = snapshot["realized_revenue_90d"] - snapshot["predicted_clv_90d"]
    correlation = snapshot[["predicted_clv_90d", "realized_revenue_90d"]].corr().iloc[0, 1]
    return {
        "customer_count": len(snapshot),
        "mae_revenue": float(error.abs().mean()),
        "bias_revenue": float(error.mean()),
        "prediction_realization_correlation": float(correlation) if pd.notna(correlation) else 0.0,
        "predicted_total_revenue": float(snapshot["predicted_clv_90d"].sum()),
        "realized_total_revenue": float(snapshot["realized_revenue_90d"].sum()),
        "data_provenance": "derived",
    }


def build_client_clv(
    input_path: Path = INPUT,
    output_path: Path = OUTPUT,
    report_path: Path = REPORT,
) -> dict[str, object]:
    staged = pd.read_parquet(input_path)
    as_of_date = pd.to_datetime(staged["transaction_at"]).max() - pd.Timedelta(
        days=PREDICTION_HORIZON_DAYS
    )
    snapshot, _ = build_customer_snapshot(staged, as_of_date)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.to_parquet(output_path, index=False)
    report = {
        "as_of_date": str(as_of_date.date()),
        "horizon_days": PREDICTION_HORIZON_DAYS,
        "evaluation": evaluate_clv(snapshot),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_client_clv(), indent=2))
