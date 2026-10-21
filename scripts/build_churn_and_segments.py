"""Build time-based churn predictions and interpretable behavioral segments."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

INPUT = Path("data/staging/uci_online_retail_ii.parquet")
CHURN_OUTPUT = Path("data/processed/churn_predictions.parquet")
SEGMENT_OUTPUT = Path("data/processed/behavioral_segments.parquet")
REPORT_OUTPUT = Path("data/processed/client_behavior_report.json")
CHURN_HORIZON_DAYS = 90
CHURN_GAP_DAYS = 90
SEGMENT_COUNT = 5
SEED = 42


def _eligible_rows(staged: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    data = staged.copy()
    data["transaction_at"] = pd.to_datetime(data["transaction_at"])
    rows = data[
        data["customer_id"].notna()
        & (data["transaction_at"] <= cutoff)
        & ~data["is_cancelled"]
        & ~data["is_adjustment"]
    ].copy()
    rows["customer_id"] = rows["customer_id"].astype("string")
    return rows


def build_behavior_features(staged: pd.DataFrame, cutoff: pd.Timestamp | str) -> pd.DataFrame:
    cutoff = pd.Timestamp(cutoff)
    rows = _eligible_rows(staged, cutoff)
    if rows.empty:
        raise ValueError("No eligible customer purchases before cutoff")
    features = rows.groupby("customer_id", as_index=False).agg(
        first_purchase_at=("transaction_at", "min"),
        last_purchase_at=("transaction_at", "max"),
        order_count=("invoice_id", "nunique"),
        line_count=("invoice_id", "size"),
        historical_revenue=("line_revenue", "sum"),
        distinct_products=("product_id", "nunique"),
    )
    features["recency_days"] = (
        cutoff - features["last_purchase_at"].dt.normalize()
    ).dt.days
    features["active_days"] = (
        cutoff - features["first_purchase_at"].dt.normalize()
    ).dt.days.clip(lower=30)
    features["purchase_cadence_days"] = features["active_days"] / features["order_count"]
    features["average_order_value"] = features["historical_revenue"] / features["order_count"]
    features["lines_per_order"] = features["line_count"] / features["order_count"]
    features["product_breadth"] = features["distinct_products"] / features["order_count"]
    features["as_of_date"] = cutoff
    return features


def add_churn_labels(
    features: pd.DataFrame,
    staged: pd.DataFrame,
    horizon_days: int = CHURN_HORIZON_DAYS,
) -> pd.DataFrame:
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive")
    data = staged.copy()
    data["transaction_at"] = pd.to_datetime(data["transaction_at"])
    cutoff = pd.Timestamp(features["as_of_date"].iloc[0])
    future = data[
        data["customer_id"].notna()
        & (data["transaction_at"] > cutoff)
        & (data["transaction_at"] <= cutoff + pd.Timedelta(days=horizon_days))
        & ~data["is_cancelled"]
        & ~data["is_adjustment"]
    ]
    active_ids = future["customer_id"].astype("string").drop_duplicates()
    labels = features[["customer_id"]].copy()
    labels["churned_90d"] = (~labels["customer_id"].isin(active_ids)).astype("int64")
    return features.merge(labels, on="customer_id", validate="one_to_one")


def train_churn_model(staged: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    max_date = pd.to_datetime(staged["transaction_at"]).max().normalize()
    cutoffs = [
        max_date - pd.Timedelta(days=CHURN_HORIZON_DAYS * 3),
        max_date - pd.Timedelta(days=CHURN_HORIZON_DAYS * 2),
    ]
    snapshots = [add_churn_labels(build_behavior_features(staged, cutoff), staged) for cutoff in cutoffs]
    feature_columns = [
        "recency_days",
        "order_count",
        "historical_revenue",
        "distinct_products",
        "purchase_cadence_days",
        "average_order_value",
        "lines_per_order",
        "product_breadth",
    ]
    train = snapshots[0].dropna(subset=feature_columns)
    test = snapshots[1].dropna(subset=feature_columns)
    scaler = StandardScaler()
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=SEED)
    x_train = scaler.fit_transform(train[feature_columns])
    x_test = scaler.transform(test[feature_columns])
    model.fit(x_train, train["churned_90d"])
    test = test.copy()
    test["churn_probability"] = model.predict_proba(x_test)[:, 1]
    test["risk_band"] = pd.cut(
        test["churn_probability"],
        bins=[-np.inf, 0.33, 0.66, np.inf],
        labels=["low", "medium", "high"],
    ).astype("string")
    test["data_provenance"] = "derived"
    report = {
        "train_as_of_date": str(cutoffs[0].date()),
        "test_as_of_date": str(cutoffs[1].date()),
        "feature_columns": feature_columns,
        "train_customers": len(train),
        "test_customers": len(test),
        "churn_rate_test": float(test["churned_90d"].mean()),
        "roc_auc": float(roc_auc_score(test["churned_90d"], test["churn_probability"])),
        "average_precision": float(average_precision_score(test["churned_90d"], test["churn_probability"])),
        "brier_score": float(brier_score_loss(test["churned_90d"], test["churn_probability"])),
        "data_provenance": "derived",
    }
    return test, report


def build_behavioral_segments(staged: pd.DataFrame, cutoff: pd.Timestamp | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    features = build_behavior_features(staged, cutoff)
    segment_columns = [
        "recency_days",
        "order_count",
        "historical_revenue",
        "distinct_products",
        "purchase_cadence_days",
        "average_order_value",
        "lines_per_order",
        "product_breadth",
    ]
    values = features[segment_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
    values["historical_revenue"] = np.log1p(values["historical_revenue"])
    values["average_order_value"] = np.log1p(values["average_order_value"].clip(lower=0))
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    model = KMeans(n_clusters=min(SEGMENT_COUNT, len(features)), random_state=SEED, n_init=20)
    features["behavior_segment"] = model.fit_predict(scaled).astype("int64")
    profiles = features.groupby("behavior_segment", as_index=False)[segment_columns].mean()
    profiles["customer_count"] = features["behavior_segment"].value_counts().sort_index().values
    profiles["data_provenance"] = "derived"
    features["data_provenance"] = "derived"
    return features, profiles


def build_client_behavior(input_path: Path = INPUT) -> dict[str, object]:
    staged = pd.read_parquet(input_path)
    churn, churn_report = train_churn_model(staged)
    cutoff = pd.to_datetime(staged["transaction_at"]).max().normalize() - pd.Timedelta(days=CHURN_HORIZON_DAYS)
    segments, profiles = build_behavioral_segments(staged, cutoff)
    CHURN_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    churn.to_parquet(CHURN_OUTPUT, index=False)
    segments.to_parquet(SEGMENT_OUTPUT, index=False)
    profiles.to_parquet(SEGMENT_OUTPUT.with_name("behavioral_segment_profiles.parquet"), index=False)
    report = {"churn": churn_report, "behavioral_segments": {"customer_count": len(segments), "segment_count": len(profiles), "data_provenance": "derived"}}
    REPORT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_client_behavior(), indent=2))
