import pandas as pd
from scripts.build_churn_and_segments import (
    add_churn_labels,
    build_behavior_features,
    build_behavioral_segments,
)


def fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "C"],
            "invoice_id": ["1", "2", "3", "4", "5"],
            "product_id": ["P1", "P2", "P1", "P1", "P3"],
            "transaction_at": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01"]
            ),
            "line_revenue": [100.0, 50.0, 20.0, 20.0, 10.0],
            "is_cancelled": [False] * 5,
            "is_adjustment": [False] * 5,
        }
    )


def test_churn_label_uses_only_future_activity() -> None:
    data = pd.concat(
        [
            fixture(),
            pd.DataFrame(
                {
                    "customer_id": ["A"],
                    "invoice_id": ["6"],
                    "product_id": ["P2"],
                    "transaction_at": pd.to_datetime(["2024-03-01"]),
                    "line_revenue": [50.0],
                    "is_cancelled": [False],
                    "is_adjustment": [False],
                }
            ),
        ],
        ignore_index=True,
    )
    features = build_behavior_features(data, "2024-02-01")
    labeled = add_churn_labels(features, data, horizon_days=90)

    assert labeled.loc[labeled["customer_id"] == "A", "churned_90d"].iloc[0] == 0
    assert labeled.loc[labeled["customer_id"] == "B", "churned_90d"].iloc[0] == 1


def test_behavioral_segments_are_complete_and_deterministic() -> None:
    features = build_behavior_features(fixture(), "2024-02-01")
    first, profiles = build_behavioral_segments(fixture(), "2024-02-01")
    second, _ = build_behavioral_segments(fixture(), "2024-02-01")

    assert len(first) == len(features)
    assert first["behavior_segment"].notna().all()
    assert profiles["customer_count"].sum() == len(first)
    assert first["behavior_segment"].tolist() == second["behavior_segment"].tolist()
