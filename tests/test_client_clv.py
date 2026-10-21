import pandas as pd
import pytest
from scripts.build_client_clv import build_customer_snapshot, evaluate_clv


def staged_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["A", "A", "B", "B", "A"],
            "invoice_id": ["1", "2", "3", "4", "5"],
            "product_id": ["P1", "P1", "P2", "P2", "P1"],
            "transaction_at": pd.to_datetime(
                ["2024-01-01", "2024-02-01", "2024-01-01", "2024-03-01", "2024-05-01"]
            ),
            "line_revenue": [100.0, 100.0, 20.0, 20.0, 100.0],
            "is_cancelled": [False, False, False, False, False],
            "is_adjustment": [False] * 5,
        }
    )


def test_clv_snapshot_excludes_future_revenue_from_features() -> None:
    snapshot, future = build_customer_snapshot(staged_fixture(), "2024-03-01")
    customer_a = snapshot.loc[snapshot["customer_id"] == "A"].iloc[0]

    assert len(future) == 1
    assert customer_a["order_count"] == 2
    assert customer_a["historical_revenue"] == 200.0
    assert customer_a["realized_revenue_90d"] == 100.0
    assert customer_a["predicted_clv_90d"] >= 0


def test_clv_evaluation_is_finite_and_counts_customers() -> None:
    snapshot, _ = build_customer_snapshot(staged_fixture(), "2024-03-01")

    report = evaluate_clv(snapshot)

    assert report["customer_count"] == 2
    assert pd.notna(report["prediction_realization_correlation"])
    assert report["mae_revenue"] >= 0


def test_clv_requires_valid_as_of_and_horizon() -> None:
    with pytest.raises(ValueError, match="horizon_days"):
        build_customer_snapshot(staged_fixture(), "2024-03-01", horizon_days=0)
