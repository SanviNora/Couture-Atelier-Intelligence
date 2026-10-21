import pandas as pd
from scripts.build_baselines import build_customer_segments, evaluate_seasonal_naive


def test_customer_segments_include_scores_and_clv() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["1", "2", "3"],
            "first_purchase_at": pd.to_datetime(["2011-01-01"] * 3),
            "last_purchase_at": pd.to_datetime(["2011-12-01", "2011-01-02", "2011-11-01"]),
            "order_count": [10, 1, 3],
            "net_revenue": [1000.0, 20.0, 200.0],
        }
    )

    segments = build_customer_segments(customers, as_of_date="2011-12-01")

    assert segments["baseline_clv_12m"].ge(0).all()
    assert segments["recency_score"].between(1, 5).all()
    assert segments["segment"].notna().all()


def test_seasonal_naive_uses_prior_week() -> None:
    dates = pd.date_range("2024-01-01", periods=40, freq="D")
    daily = pd.DataFrame(
        {
            "transaction_date": dates,
                "product_id": ["A"] * 40,
                "country": ["UK"] * 40,
            "gross_units": range(1, 41),
        }
    )

    report = evaluate_seasonal_naive(daily)

    assert report["method"] == "7-day seasonal naive"
    assert report["test_rows"] > 0
    assert report["mae_units"] >= 0
