import pandas as pd
from scripts.build_clienteling import build_recommendations


def test_clienteling_respects_consent_and_stock() -> None:
    customers = pd.DataFrame(
        {
            "customer_id": ["CA-CLIENT-1", "CA-CLIENT-2"],
            "consent_status": ["consented", "not_consented"],
        }
    )
    products = pd.DataFrame({"product_id": ["CA-PROD-1"]})
    source_products = pd.DataFrame({"product_id": ["P1"], "net_revenue": [100.0]})
    churn = pd.DataFrame({"customer_id": ["1", "2"], "churn_probability": [0.8, 0.2]})
    clv = pd.DataFrame({"customer_id": ["1", "2"], "predicted_clv_90d": [100.0, 10.0]})
    staged = pd.DataFrame(
        {
            "customer_id": ["1"],
            "product_id": ["P2"],
            "line_revenue": [20.0],
            "transaction_at": pd.to_datetime(["2024-01-01"]),
            "is_cancelled": [False],
            "is_adjustment": [False],
        }
    )
    inventory = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01"]),
            "product_id": ["CA-PROD-1"],
            "closing_stock": [3],
        }
    )

    recommendations = build_recommendations(
        customers, products, churn, clv, staged, source_products, inventory
    )

    consented = recommendations.loc[recommendations["customer_id"] == "CA-CLIENT-1"].iloc[0]
    not_consented = recommendations.loc[recommendations["customer_id"] == "CA-CLIENT-2"].iloc[0]
    assert consented["action_type"] == "re_engagement"
    assert consented["recommended_product_id"] == "CA-PROD-1"
    assert not_consented["action_type"] == "no_action"
    assert not not_consented["eligible_for_clienteling"]
