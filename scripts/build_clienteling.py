"""Build consent-aware product and next-best-action recommendations."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")
STAGED_INPUT = Path("data/staging/uci_online_retail_ii.parquet")
OUTPUT = PROCESSED_DIR / "clienteling_recommendations.parquet"
REPORT = PROCESSED_DIR / "clienteling_report.json"


def build_product_map(source_products: pd.DataFrame, synthetic_products: pd.DataFrame) -> dict[str, str]:
    selected = source_products.nlargest(len(synthetic_products), "net_revenue").reset_index(drop=True)
    if len(selected) != len(synthetic_products):
        raise ValueError("Synthetic and source product universes do not align")
    return dict(zip(selected["product_id"], synthetic_products["product_id"], strict=True))


def build_recommendations(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    churn: pd.DataFrame,
    clv: pd.DataFrame,
    staged: pd.DataFrame,
    source_products: pd.DataFrame,
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    required_customer = {"customer_id", "consent_status"}
    if not required_customer.issubset(customers.columns):
        raise ValueError("Synthetic customers must include customer_id and consent_status")
    product_map = build_product_map(source_products, products)
    data = staged.copy()
    data["customer_id"] = data["customer_id"].astype("string")
    data["transaction_at"] = pd.to_datetime(data["transaction_at"])
    purchases = data[
        data["customer_id"].notna() & ~data["is_cancelled"] & ~data["is_adjustment"]
    ]
    latest_date = pd.to_datetime(inventory["date"]).max()
    available = inventory[inventory["date"] == latest_date].groupby("product_id")["closing_stock"].sum()
    popular_source = source_products.sort_values("net_revenue", ascending=False)[
        "product_id"
    ].astype("string").tolist()
    clv_index = clv.set_index("customer_id")
    churn_index = churn.set_index("customer_id")
    rows: list[dict[str, object]] = []
    for customer in customers.itertuples(index=False):
        synthetic_id = str(customer.customer_id)
        source_id = synthetic_id.removeprefix("CA-CLIENT-")
        history = purchases[purchases["customer_id"] == source_id]
        purchased_source = set(history["product_id"].astype(str))
        candidate_source = next(
            (
                product_id
                for product_id in popular_source
                if str(product_id) not in purchased_source
                and product_id in product_map
                and available.get(product_map[product_id], 0) > 0
            ),
            None,
        )
        candidate_product = product_map.get(candidate_source) if candidate_source else None
        clv_row = clv_index.loc[source_id] if source_id in clv_index.index else None
        churn_row = churn_index.loc[source_id] if source_id in churn_index.index else None
        predicted_clv = float(clv_row["predicted_clv_90d"]) if clv_row is not None else 0.0
        churn_probability = float(churn_row["churn_probability"]) if churn_row is not None else 0.0
        if customer.consent_status != "consented":
            action = "no_action"
            reason = "personalization consent unavailable"
            eligible = False
        elif candidate_product is None:
            action = "no_action"
            reason = "no eligible in-stock product recommendation"
            eligible = True
        elif churn_probability >= 0.66:
            action = "re_engagement"
            reason = "consented client with elevated inactivity risk and available product"
            eligible = True
        elif predicted_clv >= clv["predicted_clv_90d"].quantile(0.75):
            action = "private_appointment"
            reason = "consented high-value client with available product"
            eligible = True
        else:
            action = "product_suggestion"
            reason = "consented client with available product outside purchase history"
            eligible = True
        rows.append(
            {
                "customer_id": synthetic_id,
                "recommended_product_id": candidate_product,
                "action_type": action,
                "reason": reason,
                "predicted_clv_90d": predicted_clv,
                "churn_probability": churn_probability,
                "consent_status": customer.consent_status,
                "eligible_for_clienteling": eligible,
                "recommendation_date": latest_date + pd.Timedelta(days=1),
                "data_provenance": "derived",
            }
        )
    return pd.DataFrame(rows)


def build_clienteling() -> dict[str, object]:
    customers = pd.read_parquet(SYNTHETIC_DIR / "customers.parquet")
    products = pd.read_parquet(SYNTHETIC_DIR / "products.parquet")
    inventory = pd.read_parquet(SYNTHETIC_DIR / "inventory_daily.parquet")
    churn = pd.read_parquet(PROCESSED_DIR / "churn_predictions.parquet")
    clv = pd.read_parquet(PROCESSED_DIR / "client_clv.parquet")
    staged = pd.read_parquet(STAGED_INPUT)
    source_products = pd.read_parquet(PROCESSED_DIR / "product_catalog.parquet")
    recommendations = build_recommendations(
        customers, products, churn, clv, staged, source_products, inventory
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    recommendations.to_parquet(OUTPUT, index=False)
    report = {
        "customer_count": len(recommendations),
        "consented_count": int(recommendations["consent_status"].eq("consented").sum()),
        "eligible_count": int(recommendations["eligible_for_clienteling"].sum()),
        "action_counts": recommendations["action_type"].value_counts().to_dict(),
        "recommendation_coverage": float(recommendations["recommended_product_id"].notna().mean()),
        "data_provenance": "derived",
    }
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_clienteling(), indent=2))
