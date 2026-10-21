import pandas as pd
from scripts.build_synthetic_luxury_layer import (
    build_customers,
    build_inventory,
    build_products,
    build_stores,
)


def source_customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["1", "2"],
            "first_purchase_at": pd.to_datetime(["2011-01-01", "2011-02-01"]),
            "primary_country": ["France", "Japan"],
        }
    )


def source_products() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "product_id": ["A", "B"],
            "net_revenue": [1000.0, 500.0],
        }
    )


def test_synthetic_customers_are_anonymous_and_consent_is_explicit() -> None:
    customers = build_customers(source_customers())

    assert customers["customer_id"].str.startswith("CA-CLIENT-").all()
    assert set(customers["home_region"]) == {"Western Europe", "East Asia"}
    assert set(customers["consent_status"]) <= {"consented", "not_consented"}
    assert customers["data_provenance"].eq("synthetic").all()


def test_synthetic_products_match_documented_contract() -> None:
    products = build_products(source_products())

    assert products["product_id"].is_unique
    assert products["list_price"].gt(0).all()
    assert products["launch_date"].notna().all()
    assert products["data_provenance"].eq("synthetic").all()


def test_inventory_reconciles_and_is_deterministic() -> None:
    products = build_products(source_products())
    stores = build_stores().head(1)
    demand = pd.DataFrame({"transaction_date": pd.to_datetime(["2024-03-01"])})

    first = build_inventory(products, stores, demand)
    second = build_inventory(products, stores, demand)

    assert first.equals(second)
    assert first["closing_stock"].ge(0).all()
    assert (
        first["opening_stock"]
        + first["units_received"]
        + first["transfer_in"]
        - first["units_sold"]
        - first["transfer_out"]
        == first["closing_stock"]
    ).all()
    assert first["data_provenance"].eq("synthetic").all()
