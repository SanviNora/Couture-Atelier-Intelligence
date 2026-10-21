import pandas as pd
from scripts.build_analytical_marts import (
    build_customer_360,
    build_daily_sales,
    build_product_catalog,
)


def sample_staged() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": pd.Series(["1", "1", "2"], dtype="string"),
            "product_id": pd.Series(["A", "A", "B"], dtype="string"),
            "invoice_id": pd.Series(["100", "C101", "102"], dtype="string"),
            "transaction_at": pd.to_datetime(["2011-01-01", "2011-01-02", "2011-01-01"]),
            "quantity": [2, -1, 1],
            "unit_price": [10.0, 10.0, 5.0],
            "line_revenue": [20.0, -10.0, 5.0],
            "country": ["UK", "UK", "FR"],
            "is_cancelled": [False, True, False],
            "is_adjustment": [False, False, False],
        }
    )


def test_customer_mart_separates_returns_from_sales() -> None:
    customer = build_customer_360(sample_staged())
    row = customer.loc[customer["customer_id"] == "1"].iloc[0]

    assert row["gross_revenue"] == 20.0
    assert row["return_value"] == 10.0
    assert row["net_revenue"] == 10.0
    assert row["order_count"] == 1


def test_customer_purchase_dates_exclude_cancellations() -> None:
    customer = build_customer_360(sample_staged())
    row = customer.loc[customer["customer_id"] == "1"].iloc[0]

    assert row["first_purchase_at"] == pd.Timestamp("2011-01-01")
    assert row["last_purchase_at"] == pd.Timestamp("2011-01-01")


def test_product_and_daily_marts_preserve_net_values() -> None:
    staged = sample_staged()
    product = build_product_catalog(staged)
    daily = build_daily_sales(staged)

    assert product.loc[product["product_id"] == "A", "net_units"].iloc[0] == 1
    assert daily["net_revenue"].sum() == 15.0
    assert set(product["data_provenance"]) == {"derived"}
