import pandas as pd
from scripts.build_uci_staging import transform_transactions


def test_transform_preserves_cancellations_and_missing_customers() -> None:
    raw = pd.DataFrame(
        {
            "Invoice": ["10001", "C10002"],
            "StockCode": ["P001", "P002"],
            "Description": [" Leather bag ", None],
            "Quantity": [2, -1],
            "InvoiceDate": ["2011-01-01 10:00:00", "2011-01-02 11:00:00"],
            "Price": [10.0, 25.0],
            "Customer ID": [12345, None],
            "Country": ["United Kingdom", "France"],
        }
    )

    staged = transform_transactions(raw)

    assert staged.loc[0, "product_description"] == "Leather bag"
    assert staged.loc[1, "is_cancelled"]
    assert not staged.loc[1, "has_customer_id"]
    assert staged.loc[0, "line_revenue"] == 20.0
    assert staged["data_provenance"].eq("observed_public").all()


def test_transform_removes_exact_duplicate_source_rows() -> None:
    raw = pd.DataFrame(
        {
            "Invoice": ["10001", "10001"],
            "StockCode": ["P001", "P001"],
            "Description": ["Bag", "Bag"],
            "Quantity": [1, 1],
            "InvoiceDate": ["2011-01-01", "2011-01-01"],
            "Price": [10.0, 10.0],
            "Customer ID": [12345, 12345],
            "Country": ["United Kingdom", "United Kingdom"],
        }
    )

    assert len(transform_transactions(raw)) == 1


def test_transform_classifies_zero_price_stock_adjustments() -> None:
    raw = pd.DataFrame(
        {
            "Invoice": ["489464"],
            "StockCode": ["21733"],
            "Description": ["damages"],
            "Quantity": [-96],
            "InvoiceDate": ["2011-01-01"],
            "Price": [0.0],
            "Customer ID": [None],
            "Country": ["United Kingdom"],
        }
    )

    staged = transform_transactions(raw)

    assert staged.loc[0, "is_adjustment"]
