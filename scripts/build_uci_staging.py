"""Build a normalized staging table from UCI Online Retail II."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/uci_online_retail_ii/online_retail_II.xlsx")
DEFAULT_OUTPUT = Path("data/staging/uci_online_retail_ii.parquet")
DEFAULT_SUMMARY = Path("data/staging/uci_online_retail_ii_summary.json")

COLUMN_MAP = {
    "Invoice": "invoice_id",
    "StockCode": "product_id",
    "Description": "product_description",
    "Quantity": "quantity",
    "InvoiceDate": "transaction_at",
    "Price": "unit_price",
    "Customer ID": "customer_id",
    "Country": "country",
}


def load_workbook(input_path: Path) -> pd.DataFrame:
    workbook = pd.ExcelFile(input_path)
    frames = [pd.read_excel(input_path, sheet_name=sheet) for sheet in workbook.sheet_names]
    return pd.concat(frames, ignore_index=True)


def validate_source_frame(frame: pd.DataFrame) -> None:
    required_columns = set(COLUMN_MAP)
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required source columns: {sorted(missing_columns)}")


def transform_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    validate_source_frame(frame)
    frame = frame.drop_duplicates().copy()
    transformed = frame.rename(columns=COLUMN_MAP).copy()
    transformed["invoice_id"] = transformed["invoice_id"].astype("string").str.strip()
    transformed["product_id"] = transformed["product_id"].astype("string").str.strip()
    transformed["product_description"] = (
        transformed["product_description"].astype("string").str.strip()
    )
    transformed["customer_id"] = transformed["customer_id"].astype("Int64").astype("string")
    transformed["country"] = transformed["country"].astype("string").str.strip()
    transformed["transaction_at"] = pd.to_datetime(transformed["transaction_at"], errors="coerce")
    transformed["quantity"] = pd.to_numeric(transformed["quantity"], errors="coerce").astype("Int64")
    transformed["unit_price"] = pd.to_numeric(transformed["unit_price"], errors="coerce")

    transformed["is_cancelled"] = transformed["invoice_id"].str.startswith("C", na=False)
    transformed["is_adjustment"] = (
        transformed["product_id"].eq("B")
        | transformed["invoice_id"].str.startswith("A", na=False)
        | (
            ~transformed["is_cancelled"]
            & transformed["quantity"].lt(0)
            & transformed["unit_price"].eq(0)
        )
    )
    invalid_rows = (
        transformed["invoice_id"].isna()
        | transformed["product_id"].isna()
        | transformed["transaction_at"].isna()
        | transformed["quantity"].isna()
        | transformed["unit_price"].isna()
        | (transformed["unit_price"].lt(0) & ~transformed["is_adjustment"])
        | (
            ~transformed["is_cancelled"]
            & ~transformed["is_adjustment"]
            & transformed["quantity"].lt(0)
        )
    )
    if invalid_rows.any():
        raise ValueError(f"Source validation failed for {int(invalid_rows.sum())} rows")
    transformed["line_revenue"] = transformed["quantity"] * transformed["unit_price"]
    transformed["has_customer_id"] = transformed["customer_id"].notna()
    transformed["data_provenance"] = "observed_public"

    columns = [
        "invoice_id",
        "product_id",
        "product_description",
        "quantity",
        "transaction_at",
        "unit_price",
        "line_revenue",
        "customer_id",
        "country",
        "is_cancelled",
        "is_adjustment",
        "has_customer_id",
        "data_provenance",
    ]
    return transformed[columns]


def build_summary(raw: pd.DataFrame, staged: pd.DataFrame) -> dict[str, object]:
    return {
        "source_id": "uci_online_retail_ii",
        "data_provenance": "observed_public",
        "raw_rows": len(raw),
        "staged_rows": len(staged),
        "duplicate_rows_detected": int(raw.duplicated().sum()),
        "duplicate_rows_removed": int(raw.duplicated().sum()),
        "cancelled_rows": int(staged["is_cancelled"].sum()),
        "adjustment_rows": int(staged["is_adjustment"].sum()),
        "rows_without_customer_id": int((~staged["has_customer_id"]).sum()),
        "invalid_transaction_dates": int(staged["transaction_at"].isna().sum()),
        "invalid_quantities": int(staged["quantity"].isna().sum()),
        "invalid_prices": int(staged["unit_price"].isna().sum()),
        "date_min": str(staged["transaction_at"].min()),
        "date_max": str(staged["transaction_at"].max()),
    }


def build_staging(
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    summary_path: Path = DEFAULT_SUMMARY,
) -> dict[str, object]:
    raw = load_workbook(input_path)
    staged = transform_transactions(raw)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staged.to_parquet(output_path, index=False)
    summary = build_summary(raw, staged)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    print(json.dumps(build_staging(), indent=2))


if __name__ == "__main__":
    main()
