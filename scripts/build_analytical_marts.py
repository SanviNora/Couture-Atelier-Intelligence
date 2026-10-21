"""Build customer, product, and daily-sales marts from staged transactions."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/staging/uci_online_retail_ii.parquet")
DEFAULT_OUTPUT = Path("data/processed")


BASE_COLUMNS = ["customer_id", "product_id", "invoice_id", "transaction_at", "quantity", "unit_price", "line_revenue", "country", "is_cancelled", "is_adjustment"]


def load_staged(input_path: Path) -> pd.DataFrame:
    staged = pd.read_parquet(input_path, columns=BASE_COLUMNS)
    return add_derived_metrics(staged)


def add_derived_metrics(staged: pd.DataFrame) -> pd.DataFrame:
    staged = staged.copy()
    staged["transaction_date"] = staged["transaction_at"].dt.normalize()
    sale_rows = ~staged["is_cancelled"] & ~staged["is_adjustment"]
    return_rows = staged["is_cancelled"]
    staged["sales_quantity"] = staged["quantity"].where(sale_rows, 0).astype("int64")
    staged["returned_quantity"] = (-staged["quantity"].where(return_rows, 0)).astype("int64")
    staged["sales_revenue"] = staged["line_revenue"].where(sale_rows, 0.0)
    staged["return_revenue"] = (-staged["line_revenue"].where(return_rows, 0.0)).astype("float64")
    return staged


def build_customer_360(
    staged: pd.DataFrame,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    staged = add_derived_metrics(staged)
    if as_of_date is not None:
        cutoff = pd.Timestamp(as_of_date)
        staged = staged[staged["transaction_at"] <= cutoff].copy()
    customer_rows = staged[staged["customer_id"].notna()].copy()
    customer_rows["customer_id"] = customer_rows["customer_id"].astype("string")

    purchase_rows = customer_rows[~customer_rows["is_cancelled"] & ~customer_rows["is_adjustment"]].copy()
    customers = purchase_rows.groupby("customer_id", as_index=False).agg(
        first_purchase_at=("transaction_at", "min"),
        last_purchase_at=("transaction_at", "max"),
        order_count=("invoice_id", "nunique"),
        line_count=("invoice_id", "size"),
        gross_units=("sales_quantity", "sum"),
        returned_units=("returned_quantity", "sum"),
        gross_revenue=("sales_revenue", "sum"),
        return_value=("return_revenue", "sum"),
        distinct_products=("product_id", "nunique"),
        primary_country=("country", "first"),
    )
    returns = customer_rows[customer_rows["is_cancelled"]].groupby("customer_id", as_index=False).agg(
        returned_units=("returned_quantity", "sum"),
        return_value=("return_revenue", "sum"),
    )
    customers = customers.merge(returns, on="customer_id", how="outer", suffixes=("", "_returns"))
    for column in ["gross_units", "gross_revenue", "returned_units", "return_value"]:
        return_column = f"{column}_returns"
        if return_column in customers:
            customers[column] = customers[column].fillna(0) + customers[return_column].fillna(0)
            customers = customers.drop(columns=return_column)
        else:
            customers[column] = customers[column].fillna(0)
    customers["net_units"] = customers["gross_units"] - customers["returned_units"]
    customers["net_revenue"] = customers["gross_revenue"] - customers["return_value"]
    customers["data_provenance"] = "derived"
    return customers


def build_product_catalog(staged: pd.DataFrame) -> pd.DataFrame:
    staged = add_derived_metrics(staged)
    products = staged.groupby("product_id", as_index=False).agg(
        first_seen_at=("transaction_at", "min"),
        last_seen_at=("transaction_at", "max"),
        transaction_lines=("product_id", "size"),
        unique_customers=("customer_id", "nunique"),
        gross_units=("sales_quantity", "sum"),
        returned_units=("returned_quantity", "sum"),
        gross_revenue=("sales_revenue", "sum"),
        return_value=("return_revenue", "sum"),
    )
    products["net_units"] = products["gross_units"] - products["returned_units"]
    products["net_revenue"] = products["gross_revenue"] - products["return_value"]
    products["data_provenance"] = "derived"
    return products


def build_daily_sales(staged: pd.DataFrame) -> pd.DataFrame:
    staged = add_derived_metrics(staged)
    daily = staged.groupby(["transaction_date", "product_id", "country"], as_index=False).agg(
        gross_units=("sales_quantity", "sum"),
        returned_units=("returned_quantity", "sum"),
        gross_revenue=("sales_revenue", "sum"),
        return_value=("return_revenue", "sum"),
        transaction_lines=("product_id", "size"),
    )
    daily["net_units"] = daily["gross_units"] - daily["returned_units"]
    daily["net_revenue"] = daily["gross_revenue"] - daily["return_value"]
    daily["data_provenance"] = "derived"
    return daily


def build_marts(input_path: Path = DEFAULT_INPUT, output_dir: Path = DEFAULT_OUTPUT) -> dict[str, int]:
    staged = load_staged(input_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    marts = {
        "customer_360": build_customer_360(staged, as_of_date=staged["transaction_at"].max()),
        "product_catalog": build_product_catalog(staged),
        "daily_sales": build_daily_sales(staged),
    }
    counts: dict[str, int] = {}
    for name, frame in marts.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        counts[name] = len(frame)
    return counts


if __name__ == "__main__":
    print(build_marts())
