"""Build a complete product-country-date demand panel for forecasting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INPUT = Path("data/processed/daily_sales.parquet")
OUTPUT = Path("data/processed/demand_panel.parquet")
KEY_COLUMNS = ["product_id", "country"]
VALUE_COLUMNS = [
    "gross_units",
    "returned_units",
    "gross_revenue",
    "return_value",
    "net_units",
    "net_revenue",
    "transaction_lines",
]


def build_demand_panel(daily_sales: pd.DataFrame) -> pd.DataFrame:
    daily = daily_sales.copy()
    daily["transaction_date"] = pd.to_datetime(daily["transaction_date"])
    active_windows = daily.groupby(KEY_COLUMNS, as_index=False).agg(
        active_start=("transaction_date", "min"),
        active_end=("transaction_date", "max"),
    )

    calendar_parts: list[pd.DataFrame] = []
    for row in active_windows.itertuples(index=False):
        calendar_parts.append(
            pd.DataFrame(
                {
                    "transaction_date": pd.date_range(row.active_start, row.active_end),
                    "product_id": row.product_id,
                    "country": row.country,
                }
            )
        )
    calendar = pd.concat(calendar_parts, ignore_index=True)
    panel = calendar.merge(daily, on=["transaction_date", *KEY_COLUMNS], how="left")
    panel["is_observed"] = panel["gross_units"].notna()
    for column in VALUE_COLUMNS:
        panel[column] = panel[column].fillna(0)
    panel["data_provenance"] = "derived"
    return panel.sort_values([*KEY_COLUMNS, "transaction_date"]).reset_index(drop=True)


def build_from_parquet(input_path: Path = INPUT, output_path: Path = OUTPUT) -> dict[str, int]:
    daily = pd.read_parquet(input_path)
    panel = build_demand_panel(daily)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output_path, index=False)
    return {
        "observed_rows": int(daily.shape[0]),
        "panel_rows": int(panel.shape[0]),
        "filled_zero_rows": int((~panel["is_observed"]).sum()),
        "product_country_keys": int(panel[KEY_COLUMNS].drop_duplicates().shape[0]),
    }


if __name__ == "__main__":
    print(build_from_parquet())
