"""Recommend synthetic store transfers and measure stockout impact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

INPUT = Path("data/synthetic/inventory_daily.parquet")
OUTPUT = Path("data/synthetic/transfer_recommendations.parquet")
REPORT = Path("data/synthetic/inventory_optimization_report.json")
LOOKBACK_DAYS = 14
HORIZON_DAYS = 14
SAFETY_UNITS = 2
MIN_DISPLAY_STOCK = 1
TRANSFER_CAPACITY_PER_STORE = 12
WAREHOUSE_CAPACITY = 200


def _inventory_positions(inventory: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    inventory = inventory.copy()
    inventory["date"] = pd.to_datetime(inventory["date"])
    final_date = inventory["date"].max()
    recent = inventory[inventory["date"] > final_date - pd.Timedelta(days=LOOKBACK_DAYS)]
    latest = inventory[inventory["date"] == final_date][
        ["store_id", "product_id", "closing_stock"]
    ]
    demand = recent.groupby(["store_id", "product_id"], as_index=False).agg(
        units_sold=("units_sold", "sum")
    )
    demand["daily_demand_rate"] = demand["units_sold"] / LOOKBACK_DAYS
    positions = latest.merge(demand, on=["store_id", "product_id"], how="left").fillna(
        {"units_sold": 0, "daily_demand_rate": 0}
    )
    positions["target_stock"] = np.ceil(
        positions["daily_demand_rate"] * HORIZON_DAYS + SAFETY_UNITS
    ).astype("int64")
    positions["target_stock"] = positions["target_stock"].clip(lower=MIN_DISPLAY_STOCK)
    positions["stock_surplus"] = (
        positions["closing_stock"] - positions["target_stock"]
    ).clip(lower=0)
    positions["stock_deficit"] = (
        positions["target_stock"] - positions["closing_stock"]
    ).clip(lower=0)
    return positions, final_date


def build_transfer_recommendations(inventory: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "store_id",
        "product_id",
        "opening_stock",
        "units_sold",
        "closing_stock",
    }
    missing = required.difference(inventory.columns)
    if missing:
        raise ValueError(f"Inventory data is missing required columns: {sorted(missing)}")

    positions, final_date = _inventory_positions(inventory)

    recommendations: list[dict[str, object]] = []
    donor_capacity: dict[str, int] = {}
    for product_id, product_positions in positions.groupby("product_id", sort=True):
        donors = product_positions[product_positions["stock_surplus"] > 0].sort_values(
            ["stock_surplus", "store_id"], ascending=[False, True]
        )
        recipients = product_positions[product_positions["stock_deficit"] > 0].sort_values(
            ["stock_deficit", "store_id"], ascending=[False, True]
        )
        donor_surplus = donors.set_index("store_id")["stock_surplus"].to_dict()
        recipient_deficit = recipients.set_index("store_id")["stock_deficit"].to_dict()
        for recipient, deficit in recipient_deficit.items():
            for donor in list(donor_surplus):
                donor_capacity.setdefault(donor, TRANSFER_CAPACITY_PER_STORE)
                if (
                    donor == recipient
                    or deficit <= 0
                    or donor_surplus[donor] <= 0
                    or donor_capacity[donor] <= 0
                ):
                    continue
                units = min(donor_surplus[donor], deficit, donor_capacity[donor])
                recommendations.append(
                    {
                        "recommendation_date": final_date + pd.Timedelta(days=1),
                        "product_id": product_id,
                        "from_store_id": donor,
                        "to_store_id": recipient,
                        "transfer_units": int(units),
                        "reason": "recipient stock below forecast horizon target",
                        "data_provenance": "derived",
                    }
                )
                donor_surplus[donor] -= units
                donor_capacity[donor] -= units
                deficit -= units
    return pd.DataFrame(
        recommendations,
        columns=[
            "recommendation_date",
            "product_id",
            "from_store_id",
            "to_store_id",
            "transfer_units",
            "reason",
            "data_provenance",
        ],
    )


def build_replenishment_recommendations(
    inventory: pd.DataFrame,
    transfers: pd.DataFrame,
) -> pd.DataFrame:
    positions, final_date = _inventory_positions(inventory)
    covered = transfers.groupby(["to_store_id", "product_id"])["transfer_units"].sum()
    rows: list[dict[str, object]] = []
    remaining_capacity = WAREHOUSE_CAPACITY
    for position in positions.itertuples(index=False):
        transferred = covered.get((position.store_id, position.product_id), 0)
        remaining_deficit = max(0, position.stock_deficit - transferred)
        units = min(remaining_deficit, remaining_capacity)
        if units <= 0:
            continue
        rows.append(
            {
                "recommendation_date": final_date + pd.Timedelta(days=1),
                "product_id": position.product_id,
                "to_store_id": position.store_id,
                "replenishment_units": int(units),
                "reason": "residual deficit after constrained transfers",
                "data_provenance": "derived",
            }
        )
        remaining_capacity -= units
        if remaining_capacity == 0:
            break
    return pd.DataFrame(
        rows,
        columns=[
            "recommendation_date",
            "product_id",
            "to_store_id",
            "replenishment_units",
            "reason",
            "data_provenance",
        ],
    )


def simulate_stockout_rate(
    inventory: pd.DataFrame,
    transfers: pd.DataFrame | None = None,
) -> float:
    inventory = inventory.copy()
    inventory["date"] = pd.to_datetime(inventory["date"])
    final_date = inventory["date"].max()
    recent = inventory[inventory["date"] > final_date - pd.Timedelta(days=LOOKBACK_DAYS)]
    demand_rates = recent.groupby(["store_id", "product_id"], as_index=False).agg(
        daily_demand=("units_sold", "mean")
    )
    initial = inventory[inventory["date"] == final_date].set_index(["store_id", "product_id"])[
        "closing_stock"
    ].to_dict()
    if transfers is not None and not transfers.empty:
        for row in transfers.itertuples(index=False):
            donor = (row.from_store_id, row.product_id)
            recipient = (row.to_store_id, row.product_id)
            initial[donor] = initial.get(donor, 0) - row.transfer_units
            initial[recipient] = initial.get(recipient, 0) + row.transfer_units

    stockout_days = 0
    total_days = len(demand_rates) * HORIZON_DAYS
    for row in demand_rates.itertuples(index=False):
        key = (row.store_id, row.product_id)
        stock = float(initial.get(key, 0))
        for _ in range(HORIZON_DAYS):
            stockout_days += int(stock <= 0 and row.daily_demand > 0)
            stock = max(0, stock - row.daily_demand)
    return stockout_days / total_days if total_days else 0.0


def optimize_inventory(
    input_path: Path = INPUT,
    output_path: Path = OUTPUT,
    report_path: Path = REPORT,
) -> dict[str, object]:
    inventory = pd.read_parquet(input_path)
    transfers = build_transfer_recommendations(inventory)
    replenishment = build_replenishment_recommendations(inventory, transfers)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transfers.to_parquet(output_path, index=False)
    replenishment.to_parquet(output_path.with_name("replenishment_recommendations.parquet"), index=False)
    baseline_rate = simulate_stockout_rate(inventory)
    recommended_rate = simulate_stockout_rate(inventory, transfers)
    report = {
        "lookback_days": LOOKBACK_DAYS,
        "horizon_days": HORIZON_DAYS,
        "safety_units": SAFETY_UNITS,
        "minimum_display_stock": MIN_DISPLAY_STOCK,
        "transfer_capacity_per_store": TRANSFER_CAPACITY_PER_STORE,
        "warehouse_capacity": WAREHOUSE_CAPACITY,
        "recommendation_count": len(transfers),
        "recommended_transfer_units": int(transfers["transfer_units"].sum()) if not transfers.empty else 0,
        "baseline_stockout_rate": baseline_rate,
        "recommended_stockout_rate": recommended_rate,
        "stockout_rate_change": recommended_rate - baseline_rate,
        "replenishment_recommendation_count": len(replenishment),
        "recommended_replenishment_units": int(replenishment["replenishment_units"].sum()) if not replenishment.empty else 0,
        "data_provenance": "derived",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(optimize_inventory(), indent=2))
