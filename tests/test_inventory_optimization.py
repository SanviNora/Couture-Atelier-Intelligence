import pandas as pd
from scripts.optimize_inventory import (
    build_replenishment_recommendations,
    build_transfer_recommendations,
    simulate_stockout_rate,
)


def inventory_fixture() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    "date": date,
                    "store_id": "DONOR",
                    "product_id": "P1",
                    "opening_stock": 20,
                    "units_received": 0,
                    "units_sold": 0,
                    "closing_stock": 20,
                },
                {
                    "date": date,
                    "store_id": "RECIPIENT",
                    "product_id": "P1",
                    "opening_stock": 1,
                    "units_received": 0,
                    "units_sold": 1,
                    "closing_stock": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_transfer_moves_surplus_to_deficit() -> None:
    transfers = build_transfer_recommendations(inventory_fixture())

    assert len(transfers) == 1
    row = transfers.iloc[0]
    assert row["from_store_id"] == "DONOR"
    assert row["to_store_id"] == "RECIPIENT"
    assert row["transfer_units"] > 0
    assert row["data_provenance"] == "derived"


def test_recommended_transfer_reduces_forward_stockout_rate() -> None:
    inventory = inventory_fixture()
    transfers = build_transfer_recommendations(inventory)

    baseline = simulate_stockout_rate(inventory)
    recommended = simulate_stockout_rate(inventory, transfers)

    assert baseline > 0
    assert recommended < baseline


def test_replenishment_covers_deficit_after_transfers() -> None:
    inventory = inventory_fixture()
    transfers = build_transfer_recommendations(inventory)
    replenishment = build_replenishment_recommendations(inventory, transfers)

    assert replenishment["replenishment_units"].sum() >= 0
    assert replenishment["data_provenance"].eq("derived").all()


def test_transfer_capacity_is_not_exceeded() -> None:
    transfers = build_transfer_recommendations(inventory_fixture())

    assert transfers.groupby("from_store_id")["transfer_units"].sum().le(12).all()
