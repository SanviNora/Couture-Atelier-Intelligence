import pandas as pd
from scripts.build_demand_panel import build_demand_panel


def test_panel_fills_missing_active_dates_with_zero_demand() -> None:
    daily = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(["2024-01-01", "2024-01-03"]),
            "product_id": ["A", "A"],
            "country": ["UK", "UK"],
            "gross_units": [2, 4],
            "returned_units": [0, 0],
            "gross_revenue": [20.0, 40.0],
            "return_value": [0.0, 0.0],
            "net_units": [2, 4],
            "net_revenue": [20.0, 40.0],
            "transaction_lines": [1, 1],
        }
    )

    panel = build_demand_panel(daily)
    middle = panel.loc[panel["transaction_date"] == "2024-01-02"].iloc[0]

    assert len(panel) == 3
    assert middle["gross_units"] == 0
    assert not middle["is_observed"]
    assert middle["data_provenance"] == "derived"
