import duckdb
import pandas as pd
import pytest
from scripts.build_marts import build_marts
from scripts.ingest_visuelle import validate

from couture_atelier.paths import SQL

WEEKS = range(12)


def make_raw(con: duckdb.DuckDBPyConnection, net_units: list[float]) -> None:
    sales = pd.DataFrame(
        {
            "product_id": [1, 1],
            "store_id": [10, 20],
            "season": "SS19",
            "category": "long dress",
            "color": "black",
            "fabric": "crepe",
            "image_path": "x.png",
            "release_date": pd.to_datetime(["2019-01-07", "2019-01-07"]),
            "restock_events": [1, 0],
            **{f"w{w}": [net_units[w], 1.0] for w in WEEKS},
        }
    )
    discounts = pd.DataFrame(
        {
            "product_id": [1, 1],
            "store_id": [10, 20],
            "price_index": [0.08, 0.08],
            **{f"d{w}": [0.3 if w >= 8 else 0.0, 0.0] for w in WEEKS},
        }
    )
    restocks = pd.DataFrame({"product_id": [1], "store_id": [10], "year": [2019], "week": [3], "qty": [4]})
    purchases = pd.DataFrame(
        {
            "client_id": [7, 7, 7, 7, 8],
            "store_id": [10, 10, 10, 20, 20],
            "product_id": [1, 1, 1, 1, 1],
            "purchased_at": pd.to_datetime(
                ["2019-01-08 10:00", "2019-01-08 10:00", "2019-01-20 15:00", "2019-02-01 12:00", "2019-01-09 09:00"]
            ),
            "qty": [1, 1, 1, 1, 1],
        }
    )
    con.execute("create schema raw")
    for name, frame in {"sales": sales, "price_discount": discounts, "restocks": restocks, "purchases": purchases}.items():
        con.register("frame", frame)
        con.execute(f"create table raw.{name} as select * from frame")
        con.unregister("frame")


def test_validation_flags_implausible_negative_sales(tmp_path) -> None:
    with duckdb.connect(str(tmp_path / "w.duckdb")) as con:
        make_raw(con, [2.0, 1.0, 0.0, -9.0, *[1.0] * 8])
        with pytest.raises(ValueError, match="net of returns"):
            validate(con)


def test_marts_split_returns_match_restocks_and_group_receipts(tmp_path) -> None:
    warehouse = tmp_path / "w.duckdb"
    with duckdb.connect(str(warehouse)) as con:
        make_raw(con, [2.0, 1.0, 0.0, 1.0, -1.0, *[1.0] * 7])

    counts = build_marts(warehouse, SQL / "marts")

    with duckdb.connect(str(warehouse)) as con:
        weeks = con.execute(
            "select * from marts.fct_store_product_week where store_id = 10 order by week_index"
        ).fetchdf()
        receipts = con.execute(
            "select count(distinct receipt_id) from marts.fct_purchase where client_id = 7"
        ).fetchone()[0]
        home = con.execute("select home_store_id from marts.client_profile where client_id = 7").fetchone()[0]

    assert counts["fct_store_product_week"] == 24
    assert weeks.loc[4, "units_sold"] == 0 and weeks.loc[4, "units_returned"] == 1
    assert weeks["on_discount"].sum() == 4
    # ISO week 3 of 2019 starts on 14 January, which is launch week index 1.
    assert weeks.loc[1, "restock_units"] == 4
    # Two lines at the same timestamp are one receipt: three visits in total.
    assert receipts == 3
    # Two visits to store 10 beat one more recent visit to store 20.
    assert home == 10


def test_shared_cards_are_flagged(tmp_path) -> None:
    warehouse = tmp_path / "w.duckdb"
    with duckdb.connect(str(warehouse)) as con:
        make_raw(con, [1.0] * 12)
        burst = pd.DataFrame(
            {
                "client_id": 99,
                "store_id": 10,
                "product_id": 1,
                "purchased_at": pd.date_range("2019-11-29 09:00", periods=10, freq="30min"),
                "qty": 1,
            }
        )
        con.register("burst", burst)
        con.execute("insert into raw.purchases select * from burst")

    build_marts(warehouse, SQL / "marts")

    with duckdb.connect(str(warehouse)) as con:
        flags = dict(con.execute("select client_id, is_shared_card from marts.client_profile").fetchall())
    assert flags[99] is True
    assert flags[7] is False
