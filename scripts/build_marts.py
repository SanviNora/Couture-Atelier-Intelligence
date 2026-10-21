"""Run the SQL marts in ``sql/marts`` in order against the DuckDB warehouse."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from couture_atelier.paths import SQL, WAREHOUSE

# Each check is a query that must return zero.
CHECKS = {
    "one row per product": "select count(*) - count(distinct product_id) from marts.dim_product",
    "one row per store": "select count(*) - count(distinct store_id) from marts.dim_store",
    "12 weeks per store-product": "select count(*) from (select product_id, store_id, count(*) n "
    "from marts.fct_store_product_week group by all having n <> 12)",
    "units sold non-negative": "select count(*) from marts.fct_store_product_week where units_sold < 0",
    "one row per client": "select count(*) - count(distinct client_id) from marts.client_profile",
    "one stock row per store-product-week": "select count(*) - (select count(*) from marts.fct_store_product_week) "
    "from marts.fct_stock_week",
    "every purchase has a client profile": "select count(*) from marts.fct_purchase "
    "where client_id not in (select client_id from marts.client_profile)",
}


def build_marts(warehouse: Path = WAREHOUSE, sql_dir: Path = SQL / "marts") -> dict[str, int]:
    with duckdb.connect(str(warehouse)) as con:
        con.execute("create schema if not exists marts")
        for script in sorted(sql_dir.glob("*.sql")):
            con.execute(script.read_text())
        broken = {name: n for name, sql in CHECKS.items() if (n := con.execute(sql).fetchone()[0])}
        if broken:
            raise ValueError(f"Mart checks failed: {broken}")
        tables = con.execute(
            "select table_name from information_schema.tables where table_schema = 'marts' order by 1"
        ).fetchall()
        return {name: con.execute(f"select count(*) from marts.{name}").fetchone()[0] for (name,) in tables}


if __name__ == "__main__":
    print(json.dumps(build_marts(), indent=2))
