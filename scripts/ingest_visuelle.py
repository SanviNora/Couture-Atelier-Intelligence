"""Register Visuelle 2.0 and load it into the local DuckDB warehouse.

Raw files stay where they were downloaded (``COUTURE_VISUELLE_DIR``); this
script records a checksum manifest, loads the tables into the ``raw`` schema,
and fails closed if any validation rule is broken.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from couture_atelier.paths import RAW, VISUELLE_DIR, WAREHOUSE

SOURCE_ID = "visuelle_2_0"
CITATION = (
    "Skenderi et al., 'The Multi-Modal Universe of Fast-Fashion: The Visuelle 2.0 "
    "Benchmark', CVPR Workshops 2022"
)
WEEKS = [str(week) for week in range(12)]

LOADS = {
    "sales": f"""
        select external_code as product_id, retail as store_id, season, category, color,
               fabric, image_path, release_date, restock as restock_events,
               {", ".join(f'"{w}" as w{w}' for w in WEEKS)}
        from read_csv_auto('{{dir}}/sales.csv')
    """,
    "price_discount": f"""
        select external_code as product_id, retail as store_id, price as price_index,
               {", ".join(f'"{w}" as d{w}' for w in WEEKS)}
        from read_csv_auto('{{dir}}/price_discount_series.csv')
    """,
    "restocks": """
        select external_code as product_id, retail as store_id, year, week, qty
        from read_csv_auto('{dir}/restocks.csv')
    """,
    "purchases": """
        select customer as client_id, retail as store_id, external_code as product_id,
               data as purchased_at, qty
        from read_csv_auto('{dir}/customer_data.csv', timestampformat='%Y-%m-%d %H:%M:%S')
    """,
    "gtrends": """
        select * from read_csv_auto('{dir}/vis2_gtrends_data.csv')
    """,
}

# Each rule is a query that must return zero.
VALIDATIONS = {
    "sales keys unique": "select count(*) - count(distinct (product_id, store_id)) from raw.sales",
    # Weekly sales are net of returns: small negatives are returns, anything larger is corrupt.
    "sales weeks net of returns are plausible": "select count(*) from raw.sales where least("
    + ", ".join(f"w{w}" for w in WEEKS)
    + ") < -5",
    "sales release date present": "select count(*) from raw.sales where release_date is null",
    "discounts within [0, 1]": "select count(*) from raw.price_discount where least("
    + ", ".join(f"d{w}" for w in WEEKS)
    + ") < 0 or greatest("
    + ", ".join(f"d{w}" for w in WEEKS)
    + ") > 1",
    "price index within (0, 1]": "select count(*) from raw.price_discount where price_index <= 0 or price_index > 1",
    "restock quantity positive": "select count(*) from raw.restocks where qty <= 0",
    "restock week valid": "select count(*) from raw.restocks where week not between 1 and 53",
    "purchase keys present": "select count(*) from raw.purchases where client_id is null "
    "or store_id is null or product_id is null or purchased_at is null",
    "purchase quantity positive": "select count(*) from raw.purchases where qty < 1",
    "purchase dates in range": "select count(*) from raw.purchases "
    "where purchased_at < '2016-11-01' or purchased_at >= '2022-01-01'",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_raw(con: duckdb.DuckDBPyConnection, source_dir: Path) -> dict[str, int]:
    con.execute("create schema if not exists raw")
    counts = {}
    for table, query in LOADS.items():
        con.execute(f"create or replace table raw.{table} as {query.format(dir=source_dir)}")
        counts[table] = con.execute(f"select count(*) from raw.{table}").fetchone()[0]
    return counts


def validate(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    failures = {name: con.execute(sql).fetchone()[0] for name, sql in VALIDATIONS.items()}
    broken = {name: n for name, n in failures.items() if n}
    if broken:
        raise ValueError(f"Visuelle validation failed: {broken}")
    return failures


def coverage(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Purchases that reference stores or products absent from the sales table."""
    return {
        "purchase_stores_without_sales": con.execute(
            "select count(distinct store_id) from raw.purchases "
            "where store_id not in (select store_id from raw.sales)"
        ).fetchone()[0],
        "purchase_products_without_sales": con.execute(
            "select count(distinct product_id) from raw.purchases "
            "where product_id not in (select product_id from raw.sales)"
        ).fetchone()[0],
    }


def ingest(source_dir: Path = VISUELLE_DIR, warehouse: Path = WAREHOUSE) -> dict[str, object]:
    if not source_dir.exists():
        raise FileNotFoundError(
            f"Visuelle 2.0 not found at {source_dir}. Request it at "
            "https://humaticslab.github.io/forecasting/visuelle and set COUTURE_VISUELLE_DIR."
        )
    warehouse.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect(str(warehouse)) as con:
        counts = load_raw(con, source_dir)
        validate(con)
        gaps = coverage(con)
    files = sorted(source_dir.glob("*.csv"))
    manifest = {
        "source_id": SOURCE_ID,
        "citation": CITATION,
        "source_dir": str(source_dir),
        "registered_at_utc": datetime.now(UTC).isoformat(),
        "files": {path.name: sha256(path) for path in files},
        "row_counts": counts,
        "coverage_gaps": gaps,
        "data_provenance": "observed_public",
        "redistribution": "not permitted; raw rows stay local",
    }
    out = RAW / SOURCE_ID / "manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    report = ingest()
    print(json.dumps({k: report[k] for k in ("row_counts", "coverage_gaps")}, indent=2))
