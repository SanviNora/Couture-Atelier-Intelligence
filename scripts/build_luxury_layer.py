"""Build the calibrated luxury layer on top of the Visuelle marts.

Writes the ``lux`` schema: product prices and hero flags, store formats and
advisor counts, client value tiers and consent, and purchase values in euros.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from couture_atelier.luxury import (
    calibrate_concentration,
    client_tier,
    consent_status,
    load_calibration,
    map_price_ladder,
    price_tier,
    stable_uniform,
    top_share,
)
from couture_atelier.paths import REPORTS, WAREHOUSE


def build_products(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    a = cfg["assumptions"]
    products = con.execute(
        """
        select p.product_id, p.season, p.category, p.price_index, p.has_attributes,
               coalesce(sum(w.units_sold) filter (where not w.on_discount), 0) as full_price_units,
               count(*) filter (where not w.on_discount) as full_price_store_weeks
        from marts.dim_product p
        left join marts.fct_store_product_week w using (product_id)
        group by all
        """
    ).fetchdf()
    priced = products["price_index"].notna()
    products.loc[priced, "list_price_eur"] = map_price_ladder(
        products.loc[priced, "price_index"], a["price_ladder_eur"]["min"], a["price_ladder_eur"]["max"]
    )
    products["list_price_eur"] = products["list_price_eur"].fillna(products["list_price_eur"].median())
    products["price_tier"] = price_tier(products["list_price_eur"], a["price_tiers"])
    products["full_price_velocity"] = products["full_price_units"] / products["full_price_store_weeks"].clip(lower=1)
    threshold = products.groupby("season")["full_price_velocity"].transform(
        lambda v: v.quantile(1 - a["hero_share"]["value"])
    )
    products["hero_piece"] = products["has_attributes"] & (products["full_price_velocity"] >= threshold)
    products["data_provenance"] = "synthetic"
    return products[["product_id", "season", "category", "list_price_eur", "price_tier",
                     "full_price_velocity", "hero_piece", "data_provenance"]]


def build_clients(con: duckdb.DuckDBPyConnection, cfg: dict) -> tuple[pd.DataFrame, dict]:
    a = cfg["assumptions"]
    target = cfg["targets"]["top_0_1pct_client_spend_share"]["value"]
    clients = con.execute(
        """
        select c.client_id, c.home_store_id, c.visits, c.units,
               sum(f.qty * p.list_price_eur) as base_spend_eur
        from marts.client_profile c
        join marts.fct_purchase f using (client_id)
        join lux.product p using (product_id)
        group by all
        """
    ).fetchdf()
    alpha, spend = calibrate_concentration(clients["base_spend_eur"].to_numpy(), target)
    clients["modelled_spend_eur"] = spend
    clients["spend_multiplier"] = spend / clients["base_spend_eur"]
    clients["value_tier"] = client_tier(clients["modelled_spend_eur"], a["client_tiers"])
    clients["consent_status"] = consent_status(
        clients["client_id"], clients["visits"], a["consent"]["base_rate"], a["consent"]["max_rate"], a["seed"]
    )
    book = a["advisor_book_size"]["value"]
    advisors = clients.groupby("home_store_id")["client_id"].transform("size").div(book).apply(np.ceil).astype(int)
    slot = (stable_uniform(clients["client_id"], a["seed"] + 1) * advisors).astype(int)
    clients["advisor_id"] = clients["home_store_id"].astype(str) + "-A" + (slot + 1).astype(str).str.zfill(2)
    clients["data_provenance"] = "synthetic"
    stats = {
        "alpha": round(float(alpha), 4),
        "observed_top_0_1pct_unit_share": round(top_share(clients["units"].to_numpy(float), 0.001), 4),
        "base_top_0_1pct_spend_share": round(top_share(clients["base_spend_eur"].to_numpy(), 0.001), 4),
        "modelled_top_0_1pct_spend_share": round(top_share(spend, 0.001), 4),
        "target_top_0_1pct_spend_share": target,
    }
    return clients.drop(columns=["units"]), stats


def build_stores(con: duckdb.DuckDBPyConnection, clients: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    a = cfg["assumptions"]
    stores = con.execute("select store_id, clients, client_units, in_sales_panel from marts.dim_store").fetchdf()
    cutoff = stores["client_units"].quantile(1 - a["flagship_share"]["value"])
    stores["store_format"] = np.where(stores["client_units"] >= cutoff, "flagship", "boutique")
    advisors = clients.groupby("home_store_id")["advisor_id"].nunique().rename("advisors")
    stores = stores.merge(advisors, left_on="store_id", right_index=True, how="left")
    stores["advisors"] = stores["advisors"].fillna(0).astype(int)
    stores["data_provenance"] = "synthetic"
    return stores


def build_luxury_layer(warehouse: Path = WAREHOUSE) -> dict[str, object]:
    cfg = load_calibration()
    with duckdb.connect(str(warehouse)) as con:
        con.execute("create schema if not exists lux")
        products = build_products(con, cfg)
        con.execute("create or replace table lux.product as select * from products")
        clients, stats = build_clients(con, cfg)
        con.execute("create or replace table lux.client as select * from clients")
        stores = build_stores(con, clients, cfg)
        con.execute("create or replace table lux.store as select * from stores")
        con.execute(
            """
            create or replace table lux.purchase_value as
            select f.client_id, f.store_id, f.product_id, f.purchased_at, f.receipt_id, f.qty,
                   p.list_price_eur,
                   f.qty * p.list_price_eur * c.spend_multiplier as value_eur,
                   'synthetic' as data_provenance
            from marts.fct_purchase f
            join lux.product p using (product_id)
            join lux.client c using (client_id)
            """
        )
        total_check = con.execute(
            "select abs(sum(value_eur) - (select sum(modelled_spend_eur) from lux.client)) "
            "/ (select sum(modelled_spend_eur) from lux.client) from lux.purchase_value"
        ).fetchone()[0]
        if total_check > 1e-6:
            raise ValueError(f"Purchase values do not reconcile with client spend ({total_check:.2e})")

    report = {
        "concentration": stats,
        "clients": len(clients),
        "value_tiers": clients["value_tier"].value_counts().to_dict(),
        "consent_rate": round(float(clients["consent_status"].eq("consented").mean()), 4),
        "advisors": int(stores["advisors"].sum()),
        "store_formats": stores["store_format"].value_counts().to_dict(),
        "hero_pieces": int(products["hero_piece"].sum()),
        "price_tiers": products["price_tier"].value_counts().to_dict(),
        "median_list_price_eur": float(products["list_price_eur"].median()),
        "data_provenance": "synthetic",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "luxury_layer.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_luxury_layer(), indent=2))
