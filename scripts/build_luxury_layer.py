"""Build the calibrated luxury layer on top of the Visuelle marts.

Writes the ``lux`` schema: product prices and hero flags, store formats and
advisor counts, clients with consent and advisor, and time-safe receipt and
purchase values in euros. Shared or promotional cards are not clients and
are left out.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from couture_atelier.luxury import (
    calibrate_trade_up,
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


def build_clients(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    a = cfg["assumptions"]
    clients = con.execute(
        "select client_id, home_store_id from marts.client_profile where not is_shared_card and not is_staff_like"
    ).fetchdf()
    clients["consent_status"] = consent_status(clients["client_id"], a["consent"]["rate"], a["seed"])
    book = a["advisor_book_size"]["value"]
    advisors = clients.groupby("home_store_id")["client_id"].transform("size").div(book).apply(np.ceil).astype(int)
    slot = (stable_uniform(clients["client_id"], a["seed"] + 1) * advisors).astype(int)
    clients["advisor_id"] = clients["home_store_id"].astype(str) + "-A" + (slot + 1).astype(str).str.zfill(2)
    clients["data_provenance"] = "synthetic"
    return clients


def build_receipts(con: duckdb.DuckDBPyConnection, cfg: dict) -> tuple[pd.DataFrame, dict]:
    target = cfg["targets"]["vic_revenue_share"]
    cap = cfg["assumptions"]["trade_up"]["cap_prior_visits"]
    receipts = con.execute(
        """
        select f.client_id, f.receipt_id, any_value(f.store_id) as store_id,
               min(f.purchased_at) as purchased_at, sum(f.qty) as units,
               sum(f.qty * p.list_price_eur) as base_value_eur
        from marts.fct_purchase f
        join lux.product p using (product_id)
        where f.client_id in (select client_id from lux.client)
        group by f.client_id, f.receipt_id
        """
    ).fetchdf()
    receipts = receipts.sort_values(["client_id", "purchased_at", "receipt_id"]).reset_index(drop=True)
    receipts["prior_visits"] = receipts.groupby("client_id").cumcount()
    codes = pd.factorize(receipts["client_id"])[0]
    beta, values = calibrate_trade_up(
        receipts["base_value_eur"].to_numpy(), receipts["prior_visits"].to_numpy(), codes,
        target["value"], fraction=target["client_fraction"], cap=cap,
    )
    receipts["value_eur"] = values
    receipts["data_provenance"] = "synthetic"
    base_by_client = np.bincount(codes, weights=receipts["base_value_eur"].to_numpy())
    by_client = np.bincount(codes, weights=values)
    years = (receipts["purchased_at"].max() - receipts["purchased_at"].min()).days / 365
    top_tier = np.sort(by_client)[::-1][: max(1, round(len(by_client) * 0.001))]
    stats = {
        "trade_up_beta": round(float(beta), 4),
        "trade_up_cap_prior_visits": cap,
        "target_top1pct_share": target["value"],
        "base_top1pct_share": round(top_share(base_by_client, 0.01), 4),
        "modelled_top1pct_share": round(top_share(by_client, 0.01), 4),
        "cross_check_top_0_1pct_share": round(top_share(by_client, 0.001), 4),
        "cross_check_top_0_1pct_mean_annual_spend_eur": round(float(top_tier.mean() / years)),
        "median_client_annual_spend_eur": round(float(np.median(by_client) / years)),
    }
    return receipts, stats


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
        clients = build_clients(con, cfg)
        con.execute("create or replace table lux.client as select * from clients")
        receipts, stats = build_receipts(con, cfg)
        con.execute("create or replace table lux.receipt as select * from receipts")
        stores = build_stores(con, clients, cfg)
        con.execute("create or replace table lux.store as select * from stores")
        con.execute(
            """
            create or replace table lux.purchase_value as
            select f.client_id, f.store_id, f.product_id, f.purchased_at, f.receipt_id, f.qty,
                   p.list_price_eur,
                   f.qty * p.list_price_eur * r.value_eur / r.base_value_eur as value_eur,
                   'synthetic' as data_provenance
            from marts.fct_purchase f
            join lux.product p using (product_id)
            join lux.receipt r using (client_id, receipt_id)
            """
        )
        gap = con.execute(
            "select abs(sum(value_eur) - (select sum(value_eur) from lux.receipt)) "
            "/ (select sum(value_eur) from lux.receipt) from lux.purchase_value"
        ).fetchone()[0]
        if gap > 1e-6:
            raise ValueError(f"Purchase values do not reconcile with receipts ({gap:.2e})")
        shared, staff = con.execute(
            "select count(*) filter (where is_shared_card), count(*) filter (where is_staff_like and not is_shared_card) "
            "from marts.client_profile"
        ).fetchone()

    report = {
        "concentration": stats,
        "clients": len(clients),
        "shared_cards_excluded": int(shared),
        "staff_like_excluded": int(staff),
        "receipts": len(receipts),
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
