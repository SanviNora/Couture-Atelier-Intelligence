"""Build a deterministic synthetic luxury retail layer for Couture Atelier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROCESSED_DIR = Path("data/processed")
SYNTHETIC_DIR = Path("data/synthetic")
CUSTOMER_INPUT = PROCESSED_DIR / "customer_360.parquet"
PRODUCT_INPUT = PROCESSED_DIR / "product_catalog.parquet"
DAILY_INPUT = PROCESSED_DIR / "daily_sales.parquet"
SEED = 42
HERO_PRODUCT_COUNT = 100
SIMULATION_DAYS = 90

STORES = [
    ("CA-PAR-FLG", "Paris", "Western Europe", "flagship"),
    ("CA-LON-BOT", "London", "Western Europe", "boutique"),
    ("CA-MIL-BOT", "Milan", "Southern Europe", "boutique"),
    ("CA-NYC-FLG", "New York", "North America", "flagship"),
    ("CA-TOK-BOT", "Tokyo", "East Asia", "boutique"),
    ("CA-DXB-AIR", "Dubai", "Middle East", "airport"),
]
CATEGORIES = ["leather_goods", "ready_to_wear", "shoes", "jewelry", "travel"]
MATERIALS = ["calfskin", "canvas", "silk", "wool", "gold vermeil"]
COLORS = ["noir", "cognac", "ivory", "oxblood", "forest", "navy"]
COLLECTIONS = ["heritage", "atelier", "seasonal", "travel_edit"]
PRICE_TIERS = ["entry", "core", "high", "exceptional"]
PRICE_BY_TIER = {"entry": 950.0, "core": 1850.0, "high": 4200.0, "exceptional": 9800.0}


def stable_bucket(value: object, modulo: int) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulo


def build_customers(source: pd.DataFrame) -> pd.DataFrame:
    customers = source[["customer_id", "first_purchase_at", "primary_country"]].copy()
    customers = customers.rename(columns={"primary_country": "source_country"})
    customers["customer_id"] = "CA-CLIENT-" + customers["customer_id"].astype("string")
    country_region = {
        "United Kingdom": "Western Europe",
        "France": "Western Europe",
        "Germany": "Western Europe",
        "Spain": "Southern Europe",
        "Italy": "Southern Europe",
        "Japan": "East Asia",
        "Singapore": "Southeast Asia",
        "Australia": "Oceania",
        "USA": "North America",
        "United States": "North America",
    }
    customers["home_region"] = customers["source_country"].map(country_region).fillna("Other")
    customers["signup_date"] = pd.to_datetime(customers["first_purchase_at"]).dt.date.astype("string")
    customers["acquisition_channel"] = customers["customer_id"].map(
        lambda value: ["store", "website", "event", "referral"][stable_bucket(value, 4)]
    )
    customers["consent_status"] = customers["customer_id"].map(
        lambda value: "consented" if stable_bucket(value, 10) < 8 else "not_consented"
    )
    customers["data_provenance"] = "synthetic"
    return customers[
        [
            "customer_id",
            "home_region",
            "signup_date",
            "acquisition_channel",
            "consent_status",
            "data_provenance",
        ]
    ]


def build_products(source: pd.DataFrame) -> pd.DataFrame:
    products = source.nlargest(HERO_PRODUCT_COUNT, "net_revenue").reset_index(drop=True)
    products = products.rename(columns={"product_id": "source_product_id"})
    products["product_id"] = products["source_product_id"].map(
        lambda value: f"CA-PROD-{stable_bucket(value, 10_000_000):07d}"
    )
    products["category"] = products["source_product_id"].map(
        lambda value: CATEGORIES[stable_bucket(value, len(CATEGORIES))]
    )
    products["subcategory"] = products.apply(
        lambda row: f"{row['category']}_signature_{stable_bucket(row['source_product_id'], 6) + 1}",
        axis=1,
    )
    products["material"] = products["source_product_id"].map(
        lambda value: MATERIALS[stable_bucket(value, len(MATERIALS))]
    )
    products["color_family"] = products["source_product_id"].map(
        lambda value: COLORS[stable_bucket(value, len(COLORS))]
    )
    products["collection"] = products["source_product_id"].map(
        lambda value: COLLECTIONS[stable_bucket(value, len(COLLECTIONS))]
    )
    products["price_tier"] = products["source_product_id"].map(
        lambda value: PRICE_TIERS[stable_bucket(value, len(PRICE_TIERS))]
    )
    products["launch_date"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        products["source_product_id"].map(lambda value: stable_bucket(value, 730)), unit="D"
    )
    products["list_price"] = products["price_tier"].map(PRICE_BY_TIER)
    products["heritage_flag"] = products["collection"].eq("heritage")
    products["data_provenance"] = "synthetic"
    return products[
        [
            "product_id",
            "category",
            "subcategory",
            "material",
            "color_family",
            "collection",
            "price_tier",
            "launch_date",
            "list_price",
            "heritage_flag",
            "data_provenance",
        ]
    ]


def build_stores() -> pd.DataFrame:
    stores = pd.DataFrame(STORES, columns=["store_id", "city", "region", "store_type"])
    stores["opening_date"] = pd.Timestamp("2018-01-01")
    stores["floor_area_band"] = stores["store_type"].map(
        {"flagship": "large", "boutique": "medium", "airport": "small"}
    )
    stores["tourism_index"] = stores["city"].map(
        {"Paris": 0.95, "London": 0.88, "Milan": 0.76, "New York": 0.91, "Tokyo": 0.84, "Dubai": 0.82}
    )
    stores["affluence_index"] = stores["city"].map(
        {"Paris": 0.90, "London": 0.87, "Milan": 0.78, "New York": 0.92, "Tokyo": 0.86, "Dubai": 0.89}
    )
    stores["data_provenance"] = "synthetic"
    return stores


def build_inventory(products: pd.DataFrame, stores: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    end_date = pd.to_datetime(demand["transaction_date"]).max()
    dates = pd.date_range(end=end_date, periods=SIMULATION_DAYS, freq="D")
    product_ids = products["product_id"].tolist()
    rows: list[dict[str, object]] = []
    for store in stores.itertuples(index=False):
        for product_id in product_ids:
            opening_stock = int(rng.integers(3, 16))
            for date in dates:
                expected_demand = max(0.2, float(rng.gamma(shape=1.4, scale=1.2)))
                units_received = int(rng.choice([0, 0, 0, 2, 4]))
                units_sold = min(opening_stock + units_received, int(rng.poisson(expected_demand)))
                closing_stock = opening_stock + units_received - units_sold
                rows.append(
                    {
                        "date": date,
                        "store_id": store.store_id,
                        "product_id": product_id,
                        "opening_stock": opening_stock,
                        "units_received": units_received,
                        "units_sold": units_sold,
                        "transfer_in": 0,
                        "transfer_out": 0,
                        "closing_stock": closing_stock,
                        "stockout_flag": closing_stock == 0,
                        "data_provenance": "synthetic",
                    }
                )
                opening_stock = closing_stock
    return pd.DataFrame(rows)


def build_synthetic_layer() -> dict[str, int]:
    customers = build_customers(pd.read_parquet(CUSTOMER_INPUT))
    products = build_products(pd.read_parquet(PRODUCT_INPUT))
    stores = build_stores()
    demand = pd.read_parquet(DAILY_INPUT, columns=["transaction_date"])
    inventory = build_inventory(products, stores, demand)
    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "customers": customers,
        "products": products,
        "stores": stores,
        "inventory_daily": inventory,
    }.items():
        frame.to_parquet(SYNTHETIC_DIR / f"{name}.parquet", index=False)
    manifest = {
        "layer": "Couture Atelier synthetic luxury retail",
        "seed": SEED,
        "hero_product_count": HERO_PRODUCT_COUNT,
        "simulation_days": SIMULATION_DAYS,
        "data_provenance": "synthetic",
        "source_dependencies": ["customer_360.parquet", "product_catalog.parquet", "daily_sales.parquet"],
        "tables": {name: len(frame) for name, frame in {
            "customers": customers,
            "products": products,
            "stores": stores,
            "inventory_daily": inventory,
        }.items()},
    }
    (SYNTHETIC_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest["tables"]


if __name__ == "__main__":
    print(build_synthetic_layer())
