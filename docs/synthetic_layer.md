# Synthetic Luxury Retail Layer

## Purpose

The synthetic layer translates public observed retail structure into a fictional Couture Atelier operating environment. It is not Louis Vuitton, LVMH, or any private luxury-company data.

## Generated tables

- `customers.parquet`: anonymous customer IDs, generalized regions, acquisition channel, and explicit consent status.
- `products.parquet`: 100 deterministic hero products with fictional categories, materials, colors, collections, price tiers, launch dates, and list prices.
- `stores.parquet`: six fictional boutiques, flagships, and an airport location across generalized regions.
- `inventory_daily.parquet`: 90 days of deterministic stock simulation for every store-product pair.

## Simulation rules

- Random seed: `42`.
- Product selection: top 100 observed products by derived net revenue, then remapped to fictional product IDs.
- Product attributes, pricing, and store metadata are synthetic.
- Inventory demand is simulated with a gamma-poisson process.
- Units sold cannot exceed opening stock plus receipts.
- Transfers are currently zero and will be introduced in the allocation stage.
- Stockout status is derived from closing stock.

## Provenance

Every generated table carries `data_provenance = synthetic`. The layer manifest records the seed, table sizes, simulation horizon, and source dependencies. Synthetic outcomes must never be reported as observed commercial results.

## Contract checks

For every inventory row:

```text
closing_stock = opening_stock + units_received + transfer_in
                - units_sold - transfer_out
```

The inventory generator and tests enforce nonnegative stock and this reconciliation equation.

## Known limitation

The current layer simulates inventory independently of observed product-country demand. The next inventory iteration will connect demand forecasts to store allocation and introduce transfer decisions, stockout risk, and service-level objectives.
