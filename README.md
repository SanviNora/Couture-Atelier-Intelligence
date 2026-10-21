# Couture Atelier Intelligence

A reproducible luxury retail analytics platform built from public fashion and retail data plus a clearly documented synthetic luxury layer.

## Purpose

Couture Atelier Intelligence connects three decisions:

1. Demand forecasting by product, boutique, region, and channel.
2. Client segmentation and customer lifetime value.
3. Inventory allocation and consent-aware clienteling recommendations.

The project is a portfolio simulation. It is not affiliated with Louis Vuitton, LVMH, or any other luxury house, and it does not claim to represent private company data.

## Data principles

- Public observations and simulated records remain distinguishable.
- Source licenses and provenance are recorded before use.
- Customer identities are anonymous and synthetic in the integrated layer.
- Sensitive attributes and inferred wealth are excluded.
- Forecasts, simulations, and observed outcomes are reported separately.
- Recommendations include an explicit no-action option.

## Planned stack

- Python for ingestion, feature engineering, and modeling
- DuckDB and Parquet for local analytical storage
- SQL for transformations
- pytest for data-quality checks
- Streamlit for the executive interface

## Repository map

```text
src/couture_atelier/  package code
sql/                  analytical transformations
configs/              source and project configuration
data/                 raw, staging, synthetic, and processed data (ignored)
tests/                automated checks
docs/                 project documentation
notebooks/            exploratory analysis
```

## Current milestone

The project is being rebuilt on **Visuelle 2.0**, real fashion retail data from one retailer: 665k clients, 110 stores, store-level weekly sales and discounts, and weekly restocks. On top of it sits a **luxury layer calibrated to published figures** (`configs/calibration.yml`). For example, the top 0.1% of clients are calibrated to hold 23% of spend (BCG-Altagamma 2025), up from the 4.7% this retailer shows.

Rebuild the data foundation (about 10 seconds):

```bash
make data   # ingest Visuelle, build SQL marts, build the luxury layer
make test
```

Raw Visuelle files stay local and are never published; see `docs/data_card.md`. The modules after the data foundation (client value, clienteling, demand, allocation) are being rebuilt in that order; until then those scripts still use the retired UCI outputs.

## Study guide

Use [docs/COUTURE_ATELIER_STUDY_GUIDE.md](docs/COUTURE_ATELIER_STUDY_GUIDE.md) as the complete ELI5-to-technical manual for studying and explaining this project.

Download the formatted [COUTURE_ATELIER_STUDY_GUIDE.pdf](docs/COUTURE_ATELIER_STUDY_GUIDE.pdf) version for offline study.
