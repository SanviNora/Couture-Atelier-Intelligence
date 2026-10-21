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
- Streamlit or Tableau for the executive interface

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

The UCI Online Retail II source has been acquired, profiled, validated, staged, and transformed into customer, product, daily-sales, demand-panel, baseline-segmentation, and stronger demand-model artifacts. Raw and generated data are acquired locally and are not committed.

## Next step

The stronger demand benchmark, synthetic luxury retail layer, inventory-transfer decision layer, improved CLV baseline, churn model, behavioral segments, and first clienteling decision layer are complete. See `docs/data_contract.md`, `docs/forecasting.md`, `docs/synthetic_layer.md`, `docs/inventory_optimization.md`, `docs/client_intelligence.md`, `docs/clienteling.md`, and `docs/ingestion.md` for current lineage, modeling, simulation, and validation rules.

## Study guide

Use [docs/COUTURE_ATELIER_STUDY_GUIDE.md](docs/COUTURE_ATELIER_STUDY_GUIDE.md) as the complete ELI5-to-technical manual for studying and explaining this project.

Download the formatted [COUTURE_ATELIER_STUDY_GUIDE.pdf](docs/COUTURE_ATELIER_STUDY_GUIDE.pdf) version for offline study.
