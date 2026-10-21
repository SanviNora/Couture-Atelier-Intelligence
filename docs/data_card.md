# Data Card

What data this project uses, where it comes from, and what may be shown publicly.

## Sources

| Source | What it provides | Terms | Public? |
|---|---|---|---|
| Visuelle 2.0 (Skenderi et al., CVPRW 2022) | Real fashion retail: 665k anonymised clients, 110 stores, weekly restocks, store-level sales and discounts, 5.5k products | Access by request form; authors ask for citation; no license stated | **No.** Raw files stay local. Only aggregates and model outputs are published |
| Calibrated luxury layer | Price tiers, hero pieces, VIC spend curve, consent, advisors, outreach experiments | Created by this project; every target cites its source in `configs/calibration.yml` | Yes |
| Google Trends (Oct 2021–Oct 2026) | Weekly US search interest for 30 luxury, aspirational and premium products, brands and themes, plus worldwide brand interest; 6 manual downloads | Google Trends terms, with attribution; raw downloads kept local with checksums | Aggregates and charts only (`reports/market_pulse.csv`) |
| Published industry figures | Top-client spend shares, revenue mix, regulation | Cited reports (Bain, BCG-Altagamma, LVMH, Kering, EU ESPR) | Yes, as cited numbers |
| UCI Online Retail II (legacy) | Former transaction source, UK giftware 2009–2011 | CC BY 4.0 | Retired from the pipeline |

## Public / private rule

- Raw files live outside the repository (`COUTURE_VISUELLE_DIR`) and under `data/`, which Git ignores.
- The live demo reads only `public_bundle/`. Every file there is declared in `public_bundle/MANIFEST.json` with a provenance of `aggregate`, `synthetic`, `model_output`, or `public_source`.
- `tests/test_public_bundle.py` fails if the bundle contains undeclared files, real client identifiers, raw transaction timestamps, or aggregate groups smaller than 10.
- Clients shown in the demo are synthetic. No real client's purchase history is ever displayed.

## Known limits

- Visuelle 2.0 is one Italian fast-fashion retailer (2016–2021). Prices are max-normalised, so all euro figures come from the calibrated luxury layer and are labelled as modelled.
- Luxury-specific behaviour (top-client concentration, scarcity, full-price policy) is modelled from published figures, not observed.
