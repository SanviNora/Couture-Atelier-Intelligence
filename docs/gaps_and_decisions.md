# Gaps and Decisions Log

Every data issue, modelling shortcut, and judgment call found while building the project. Each one is either fixed, flagged for a later phase, or accepted as a stated limitation. The interview questions column feeds the interview guide.

Status: **Fixed** (resolved in code, with a test where possible) · **Flagged** (kept and marked; a later phase decides) · **Accepted** (a stated limitation)

| ID | Area | Finding | Status | Where handled | Interview question it invites |
|---|---|---|---|---|---|
| G1 | Data | Visuelle weekly sales are net of returns: 3,445 returned units against 1.6M sold, almost all −1, clustered in weeks 4–5 and 9–10. Validation stopped the pipeline on first run. | Fixed | `ingest_visuelle.py` allows negatives down to −5 and stops on anything larger; `fct_store_product_week` splits `units_sold` and `units_returned`; tested | "Your pipeline failed on its first run. What did you do?" |
| G2 | Data | Returns cannot be linked to clients; client purchase records contain only purchases. Client spend is not net of returns. | Accepted | Returns are 0.2% of units, so the effect on client value is negligible | "How do returns affect your CLV numbers?" |
| G3 | Data | Client purchases reference 19 stores and 222 products that are not in the sales panel. | Flagged | `dim_store.in_sales_panel`, `dim_product.has_attributes` | "How did you handle records that don't join cleanly?" |
| G4 | Data | One store-product week has a missing discount value; reshaping silently dropped that week until nulls were kept. | Fixed | Reshape keeps nulls; `discount_missing` flag; discount treated as 0 | "How do you catch silent row loss in SQL?" |
| G5 | Data | Visuelle prices are max-normalised for confidentiality, so real euro prices are unknown. | Accepted | Price index mapped onto a €650–€9,500 ladder by percentile (`configs/calibration.yml`); all euro figures are labelled as modelled | "Where do your euro numbers come from?" |
| G6 | Data | About 25 IDs are shared or promotional cards, not people. The top three made 18,737, 7,543 and 4,953 visits across 69 stores in about four days around Black Friday 2017 and New Year 2018. They inflated top-client concentration from 2.7% to 4.7%. | Fixed | Rule: an ID with 10+ receipts in one day or 4+ stores in one day is a shared card; excluded from client modelling, kept in store demand | "How did you find and handle outliers in client data?" |
| G7 | Leakage | The first luxury layer set each client's spend multiplier from their whole history, including future periods that phase 2 predicts. A model trained on those values would have seen the future. | Fixed | Replaced with a time-safe trade-up: each purchase is worth list value × (1 + prior visits)^β, using only past visits; one global β calibrated to the 23% target | "Did you have any leakage, and how did you find it?" |
| G8 | Leakage | Home store (and so advisor assignment) uses the client's whole history. | Accepted | Used only for operational assignment, not as a model feature | "Is anything in your features computed with future data?" |
| G9 | Scope | Visuelle is one Italian fast-fashion retailer (2016–2021), not a luxury house. | Accepted | Luxury behaviour comes from the calibrated layer, with every target cited; stated in the data card | "Why should a luxury house trust results from fast-fashion data?" |
| G10 | Scope | 2020 is distorted by COVID store closures. | Flagged | Phase 2 evaluates on 2018 and 2019, and reports 2020 separately as a stress test | "How would your model behave under a shock?" |
| G11 | Scope | Google Trends data covers Visuelle's categories, colours and fabrics, not real luxury products. | Accepted | Cut to fit the 1–2 week plan; possible extension | "What external signals would you add?" |
| G12 | Assumption | Consent is modelled as independent of behaviour, at a fixed opt-in rate, to avoid leaking future visits into consent. | Accepted | `configs/calibration.yml` | "Why not model consent from engagement?" |
| G13 | Retired | The original UCI data was UK giftware with many wholesale buyers. | Fixed | Replaced by Visuelle 2.0 | "Why did you change datasets mid-project?" |
