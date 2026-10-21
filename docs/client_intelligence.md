# VIC Client Intelligence

Who will be valuable next year, who might become a VIC, and who is drifting away. Built on 665k real clients from Visuelle 2.0 with the calibrated luxury layer (`docs/data_card.md`).

Run: `make all` (data foundation, then models). Full metrics: `reports/client_value.json`; decile calibration: `reports/client_value_deciles.csv`.

## Set-up

Three chronological windows. Features use only purchases up to the cut-off; the outcome is the next 12 months.

| Window | Cut-off | Outcome | Role |
|---|---|---|---|
| Train | 31 Dec 2017 | 2018 | Fits the machine-learning models |
| Test | 31 Dec 2018 | 2019 | Main evaluation |
| COVID | 31 Dec 2019 | 2020 | Stress test, reported separately |

Shared or promotional cards (25 IDs) and staff-like accounts (36 IDs with 100+ visits) are excluded from client modelling.

## Models compared

- **Naive:** next year's spend equals the last 12 months.
- **Probabilistic:** BG/NBD for visits and Gamma-Gamma for spend per visit, implemented from the original papers (`src/couture_atelier/clv.py`). The fitted dropout parameter is below 1, so the textbook formula for expected visits does not apply; expected visits are computed exactly by integrating over each client's churn probability, and tested against the textbook formula where it does apply.
- **Gradient boosting:** Poisson regression for spend and classifiers for VIC upgrade and return, trained on the 2018 outcome window and tested on 2019. Features include recency, 12-month and 90/180-day activity, full-price and hero-piece share, and the BG/NBD outputs.

## Results (test window: 2019)

| Question | Naive | Probabilistic | Gradient boosting |
|---|---:|---:|---:|
| Rank clients by next-year spend (Spearman) | 0.383 | 0.401 | **0.426** |
| Next year's VICs found in predicted top 1% | 45.0% | **45.6%** | 44.2% |
| Total spend bias | −15% | −33% | +36% |
| Will the client return next year? (AUC) | 0.659 (recency) | 0.371 (P alive) | **0.714** |
| VIC upgrade among non-VICs (AUC / avg. precision) | 0.928 / **0.131** | | **0.935** / 0.130 |

What this means:

- **Ranking:** gradient boosting ranks clients best. That ranking is what clienteling uses.
- **Finding next year's VICs:** last year's spend is a hard baseline to beat, because VIC status is persistent. No model clearly beats it, and the project says so.
- **Totals:** no model is calibrated in total. The probabilistic model under-predicts because Gamma-Gamma assumes spend per visit is stable, while clients here trade up. Gradient boosting over-predicts because it learned from 2017, when most clients were new and trading up fast. Predictions are used for ranking and prioritisation; totals would need recalibration before any finance use.
- **Return and lapse:** BG/NBD's "probability still active" ranks worse than random (AUC 0.37), because it treats every one-time buyer as certainly active. A dedicated return classifier fixes this (AUC 0.71).
- **VIC upgrade:** the model ranks upgrade candidates slightly better overall (AUC 0.935 vs 0.928), but its precision on the very top candidates is no better than ranking by current spend.

## COVID stress test (2020)

Return rate fell from 37.5% to 10.6%. Every model over-predicted total spend by 280–690%, and ranking quality fell from about 0.42 to 0.24. None of these models can anticipate a shock like store closures; in production this is what drift monitoring and a fallback rule are for.

## Output for later phases

`lux.client_value` scores every client as of 31 December 2019: value tier (top tier, VIC, core, aspirational, lapsed), last-12-month spend, predicted next-year spend, probability of returning, VIC upgrade probability, and full-price and hero-piece shares.

| Tier | Clients |
|---|---:|
| Top tier (top 0.1% by 12-month spend) | 619 |
| VIC (next 0.9%) | 5,571 |
| Core (next 9%) | 55,717 |
| Aspirational | 311,004 |
| Lapsed (no visit in 12 months) | 246,165 |
