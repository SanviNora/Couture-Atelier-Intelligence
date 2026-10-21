# VIC Client Intelligence

Who will be valuable next year, who might become a VIC, and who is drifting away. Built on 665k real clients from Visuelle 2.0 with the calibrated luxury layer (`docs/data_card.md`).

```bash
make select   # model selection on pre-test data only (about 30 minutes); writes configs/client_models.yml
make models   # refit the selection, test once on 2019, score clients
```

Outputs: `reports/client_value.json` (test results with confidence intervals), `reports/client_experiments.csv` (every configuration tried), `reports/sensitivity_client.csv`, `reports/client_feature_importance.csv`, `reports/client_value_deciles.csv`.

## How the models were chosen

| Data | Cut-offs | Outcome | Use |
|---|---|---|---|
| Training | 30 Jun, 31 Aug, 31 Oct, 31 Dec 2017 (pooled) | Next 12 months, all before 2019 | Fit models |
| Validation | 20% of clients at 31 Dec 2017, never trained on | 2018 | Choose models |
| Test | 31 Dec 2018 | 2019 | Used once, at the end |
| Stress test | 31 Dec 2019 | 2020 (COVID) | Reported separately |

Tried on validation (`reports/client_experiments.csv`): BG/NBD and MBG/NBD with Gamma-Gamma; gradient boosting trained on one cut-off or four pooled; three feature sets; direct Poisson, log-target and hurdle ("will they return" × "how much") structures; random hyperparameter searches; logistic regression; total recalibration; and rank blends.

Selection rules, fixed before the test:
- **Spend:** best ranking; near-ties (within 0.005 Spearman) broken by the share of next year's VICs found. Chosen: hurdle model.
- **Return:** a simpler model wins if within 0.002 AUC of the best. Chosen: logistic regression.
- **VIC upgrade:** best average precision. Chosen: class-balanced gradient boosting.

Shared or promotional cards (25 IDs) and staff-like accounts (36 IDs) are excluded.

## Results on the 2019 test

Differences are model minus baseline, with 95% paired-bootstrap intervals (200 resamples of clients).

| Question | Baseline | Selected model | Difference [95% CI] |
|---|---:|---:|---|
| Will the client return next year? (AUC) | 0.659 (recency) | **0.716** | +0.057 [0.055, 0.058] |
| Of next year's VICs, share found in predicted top 1% | 45.0% (last year's spend) | **46.6%** | +1.6 pts [0.9, 2.3] |
| VIC upgrade among non-VICs (avg. precision) | 0.131 (current spend) | **0.153** | +0.022 [0.012, 0.031] |
| Rank clients by next-year spend (Spearman) | 0.383 (last year's spend) | 0.399 | +0.016 [0.014, 0.019] |
| Total spend bias | −15% | +46% | |

Other models on the same test:

| Model | Spend Spearman | Total spend bias | Return AUC |
|---|---:|---:|---:|
| BG/NBD + Gamma-Gamma | 0.401 | −33% | 0.371 (P alive) |
| MBG/NBD + Gamma-Gamma | **0.406** | −34% | 0.652 (P alive) |

What this means:

- **Return prediction is the strongest result** and uses real behaviour only (visits are observed, not modelled). BG/NBD's "probability still active" ranks worse than random because it treats every one-time buyer as certainly active; MBG/NBD fixes that (0.37 → 0.65), and logistic regression does better still.
- **VIC identification and upgrade improve significantly** over current spend, which is a hard baseline because VIC status is persistent.
- **Spend ranking:** all approaches land between 0.38 and 0.41. The gain over the naive baseline is significant but small; the limit is the information in one to two years of history.
- **Totals are not calibrated.** Pooling cut-offs removed the bias on validation (about 0%) but it returned on the test (+46%): validation shares the training era, when clients were young and trading up fast; by 2019 they had matured. Spend predictions are used for ranking, not euro totals. See gaps log G29.

## Real behaviour only: next-year visits

Visits come straight from Visuelle with no modelled euros.

| Model | Visits Spearman | Total visits bias |
|---|---:|---:|
| Last year's visits | 0.362 | |
| BG/NBD expected visits | 0.391 | +4.9% |
| MBG/NBD expected visits | 0.393 | **+3.9%** |
| Return probability (logistic) | **0.400** | |

On real behaviour, MBG/NBD is close to calibrated in total. The euro bias comes from spend per visit and the modelled trade-up, not from visit prediction.

## Do the conclusions depend on the luxury-layer assumptions?

The whole pipeline was rerun under 9 calibrations: VIC revenue share 25%, 30% or 35%, and trade-up cap 6, 12 or 24 prior visits (`reports/sensitivity_client.csv`).

| Comparison | Range across all 9 variants | Holds in all 9? |
|---|---|---|
| Spend Spearman, model minus naive | +0.014 to +0.029 | Yes |
| Next year's VICs found, model minus naive | +1.0 to +1.9 pts | Yes |
| Upgrade avg. precision, model minus naive | +0.016 to +0.031 | Yes |
| Return AUC | 0.716 to 0.717 (recency 0.659) | Yes |
| Total spend bias | +40% to +47% | Consistently biased |

Relative conclusions are robust. Absolute levels are not: for example the share of next year's VICs found ranges from 40% to 54% depending on the trade-up cap, so absolute figures are reported as dependent on the calibration.

## What drives the models

Permutation importance on 60k test clients (`reports/client_feature_importance.csv`):

- **Return:** lifetime visits, product breadth over the last 12 months, and largest and average visit value. Loyal, broad shoppers come back; this is real Visuelle behaviour.
- **VIC upgrade:** product breadth, lifetime visits, largest visit value, and expected visits. Upgraders explore more of the collection before they spend more.
- **Spend:** dominated by lifetime visits. Part of this is the model learning the luxury layer's own trade-up rule, since purchase value depends on prior visits by design (gaps log G33).

## COVID stress test (2020)

Return rate fell from 37.5% to 10.6%. Every spend model over-predicted totals by 270–750% and ranking fell from about 0.40 to 0.23–0.25. The return model kept its edge over recency (AUC 0.721 vs 0.675). No model trained on normal years anticipates store closures; production use needs drift monitoring and a fallback rule.

## Output for later phases

`lux.client_value` scores every client as of 31 December 2019: value tier, last-12-month spend, predicted next-year spend (for ranking), probability of returning, MBG/NBD expected visits and probability active, VIC upgrade probability, and full-price, hero-piece and category breadth.

| Tier | Clients |
|---|---:|
| Top tier (top 0.1% by 12-month spend) | 619 |
| VIC (next 0.9%) | 5,571 |
| Core (next 9%) | 55,717 |
| Aspirational | 311,004 |
| Lapsed (no visit in 12 months) | 246,165 |
