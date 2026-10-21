# Demand: Lost Sales and Launch Forecasts

How much clients wanted each piece in each boutique, including the sales a store lost when it ran out, and a forecast of full-price demand for new launches that the allocation phase uses.

```bash
python scripts/build_demand.py            # recovery study, launch forecast (about 2 minutes)
python -m scripts.scarcity_scenario       # the luxury scarcity trap (about 1 minute)
```

Outputs: `reports/demand.json`, `reports/scarcity_scenario.json`, `marts.fct_stock_week`, `marts.fct_demand_week`, `lux.demand_forecast`.

## Rebuilding stock

Visuelle records weekly sales and restocks but no stock level. Stock is rebuilt per store, piece and week as deliveries logged before launch, plus restocks and returns, minus sales (`sql/marts/06_fct_stock_week.sql`). With the pre-launch delivery as opening stock, supply covers sales in 99.4% of store-weeks; the rest are flagged.

A week is **censored** when the store sold out during it, or had nothing to sell; recorded sales are then a floor, not demand.

## Finding: this retailer rarely runs out

| Measure | Value |
|---|---:|
| Store-weeks censored | 1.9% |
| Store-pieces that ever sell out | 7.2% |
| Typical stock vs weekly sales | 8 units on hand, about 1 sold a week (about 7 weeks of cover) |
| Censored weeks, best-selling tenth vs slowest tenth | 4.4% vs 7.1% |
| Hero pieces | 0.9% of weeks |

Fast-fashion stock is generous and best-sellers are restocked first, so lost sales are a small issue here (recovery adds 0.74% to demand). Luxury is the opposite: scarcity is deliberate and restocks are slow. The method is built and proven here, and its value is shown in the scarcity scenario below.

## Recovering demand

A Poisson gradient-boosting model of weekly demand is trained on weeks where stock lasted (features: week since launch, discount, price, category, last week's sales, the same piece's sales in other stores that week, the store's total sales that week, opening stock). For a sold-out week the recovered demand is the expected demand given that it reached the units sold:

E[D | D ≥ c] = μ · P(D ≥ c − 1) / P(D ≥ c), for D ~ Poisson(μ)

**Validation by artificial censoring.** On store-pieces that never sold out, a share of weeks is capped at a stock level set before the week (between 0 and twice the expected demand), and each method tries to recover the known demand:

| Weeks censored | Naive (sales = demand) | Max(sales, expected) | **Conditional** |
|---|---:|---:|---:|
| 10% | −64.8% | −13.9% | **−1.5%** |
| 30% | −64.6% | −13.6% | **−1.2%** |
| 50% | −64.9% | −13.3% | **−0.8%** |

Total bias of recovered demand in the censored weeks. The conditional method is close to unbiased; the simpler "max" rule under-recovers by about 13%.

## Launch forecast

For each store and new piece: the **full-price weekly demand rate** (mean recovered weekly demand over weeks without a discount, at least 4 such weeks), which is what a luxury house that does not discount needs for allocation.

Split by release date: train before 2019 (57,158 store-pieces), choose on H1 2019 (19,500), refit on both, and test once on Jul–Nov 2019 launches (18,412; these end before the 2020 lockdowns).

Features known at launch: category, colour, fabric, price, launch month, the piece's launch breadth relative to other launches that quarter, and the store's full-price rates for pieces launched 3–12 months earlier (overall and in the same category). Launch breadth is used as a within-quarter rank because it drifted from 33 stores per piece in 2017–18 to 41–44 in 2019; the raw count made every model over-forecast by 20–30%.

| Model | Validation WAPE | Test WAPE | Test bias |
|---|---:|---:|---:|
| Store × category average | 0.468 | 0.470 | +10.5% |
| **Gradient boosting, base features (chosen)** | 0.372 | **0.368** | +7.7% |
| + Silhouette image features | 0.371 | 0.371 | +9.6% |
| + Google Trends at launch | 0.379 | 0.373 | +8.8% |
| + both | 0.375 | 0.368 | +9.2% |
| + the retailer's opening allocation (not selectable) | 0.364 | 0.369 | +10.7% |

- **22% lower error than the store × category average.**
- **Image features and Google Trends add nothing measurable.** The simplest feature set within 0.005 WAPE of the best is used. Store history and launch breadth already carry most of the signal for a piece's store-level demand.
- The retailer's own opening allocation helps slightly on validation but is excluded from selection, so the forecast does not simply echo the merchant's existing plan.
- **Uncertainty:** the 10th–90th percentile interval covers 79.6% of test outcomes (nominal 80%).
- Test demand in H2 2019 ran slightly below earlier years, which explains most of the +7.7% bias.

## The luxury scarcity trap

Same real weekly demand, but each boutique gets 70% of a planner's estimate as opening stock and no restock, as for a scarce luxury piece. 38.7% of weeks then sell out and 38.6% of demand is lost. Two planners each train the same launch forecast on what they can see, then allocate the Jul–Nov 2019 launches from it:

| | Treats capped sales as demand | Recovers demand (one pass) | **Recovers demand with EM** |
|---|---:|---:|---:|
| Recovered demand in sold-out weeks vs truth | −80.9% | −46.6% | **−36.6%** |
| Forecast bias vs true demand | −21.8% | −9.3% | **−6.1%** |
| Forecast WAPE vs true demand | 0.390 | 0.362 | **0.359** |
| Units allocated to the next launches | 193,926 | 225,155 | 232,999 |
| Demand lost to stock-outs | 29.1% | 21.1% | **19.3%** |
| Stock left unsold | 8.5% | 12.4% | 13.4% |

Learning demand from capped sales shrinks the next allocation, which caps sales again: the error compounds season after season. Recovering demand first halves the forecast bias and cuts lost demand by 8 points, at the cost of more unsold stock. Deciding how much of that trade-off to accept, under the EU ban on destroying unsold goods, is the allocation phase's job.

**One pass vs EM.** A weekly model trained only on weeks where stock lasted learns from the lower-demand weeks. Expectation-maximisation fixes that selection: replace each sold-out week's target with its expected demand given the units sold, refit on all weeks, and repeat (`em_recover` in `src/couture_atelier/demand.py`). Mean recovered demand in sold-out weeks rises 0.82 → 0.90 → 0.94 → 0.96 → 0.98 over five iterations, against a true 1.54.

EM cannot close the whole gap here: under heavy scarcity the signals themselves (last week's sales, the same piece's sales in other stores) come from sold-out stock too, so they carry little information about true demand. Better uncapped signals, such as waitlists, client requests or store traffic, are what a luxury house would add.
