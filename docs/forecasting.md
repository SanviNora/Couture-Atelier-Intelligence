# Demand Forecasting

## Current model

The first stronger model is a `HistGradientBoostingRegressor` trained on `log1p(gross_units)` using the complete product-country-date demand panel.

Features:

- Prior demand at 1, 7, 14, and 28 days
- Seven-day and 28-day rolling means using prior observations only
- Day of week, month, and day of month
- Product-country active age
- Days since the previous sale, using information available before the prediction date

The model uses a deterministic random seed and a maximum training sample of 750,000 rows. The holdout is the final 28 calendar days and is never used for training.

## Current benchmark

| Model | MAE | RMSE | Bias |
|---|---:|---:|---:|
| 7-day seasonal naive | 6.88 | 242.30 | 5.24 |
| Feature-based gradient boosting | 5.79 | 241.93 | 4.33 |

Metrics are in units and are calculated on the same 116,329-row chronological holdout. Results are derived from public observed transactions and should not be interpreted as causal business impact.

## Limitations

- The first model is a one-step, rolling-origin benchmark using known prior demand features.
- It does not yet model boutique inventory availability, price, promotions, product attributes, or external trends.
- It does not yet produce calibrated prediction intervals.
- The training sample cap improves runtime and reproducibility; later experiments should test sensitivity to the cap.
- The source is online giftware rather than luxury fashion, so the model is a methodological benchmark until the synthetic Couture Atelier layer is added.

## Next forecasting improvements

1. Add intermittent-demand benchmarks such as Croston-style methods.
2. Add product lifecycle, category, and price features from the fashion source.
3. Add boutique inventory and stockout censoring in the synthetic layer.
4. Compare hierarchical and probabilistic forecasts.
