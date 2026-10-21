# Advanced Client Intelligence

## Completed components

Advanced client intelligence now includes:

1. Improved as-of CLV baseline.
2. Time-based churn prediction.
3. Deterministic behavioral segmentation.

## CLV baseline

The first advanced client-intelligence artifact is an as-of customer value snapshot. Historical features are calculated only from purchases on or before the as-of date. Realized revenue is measured separately in the following 90-day evaluation window.

Features include:

- Purchase frequency and order count
- Historical net revenue
- Average order value with shrinkage prior
- Active customer age
- Recency
- Product breadth
- Return rate
- Purchase rate
- Recency-adjusted expected orders

The current model estimates 90-day value using a transparent purchase-rate and monetary-value formula. It is intentionally interpretable and serves as a benchmark before more advanced probabilistic CLV.

## Current result

- As-of date: September 10, 2011
- Evaluation horizon: 90 days
- Customers evaluated: 5,283
- MAE: 467.26 revenue units
- Prediction-to-realization correlation: 0.686
- Prediction provenance: derived from public observed transactions

The model underpredicts aggregate future revenue in this initial run. This is expected for a conservative shrinkage baseline and is a clear calibration target, not a result to hide.

## Outputs

- `data/processed/client_clv.parquet`
- `data/processed/client_clv_report.json`

## Churn prediction

Churn is defined as no eligible purchase during the 90 days after an as-of date. An earlier historical snapshot is used for training and a later chronological snapshot for evaluation.

Features include recency, order count, historical revenue, distinct products, purchase cadence, average order value, lines per order, and product breadth.

Current evaluation:

- Training as-of date: March 14, 2011
- Test as-of date: June 12, 2011
- Test customers: 4,978
- Test churn rate: 67.8%
- ROC-AUC: 0.816
- Average precision: 0.889
- Brier score: 0.173

These are predictive metrics, not causal retention impact. The next improvement is probability calibration across multiple rolling as-of dates.

## Behavioral segments

Five deterministic K-means segments are created from standardized recency, order count, historical revenue, product breadth, purchase cadence, average order value, lines per order, and distinct-product features. Segment profiles contain mean feature values and customer counts so clusters can be named by business stakeholders.

## Outputs

- `data/processed/client_clv.parquet`
- `data/processed/client_clv_report.json`
- `data/processed/churn_predictions.parquet`
- `data/processed/behavioral_segments.parquet`
- `data/processed/behavioral_segment_profiles.parquet`
- `data/processed/client_behavior_report.json`

## Remaining improvements

- Calibrate CLV and churn probabilities.
- Evaluate churn across multiple rolling windows.
- Add category, channel, consent, and boutique behavior after the clienteling layer exists.
- Test segment stability across time and cluster counts.
- Connect value, churn, consent, and behavior to next-best-action recommendations.
