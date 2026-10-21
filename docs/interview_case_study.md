# Interview-Ready Case Study

## Title

Couture Atelier Intelligence: connecting demand, inventory, and client relevance.

## Business problem

Luxury retail teams need to protect product desirability while improving availability and relationship quality. A dashboard that only reports sales is not enough: the decision must connect demand, stock, client value, churn risk, consent, and action.

## Approach

I built a reproducible portfolio simulation using public retail observations and a clearly labeled synthetic luxury layer. The pipeline validates source data, separates cancellations and accounting adjustments, creates analytical marts, builds a complete demand panel, and produces decision artifacts.

## Technical work

- Python, pandas, Parquet, DuckDB-compatible analytical structure
- Chronological forecasting validation
- Feature-based gradient boosting demand model
- As-of CLV features with future-window evaluation
- Time-based churn classification
- Deterministic behavioral segmentation
- Consent-aware clienteling recommendations
- Inventory transfer simulation with reconciliation checks
- Streamlit and Plotly executive dashboard

## Results

- Demand MAE improved from 6.88 to 5.79.
- Churn ROC-AUC reached 0.816 on a later time-based cohort.
- CLV correlation with realized next-90-day revenue reached 0.686.
- Constrained transfers and warehouse replenishment reduced projected stockouts by 0.49 percentage points.
- Clienteling generated 5,942 auditable decisions, including 1,176 no-action outcomes.

## Important judgment

The project does not claim to represent Louis Vuitton, LVMH, or private luxury data. Public datasets are used for methodology; luxury products, boutiques, inventory, consent, and appointments are synthetic. This distinction protects the analysis from false precision and demonstrates responsible data practice.

## Tradeoffs

The first models prioritize interpretability and reproducibility over maximal complexity. The next improvements are probability calibration, transfer cost and lead-time modeling, visual/category affinity, and observed clienteling response outcomes.

## Interview narrative

1. Start with the executive decision, not the model.
2. Explain how data provenance and temporal splits prevent misleading conclusions.
3. Show how the forecast informs inventory movement.
4. Show how CLV and churn change the clienteling priority.
5. Explain why no-action and consent are product features, not edge cases.
6. Close with the limitations and the next experiment.
