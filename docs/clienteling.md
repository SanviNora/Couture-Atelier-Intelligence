# Recommendations and Clienteling

## Current decision layer

The clienteling engine creates one auditable decision per synthetic client. It combines:

- Explicit personalization consent
- Current product availability
- Historical product exclusion
- 90-day predicted CLV
- 90-day churn probability
- Catalog ranking by derived product value

Possible actions:

- `re_engagement`: consented clients with elevated churn probability.
- `private_appointment`: consented clients in the top CLV quartile.
- `product_suggestion`: consented clients with an eligible available product.
- `no_action`: no consent or no eligible product.

Every row includes a reason, consent status, eligibility flag, product ID, CLV, churn probability, recommendation date, and provenance.

## Current output

- 5,942 client decisions
- 4,766 consented clients
- 1,176 no-action decisions
- 2,184 product suggestions
- 1,553 re-engagement actions
- 1,029 private-appointment actions
- 100% product recommendation coverage in the current synthetic hero-product universe

These are derived simulation outputs, not actual clienteling conversions or commercial impact.

## Privacy and brand rules

- Non-consented clients never receive a clienteling action.
- The system can recommend no action.
- No inferred wealth, sensitive attributes, or identity data is used.
- Recommendations are decision support for an associate, not automatic outreach.
- Product availability is checked before recommending a product.

## Outputs

- `data/processed/clienteling_recommendations.parquet`
- `data/processed/clienteling_report.json`

## Next improvements

1. Add category affinity and visual similarity.
2. Add appointment and response outcomes.
3. Evaluate recommendation precision, coverage, diversity, and acceptance.
4. Add associate workload and contact-frequency limits.
5. Connect inventory transfers and clienteling priorities in one executive workspace.
