# Inventory, Transfers, and Stockouts

## Current decision layer

The inventory optimizer uses the final 14 days of synthetic store-product history to estimate daily demand, then targets 14 days of coverage plus two safety units.

For each product:

- Stores above target become potential donors.
- Stores below target become recipients.
- Transfers are matched deterministically by largest surplus and largest deficit.
- A donor cannot transfer more than its calculated surplus.
- A store cannot transfer to itself.
- Each donor is capped at 12 transfer units.
- Every store retains at least one display unit in its target stock.
- Residual deficits can be covered by a central warehouse up to 200 units.

## Current result

| Metric | Value |
|---|---:|
| Lookback window | 14 days |
| Projection horizon | 14 days |
| Safety stock | 2 units |
| Transfer recommendations | 13 |
| Transfer units | 59 |
| Replenishment recommendations | 14 |
| Replenishment units | 200 |
| Baseline projected stockout rate | 69.85% |
| Recommended projected stockout rate | 69.36% |
| Change | -0.49 percentage points |

The result is a derived simulation, not observed commercial impact. The warehouse capacity, minimum display stock, and transfer capacity are scenario assumptions and should be sensitivity-tested.

## Outputs

- `data/synthetic/transfer_recommendations.parquet`
- `data/synthetic/replenishment_recommendations.parquet`
- `data/synthetic/inventory_optimization_report.json`

## Remaining improvements

1. Use the demand model forecast instead of recent-average demand.
2. Add transfer costs and lead times.
3. Optimize service level and full-price availability together.
4. Run sensitivity analysis across safety-stock policies.
