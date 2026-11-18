# Allocation and Full-Price Stock

Where scarce new pieces go (which boutiques, then which clients), and what happens to stock that does not sell now that the EU bans destroying it.

```bash
python scripts/build_allocation.py              # store allocation, supply sweep, end of season, buy depth (10 seconds)
python -m scripts.build_client_allocation       # hero pieces to clients, with relationship rules (2 minutes)
```

Outputs: `reports/allocation.json`, `reports/allocation_policies.csv`, `reports/client_allocation.json`. Assumptions: `configs/allocation.yml`.

## 1. Store allocation of a scarce launch

For each of the 890 Jul–Nov 2019 launches (18,412 store-piece pairs), a scarce quantity equal to 60% of forecast season demand is split across the boutiques that carried it, with at most 12 units per boutique and one unit for every flagship. Policies are judged on true full-price demand: recovered weekly demand × 12 weeks, since a luxury house does not mark down.

The optimal rule models each boutique's season demand as a negative binomial: Poisson sales noise around a mean whose own uncertainty comes from the forecast's 10th–90th percentile spread. Expected sales from x units, E[min(D, x)], are concave in x, so giving each next unit to the boutique most likely to sell it is provably optimal (checked against brute force in the tests).

| Policy | Full-price sell-through | Demand served | Revenue | Units left unsold |
|---|---:|---:|---:|---:|
| Retailer's own opening allocation (rescaled) | 90.1% | 55.7% | €529.1M | 15,119 |
| Store size | 92.3% | 57.1% | €541.4M | 11,743 |
| Equal split | 95.2% | 58.8% | €558.7M | 7,447 |
| Optimal, ignoring forecast uncertainty | 95.4% | 59.0% | €560.3M | 7,046 |
| **Optimal, with forecast uncertainty** | 95.5% | 59.0% | €560.6M | 6,983 |
| Proportional to forecast | 95.6% | 59.1% | €561.3M | 6,808 |
| Oracle (true demand known) | 100.0% | 60.9% | €580.8M | 0 |

How the gap changes with supply (revenue):

| Policy | 60% of forecast | 80% | 100% | 120% |
|---|---:|---:|---:|---:|
| Retailer's opening allocation | €529.1M | €633.3M | €686.8M | €703.8M |
| Equal split | €558.7M | €652.2M | €692.8M | €704.9M |
| **Optimal, with uncertainty** | €560.6M | €654.3M | €694.2M | €705.2M |
| Oracle | €580.8M | €678.5M | €704.4M | €707.6M |

- **Using the forecast at all beats the retailer's own rule by €31.5M (+6%) at 60% supply** and halves unsold stock. The gap closes as supply grows (+€1.4M at 120%).
- **Among sensible rules the split matters little.** Even an equal split comes within €2M, and the optimal rule with forecast uncertainty does no better than a simple split in proportion to the forecast (which is €0.7M ahead, within noise): the optimum is only optimal for the model's own demand distribution. When every boutique wants more than it gets, almost any unit sells.
- **Hero pieces sell out under every rule** (97.9% sell-through): for them, the store split is not the lever. Which clients get them is.

The retailer's opening allocation is its initial shipment before restocks, so the comparison flatters the forecast-based rules somewhat (gaps log G58).

## 2. Which clients get a hero piece

For the 15 hero launches, each boutique's units are offered to its clients through a two-stage integer programme (`scipy.optimize.milp`):

- at most the boutique's units of each piece, and every unit offered if the rules allow it
- at most one hero piece per client per season
- consent to personal contact required; clients who bought the same piece twice or more are excluded as a reseller risk (22,802 clients)
- at least 30% of offers to **rising clients**: not yet VIC, spending in the top 10%, and accelerating

Each offer's value is the expected purchase (acceptance chance × price) plus a relationship value: a hero purchase is assumed to deepen the relationship by 5% of the client's annual spend. Acceptance chances combine category affinity and engagement; both are documented assumptions.

| Policy | Revenue now | Relationship value | Total | Units sold now | Offers to VICs | Offers to rising clients |
|---|---:|---:|---:|---:|---:|---:|
| Revenue only (greedy) | €22.7M | €3.4M | €26.1M | 6,743 | 0.3% | 1.4% |
| VICs first (greedy) | €20.3M | €23.2M | €43.5M | 6,328 | 15.0% | 1.2% |
| **Relationship value (optimised)** | €17.0M | €50.2M | €67.2M | 4,797 | 15.0% | 36.1% |
| Relationship value + 30% pipeline reserve | €17.0M | €50.1M | €67.1M | 4,798 | 14.9% | 36.7% |

- The optimiser beats "VICs first" by 54% in total value: it reaches high-value clients and the rising clients a VIC-first rule ignores.
- The 30% pipeline reserve costs 0.14%; the value-optimal answer already gives about 36% of offers to rising clients.
- Relationship-focused offers sell fewer pieces now; unsold hero units can go to a waitlist round.

**The assumption decides the trade-off**, so it is reported as a sensitivity (pipeline reserve kept):

| Relationship value per hero purchase | Revenue now | Units sold now | Offers to VICs | Offers to rising clients |
|---|---:|---:|---:|---:|
| 0% | €22.1M | 6,276 | 0.2% | 27.2% |
| 1% | €20.4M | 5,549 | 14.6% | 27.3% |
| 2% | €19.1M | 5,226 | 14.8% | 29.7% |
| 5% | €17.0M | 4,798 | 14.9% | 36.7% |
| 10% | €16.2M | 4,653 | 14.9% | 38.8% |

Serving every eligible VIC costs about 8% of near-term hero revenue (0% → 1%); beyond that, a higher weight mostly shifts offers to rising clients. That is the decision to put to a client-development lead.

## 3. End of season under the destruction ban

With the optimal store allocation, 6,983 units are left at the end of the season. Moving them to boutiques with unmet demand for the same piece absorbs 6,751 of them (€26.1M recovered after a 3% transfer cost); this assumes rebalancing knows where demand is left, so it is an upper bound. 232 units remain, with €0.23M of cost tied up. Their net value by route: carry over €0.10M, recycle €-0.02M, destroy (no longer allowed) €-0.02M.

## 4. How the ban changes how much to buy

The newsvendor answer: buy until the chance of selling the last unit equals the critical ratio, (price − cost) / (price − salvage). With a 70% gross margin (an assumption, in line with listed luxury houses; LVMH Fashion & Leather Goods reported a 35% operating margin in 2025), per launch:

| Regime | Value of an unsold unit | Critical ratio | Units bought vs mean demand | Expected unsold share |
|---|---:|---:|---:|---:|
| Before the ban: destroy (brand protection) | -2% | 0.69 | 1.15× | 22% |
| After the ban: carry over (with brand dilution cost) | +13% | 0.81 | 1.33× | 30% |
| After the ban: brand forbids carry-over, recycle only | -3% | 0.68 | 1.14× | 22% |
| Aspirational tier: certified pre-owned at 30% of price | +30% | 1.00* | 2.22× | 55% |

\* Leftover recovers its full cost, so overstocking is free; capped at a 99% service level.

- **The ban only cuts buy depth if the house refuses every non-destructive route.** If carry-over is acceptable (even after a 10% brand-dilution cost), an unsold unit is worth more than a destroyed one, and the optimal buy goes up, not down.
- **If brand policy forbids carrying old pieces**, recycling is the only route, and the buy is essentially what it was when destruction was allowed (1.14× vs 1.15× mean demand).
- **For the aspirational tier**, a certified pre-owned or outlet route that recovers cost makes overstocking nearly free, which is exactly the discount habit luxury houses avoid. That trade-off is the subject of the tier dial.
