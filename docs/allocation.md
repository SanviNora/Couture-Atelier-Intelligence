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
| Retailer's own opening allocation (rescaled) | 90.2% | 55.8% | €530.0M | 15,114 |
| Store size | 92.4% | 57.2% | €542.3M | 11,698 |
| Equal split | 95.2% | 58.9% | €559.6M | 7,415 |
| Optimal, ignoring forecast uncertainty | 95.4% | 59.0% | €561.0M | 7,054 |
| **Optimal, with forecast uncertainty** | 95.5% | 59.1% | €561.6M | 6,943 |
| Proportional to forecast | 95.6% | 59.1% | €562.0M | 6,827 |
| Oracle (true demand known) | 100.0% | 61.0% | €580.9M | 0 |

How the gap changes with supply (revenue):

| Policy | 60% of forecast | 80% | 100% | 120% |
|---|---:|---:|---:|---:|
| Retailer's opening allocation | €530.0M | €634.1M | €687.1M | €704.4M |
| Equal split | €559.6M | €652.6M | €692.9M | €705.3M |
| **Optimal, with uncertainty** | €561.6M | €654.9M | €694.2M | €705.7M |
| Oracle | €580.9M | €679.5M | €704.6M | €707.9M |

- **Using the forecast at all beats the retailer's own rule by €31.6M (+6%) at 60% supply** and halves unsold stock. The gap closes as supply grows (+€1.3M at 120%).
- **Among sensible rules the split matters little.** Even an equal split comes within €2M, and the optimal rule with forecast uncertainty does no better than a simple split in proportion to the forecast (which is €0.4M ahead, within noise): the optimum is only optimal for the model's own demand distribution. When every boutique wants more than it gets, almost any unit sells.
- **Hero pieces sell out under every rule** (97.7% sell-through): for them, the store split is not the lever. Which clients get them is.

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
| Revenue only (greedy) | €22.8M | €3.4M | €26.2M | 6,768 | 0.4% | 1.4% |
| VICs first (greedy) | €20.4M | €23.2M | €43.6M | 6,353 | 14.9% | 1.2% |
| **Relationship value (optimised)** | €17.0M | €50.3M | €67.4M | 4,820 | 14.9% | 36.1% |
| Relationship value + 30% pipeline reserve | €17.0M | €50.2M | €67.3M | 4,821 | 14.8% | 36.7% |

- The optimiser beats "VICs first" by 55% in total value: it reaches high-value clients and the rising clients a VIC-first rule ignores.
- The 30% pipeline reserve costs 0.14%; the value-optimal answer already gives about 36% of offers to rising clients.
- Relationship-focused offers sell fewer pieces now; unsold hero units can go to a waitlist round.

**The assumption decides the trade-off**, so it is reported as a sensitivity (pipeline reserve kept):

| Relationship value per hero purchase | Revenue now | Units sold now | Offers to VICs | Offers to rising clients |
|---|---:|---:|---:|---:|
| 0% | €22.1M | 6,303 | 0.2% | 27.1% |
| 1% | €20.5M | 5,572 | 14.5% | 27.3% |
| 2% | €19.2M | 5,251 | 14.8% | 29.6% |
| 5% | €17.0M | 4,821 | 14.8% | 36.7% |
| 10% | €16.3M | 4,673 | 14.8% | 38.9% |

Serving every eligible VIC costs about 8% of near-term hero revenue (0% → 1%); beyond that, a higher weight mostly shifts offers to rising clients. That is the decision to put to a client-development lead.

## 3. End of season under the destruction ban

With the optimal store allocation, 6,943 units are left at the end of the season. Moving them to boutiques with unmet demand for the same piece absorbs 6,718 of them (€25.7M recovered after a 3% transfer cost); this assumes rebalancing knows where demand is left, so it is an upper bound. 225 units remain, with €0.27M of cost tied up. Their net value by route: carry over €0.12M, recycle €-0.03M, destroy (no longer allowed) €-0.02M.

## 4. How the ban changes how much to buy

The newsvendor answer: buy until the chance of selling the last unit equals the critical ratio, (price − cost) / (price − salvage). With a 70% gross margin (an assumption, in line with listed luxury houses; LVMH Fashion & Leather Goods reported a 35% operating margin in 2025), per launch:

| Regime | Value of an unsold unit | Critical ratio | Units bought vs mean demand | Expected unsold share |
|---|---:|---:|---:|---:|
| Before the ban: destroy (brand protection) | -2% | 0.69 | 1.15× | 23% |
| After the ban: carry over (with brand dilution cost) | +13% | 0.81 | 1.34× | 30% |
| After the ban: brand forbids carry-over, recycle only | -3% | 0.68 | 1.14× | 22% |
| Aspirational tier: certified pre-owned at 30% of price | +30% | 1.00* | 2.25× | 56% |

\* Leftover recovers its full cost, so overstocking is free; capped at a 99% service level.

- **The ban only cuts buy depth if the house refuses every non-destructive route.** If carry-over is acceptable (even after a 10% brand-dilution cost), an unsold unit is worth more than a destroyed one, and the optimal buy goes up, not down.
- **If brand policy forbids carrying old pieces**, recycling is the only route, and the buy is essentially what it was when destruction was allowed (1.14× vs 1.15× mean demand).
- **For the aspirational tier**, a certified pre-owned or outlet route that recovers cost makes overstocking nearly free, which is exactly the discount habit luxury houses avoid. That trade-off is the subject of the tier dial.
