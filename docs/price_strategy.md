# Cut, ladder or hold

`scripts/price_strategy.py` · `configs/price_strategy.yml` · `reports/price_strategy.json`

## The question

By late 2026, years of price rises had shrunk the luxury client base (about 400 million people in 2022, about 340 million in 2025, Bain–Altagamma), and 70% of luxury consumers say they have walked away from a purchase they felt was overpriced (BCG–Altagamma). Houses answered three ways:

- **Cut**: lower list prices on the core range (Gucci, autumn 2026; Louis Vuitton said cuts are on the table).
- **Ladder**: hold the icons and add pieces underneath them (Chanel bags under €5,000, Dior items under €1,000).
- **Hold**: keep prices and scarcity (Hermès, which grew 17% in the Americas in Q1 2026).

This module judges the three on the project's data: real client behaviour and price response from Visuelle 2.0, the luxury policy's economics from the Brand Tier Dial.

## 1. Cut: the volume a cut needs, and what real demand gives

A permanent cut of *c* on a range with a 70% gross margin only pays if volume rises enough to cover the margin given away on every unit that would have sold anyway. Extra units come from stock already bought, so each one only gives up its exit value (13% of list price); this is the assumption most favourable to cutting.

The volume response comes from 5,300 real shallow markdowns (mean depth 13.7%), measured by staggered difference-in-differences against stores selling the same piece without a markdown. In the first two weeks sales rise 1.57×, but by weeks three and four the lift is 1.07×: novelty and pulled-forward demand fade. A permanent cut is a permanent price, so the sustained response is the relevant one (elasticity about −0.5); the first-two-weeks response (about −3.1) is reported only as an upper bound.

| Cut | Volume needed to break even | Sustained response | Profit change (sustained) | Upper bound (first two weeks) |
|---|---|---|---|---|
| 5% | +6% | +3% | −€24M | +€72M |
| 10% | +13% | +5% | −€48M | +€146M |
| 15% | +21% | +8% | −€73M | +€91M (capped by stock) |

On the luxury policy's €806M of revenue, every cut loses money once the novelty fades.

## 2. Ladder: are entry-first clients worth acquiring?

209,058 real clients whose first visit fell in 2017, followed for two years (ending before COVID). Spend is measured at list prices, before the luxury trade-up, so it reflects real behaviour only.

| First piece | Clients | Came back within 2 years | Visits | List spend, 2 years | Top spend decile |
|---|---|---|---|---|---|
| Entry (≤ €1,300) | 35,037 | 61.5% | 2.33 | €12,842 | 8.8% |
| Core | 75,250 | 62.3% | 2.30 | €13,703 | 9.7% |
| High | 66,586 | 61.7% | 2.26 | €14,650 | 10.8% |
| Exceptional | 32,185 | 60.3% | 2.13 | €14,574 | 10.4% |

Compared only within the same boutique and month (763 boutique-months, boutique-level bootstrap), entry-first clients come back **1.1 points less often** [−1.8, −0.5] and reach the top spend decile 1.8 points less often. Entry pieces bring clients who are nearly, not quite, as good. Budget and intent are not observed, so this is not a causal claim that entry pieces create loyalty.

**Economics.** An entry line adding 5% to units sold (10,536 pieces at a mean €961, against €2,259 for core) earns little now but brings new clients: 37% of entry purchases are first visits, and an entry-first client spends €12,842 at list prices over the next two years.

| Share of entry purchases that would otherwise have been a core piece | Profit now | Next two years | Total |
|---|---|---|---|
| 0% | €7.1M | €35.3M | **€42.4M** |
| 25% | €2.9M | €26.5M | **€29.4M** |
| 50% | −€1.2M | €17.6M | **€16.4M** |

## 3. Hold

Holding is the baseline for this season's margin. Its risk is the client base: the return rate in this data fell every year (51% → 37.5%, see G75), and the Brand Tier Dial shows that the cost of luxury discipline is having nowhere to send unsold stock, not the markdowns themselves.

## Recommendation

Ladder, don't cut. A cut gives margin away on every piece that would have sold anyway and real demand does not come back strongly enough to pay for it; a ladder prices only the new pieces and brings clients who are nearly as loyal as those who start higher. Hold the icons.

## Limits

- The price response is measured on markdowns at a fast-fashion-priced retailer, not on luxury list-price changes; luxury demand is likely less price-sensitive, which makes cutting look even worse.
- Entry-line size and cannibalisation are assumptions, shown as a range.
- Brand effects of cutting (reference prices, resale values) are not modelled; they would add to the cost of a cut.
