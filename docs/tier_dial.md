# The Brand Tier Dial

The same demand, run under three brand policies (luxury, aspirational and premium), with every step priced: how much to buy, whether and when to discount, and where unsold stock goes. The question it answers is the one every brand moving up-market faces: *what does it cost, and what does it earn, to stop discounting?*

```bash
python scripts/discount_effects.py   # markdown lift and discount-acquired clients, on real data (1 minute)
python scripts/build_tier_dial.py    # the dial (seconds)
```

Outputs: `reports/discount_effects.json`, `reports/tier_dial.json`, `reports/market_pulse.csv`. Assumptions: `configs/tier_dial.yml`, `configs/allocation.yml`.

## What discounting does, measured on real data

### Markdown lift

The same piece starts its markdown in different weeks in different stores. For each piece and start week, stores that start discounting are compared with stores selling the same piece that have not started yet, over the same weeks (a staggered difference-in-differences; 10,448 markdown events on 3,859 pieces). A placebo moves the "start" two weeks earlier, where there should be no effect.

| Depth at start | Events | Lift, weeks 1–2 [95% CI] | Lift, weeks 3–4 | Placebo (should be ≈1) |
|---|---:|---:|---:|---:|
| up to 20% | 5,184 | **1.56×** [1.50, 1.64] | 1.08× | 0.97 [0.94, 1.00] |
| 20-35% | 4,105 | **3.31×** [3.09, 3.57] | 2.71× | 0.79 [0.74, 0.84] |
| 35-55% | 1,085 | **3.13×** [2.73, 3.64] | 1.86× | 0.76 [0.69, 0.82] |
| over 55% | 74 | **2.92×** [1.52, 7.38] | 1.27× | 0.23 [0.12, 0.39] |

- **Shallow markdowns pass the placebo cleanly** and lift sales 1.57× at first, but the effect almost disappears by weeks 3–4 (1.08×): mostly demand pulled forward.
- **Deeper markdowns fail the placebo** (0.76–0.79): the retailer discounts where sales are already slipping. The bias could run either way, so the dial is also run with the extra lift above the clean shallow estimate halved ("conservative").

### Do discounts attract weaker clients?

152,983 clients whose first purchase could be labelled; 32% started on a discount. Matched exactly on store, category, month and price tier (35,019 discount starters in 9,812 matched cells):

| | Raw | Matched |
|---|---:|---:|
| Return within 12 months, discount start minus full-price start | -1.8 points | +0.4 points |
| Spend in the next 12 months | | +12.9% |

**In this fast-fashion data, discount-acquired clients are not worse.** The luxury view that discounting cheapens the client base cannot be shown here, so the dial treats it as an explicit assumption: a brand cost per unit sold on discount of 0%, 5%, 10% or 20% of list price.

## The dial

For the 890 Jul–Nov 2019 launches (18,412 store-piece pairs). Weekly demand = true full-price demand × the lifecycle curve of never-discounted pieces (week 11 sells at 16% of week 0) × the markdown lift for the discount and weeks since it started. Each policy buys at the newsvendor service level set by its exit value for leftovers, on pooled demand across a piece's stores, then splits stock by forecast; sales are simulated week by week from stock. All tiers share a 70% gross margin, so only the policy differs.

Central lift:

| Policy | Units bought | Full-price share of sales | Sell-through | Revenue | Gross profit | Margin | Profit if each discounted unit costs 10% brand value | … 20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Luxury: full price, carry-over or recycle | 288,287 | 100% | 73% | €807M | **€515M** | 63.8% | €515M | €515M |
| Aspirational: one 20% event from week 9, outlet exit | 299,437 | 88% | 76% | €854M | **€591M** | 69.3% | €581M | €570M |
| Premium: markdowns optimised per piece, off-price exit | 290,143 | 79% | 82% | €862M | **€560M** | 64.9% | €540M | €520M |
| Retailer's actual markdown pattern (reference) | 290,143 | 75% | 80% | €837M | **€538M** | 64.2% | €515M | €493M |

Conservative lift:

| Policy | Units bought | Full-price share of sales | Sell-through | Revenue | Gross profit | Margin | … 10% brand cost | … 20% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Luxury: full price, carry-over or recycle | 288,287 | 100% | 73% | €807M | **€515M** | 63.8% | €515M | €515M |
| Aspirational: one 20% event from week 9, outlet exit | 299,437 | 89% | 75% | €842M | **€584M** | 69.4% | €575M | €566M |
| Premium: markdowns optimised per piece, off-price exit | 290,143 | 81% | 79% | €836M | **€539M** | 64.5% | €523M | €507M |
| Retailer's actual markdown pattern (reference) | 290,143 | 77% | 78% | €825M | **€528M** | 64.0% | €508M | €488M |

Premium's optimiser most often chose: week 8, 20% (231 pieces), week 6, 20% (212 pieces), week 8, 30% (205 pieces), week 10, 30% (84 pieces), week 4, 20% (80 pieces).

## Elevation: what a discounting brand gives up by moving up-market

Starting from the retailer's real markdown pattern, the brand keeps only its last N weeks of markdowns (outlet kept), then closes the outlet exit as well:

| Step | Full-price share | Revenue | Gross profit (central) | Margin | Gross profit (conservative) |
|---|---:|---:|---:|---:|---:|
| Markdowns in the last 12 weeks, outlet kept | 74% | €850M | €578.3M | 68.0% | €570.8M |
| Markdowns in the last 6 weeks, outlet kept | 86% | €842M | €578.3M | 68.7% | €573.7M |
| Markdowns in the last 4 weeks, outlet kept | 91% | €837M | €578.1M | 69.0% | €574.3M |
| Markdowns in the last 2 weeks, outlet kept | 96% | €830M | €577.3M | 69.5% | €575.2M |
| **No markdowns**, outlet kept | 100% | €823M | €576.2M | 70.0% | €574.8M |
| No markdowns, **no outlet** (luxury) | 100% | €810M | €517.6M | 63.9% | €517.0M |

- **Dropping every markdown costs almost nothing:** profit falls 0.4% under the central lift and rises under the conservative one, while margin climbs from 68% to 70%. With any brand cost of discounting, it pays. Because the lift fades, markdowns mostly move sales earlier rather than create them.
- **The real cost of luxury discipline is refusing an exit for unsold stock.** Closing the outlet route costs about €59M (10%).

How much the exit is worth (no markdowns, buy re-optimised for each value):

| Value recovered per unsold unit | Units bought | Gross profit | Margin |
|---:|---:|---:|---:|
| 13% | 288,287 | €518.6M | 63.9% |
| 20% | 295,977 | €540.8M | 66.2% |
| 25% | 299,437 | €559.7M | 68.0% |
| 30% | 299,437 | €575.1M | 70.0% |

Moving from carry-over (13% of price, after a brand-dilution cost) to a controlled outlet or certified pre-owned channel (30%) is worth about €56M. That is the number a luxury house weighs against the brand risk of a resale channel, and, since July 2026, the EU ban on destroying unsold goods makes some non-destructive exit unavoidable.

## Market context (Google Trends, US, last 12 months vs Oct 2021–Oct 2022)

| Signal | Change |
|---|---:|
| Coach Tabby | +755% (now about twice LV Speedy's search interest) |
| Louis Vuitton Speedy / Chanel Classic Flap / Hermès Birkin | +234% / +167% / +66% |
| Gucci (brand, US / worldwide) | -32% / -18% |
| Ralph Lauren / Calvin Klein / Tommy Hilfiger | +83% / +73% / -7% |
| "Luxury resale" | +449% |
| "Quiet luxury" (index level) | 0.5 → 13.6 |

Search interest is not sales, and these series cannot be joined to the 2016–2021 Visuelle data. They are context: aspirational icons and resale are where interest grew fastest, which is exactly the space where the dial's two levers — fewer markdowns, a controlled exit channel — matter most.
