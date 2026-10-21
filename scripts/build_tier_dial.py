"""The Brand Tier Dial: one demand, three brand policies, priced end to end.

For the Jul-Nov 2019 launches (true full-price demand per store and piece,
recovered from Visuelle), each policy decides how much to buy, whether and
when to discount, and where unsold stock goes:

- luxury: full price all season; leftover carried over or recycled
- aspirational: one end-of-season event at 20%; leftover to outlet / pre-owned
- premium: markdown start and depth optimised per piece on the forecast
- the retailer's actual markdown pattern (real Visuelle discounts), for reference

Weekly demand = true full-price rate x lifecycle curve (never-discounted
pieces) x markdown lift (staggered difference-in-differences,
reports/discount_effects.json). Sales are simulated week by week from stock.
Then an "elevation" sweep: a discounting brand cuts promotions step by step.

Writes ``reports/tier_dial.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import nbinom

from couture_atelier.allocation import critical_ratio, demand_distribution, proportional_allocation
from couture_atelier.paths import CONFIGS, REPORTS, WAREHOUSE

WEEKS = 12
EXIT_VALUES = (0.13, 0.20, 0.25, 0.30)  # carry-over (luxury default) up to outlet / certified pre-owned


def lifecycle_shape(con: duckdb.DuckDBPyConnection) -> np.ndarray:
    shape = con.execute(
        """
        with never as (select product_id, store_id from marts.fct_stock_week group by all having not bool_or(on_discount))
        select week_index, avg(units_sold) as units from marts.fct_stock_week join never using (product_id, store_id)
        where not censored group by 1 order by 1
        """
    ).fetchdf()["units"].to_numpy()
    return shape / shape.mean()


def lift_curve(effects: dict, scenario: str):
    """Markdown lift by depth and by weeks since the markdown started (weeks 1-2, then the faded level)."""
    bins = [b for b in effects["markdown_lift"]["by_depth"] if b["events"] >= 500]
    depths = np.array([0.0] + [b["mean_depth"] for b in bins])
    curves = {}
    for key, column in (("early", "lift"), ("late", "lift_weeks_3_4")):
        lifts = np.maximum.accumulate(np.array([1.0] + [b[column] for b in bins]))  # deeper never sells less
        if scenario == "conservative":
            shallow = lifts[1]
            lifts = np.where(lifts > shallow, shallow + 0.5 * (lifts - shallow), lifts)
        curves[key] = lifts

    def lift(depth: np.ndarray, age: np.ndarray) -> np.ndarray:
        early = np.interp(depth, depths, curves["early"])
        late = np.interp(depth, depths, curves["late"])
        return np.where(age < 2, early, late)

    return lift


def pooled_buy(data: pd.DataFrame, n: np.ndarray, p: np.ndarray, ratio: float) -> np.ndarray:
    """Buy per piece on pooled demand across its stores, then split by forecast (risk pooling)."""
    mean = nbinom.mean(n, p)
    var = nbinom.var(n, p)
    stock = np.zeros(len(data))
    for idx in data.groupby("product_id").indices.values():
        m, v = mean[idx].sum(), var[idx].sum()
        shape = m**2 / max(v - m, 1e-6)
        total = int(nbinom.ppf(ratio, shape, shape / (shape + m)))
        stock[idx] = proportional_allocation(total, mean[idx], np.full(len(idx), 10**6))
    return stock


def simulate(rate: np.ndarray, shape: np.ndarray, schedule: np.ndarray, stock: np.ndarray, lift, rng) -> dict:
    """Week-by-week sales from stock. ``schedule`` is the discount depth per pair and week."""
    left = stock.astype(float).copy()
    sold_full, sold_disc, revenue_share = np.zeros(len(rate)), np.zeros(len(rate)), np.zeros(len(rate))
    age = np.zeros(len(rate))
    for w in range(WEEKS):
        depth = schedule[:, w]
        continuing = schedule[:, w - 1] > 0 if w > 0 else np.zeros(len(rate), dtype=bool)
        age = np.where(depth > 0, np.where(continuing, age + 1, 0), 0)  # weeks since this markdown started
        demand = rng.poisson(rate * shape[w] * lift(depth, age))
        sold = np.minimum(demand, left)
        left -= sold
        on_sale = depth > 0
        sold_full += np.where(on_sale, 0, sold)
        sold_disc += np.where(on_sale, sold, 0)
        revenue_share += sold * (1 - depth)
    return {"sold_full": sold_full, "sold_disc": sold_disc, "revenue_units": revenue_share, "left": left}


def summarise(name: str, out: dict, stock: np.ndarray, price: np.ndarray, cost_share: float,
              leftover_value: float, brand_costs: list[float]) -> dict:
    revenue = float((out["revenue_units"] * price).sum())
    cogs = float((stock * price * cost_share).sum())
    leftover = float((out["left"] * price * leftover_value).sum())
    profit = revenue - cogs + leftover
    sold = out["sold_full"] + out["sold_disc"]
    row = {
        "policy": name,
        "units_bought": int(stock.sum()), "units_sold": int(sold.sum()),
        "full_price_share_of_units_sold": round(float(out["sold_full"].sum() / sold.sum()), 4),
        "sell_through": round(float(sold.sum() / stock.sum()), 4),
        "unsold_units": int(out["left"].sum()),
        "revenue_eur_m": round(revenue / 1e6, 1), "gross_profit_eur_m": round(profit / 1e6, 1),
        "profit_margin": round(profit / revenue, 4),
        "average_discount_on_units_sold": round(float(1 - out["revenue_units"].sum() / sold.sum()), 4),
    }
    for c in brand_costs:
        brand = float((out["sold_disc"] * price * c).sum())
        row[f"profit_after_brand_cost_{int(c * 100)}pct_eur_m"] = round((profit - brand) / 1e6, 1)
    return row


def build_tier_dial(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "tier_dial.yml").read_text())
    effects = json.loads((REPORTS / "discount_effects.json").read_text())
    with duckdb.connect(str(warehouse), read_only=True) as con:
        shape = lifecycle_shape(con)
        data = con.execute(
            """
            select f.product_id, f.store_id, f.rate_recovered, f.forecast_mean, f.forecast_q10, f.forecast_q90,
                   p.list_price_eur
            from lux.demand_forecast f join lux.product p using (product_id) order by 1, 2
            """
        ).fetchdf()
        actual = con.execute(
            """
            select product_id, store_id, list(discount order by week_index) as schedule
            from marts.fct_stock_week group by all
            """
        ).fetchdf()
    data = data.merge(actual, on=["product_id", "store_id"], how="left")
    rate, price = data["rate_recovered"].to_numpy(), data["list_price_eur"].to_numpy()
    n, p = demand_distribution(*(data[c].to_numpy() * WEEKS for c in ("forecast_mean", "forecast_q10", "forecast_q90")))
    cost_share = 1 - cfg["gross_margin"]
    brand_costs = cfg["brand_cost_per_discounted_unit"]

    def buy(tier: dict) -> np.ndarray:
        ratio = min(critical_ratio(1.0, cost_share, tier["leftover_value"]), tier["service_level_cap"])
        return pooled_buy(data, n, p, ratio)

    def event_schedule(start: int, depth: float, size: int) -> np.ndarray:
        schedule = np.zeros((size, WEEKS))
        schedule[:, start:] = depth
        return schedule

    results: dict[str, list] = {}
    for scenario in cfg["lift_scenarios"]:
        lift = lift_curve(effects, scenario)
        rng = np.random.default_rng(cfg["seed"])
        rows = []
        tiers = cfg["tiers"]

        lux_stock = buy(tiers["luxury"])
        out = simulate(rate, shape, np.zeros((len(data), WEEKS)), lux_stock, lift, rng)
        rows.append(summarise("luxury", out, lux_stock, price, cost_share, tiers["luxury"]["leftover_value"], brand_costs))

        asp = tiers["aspirational"]
        asp_stock = buy(asp)
        schedule = event_schedule(asp["promotion"]["start_week"], asp["promotion"]["depth"], len(data))
        out = simulate(rate, shape, schedule, asp_stock, lift, rng)
        rows.append(summarise("aspirational", out, asp_stock, price, cost_share, asp["leftover_value"], brand_costs))

        # Premium: choose each piece's markdown on the forecast (expected demand), then replay the truth.
        prem = tiers["premium"]
        prem_stock = buy(prem)
        forecast_rate = data["forecast_mean"].to_numpy()
        grid = [(s, d) for s in prem["markdown_grid"]["start_weeks"] for d in prem["markdown_grid"]["depths"]]
        grid.append((WEEKS, 0.0))  # no markdown
        expected_profit = []
        for start, depth in grid:
            plan = simulate(forecast_rate, shape, event_schedule(start, depth, len(data)), prem_stock, lift,
                            np.random.default_rng(1))
            pair_profit = (plan["revenue_units"] + plan["left"] * prem["leftover_value"]) * price
            expected_profit.append(pd.Series(pair_profit).groupby(data["product_id"]).sum())
        best = pd.concat(expected_profit, axis=1).to_numpy().argmax(axis=1)
        chosen = dict(zip(sorted(data["product_id"].unique()), [grid[i] for i in best], strict=True))
        schedule = np.zeros((len(data), WEEKS))
        for i, pid in enumerate(data["product_id"]):
            start, depth = chosen[pid]
            schedule[i, start:] = depth
        out = simulate(rate, shape, schedule, prem_stock, lift, rng)
        rows.append(summarise("premium_optimised_markdowns", out, prem_stock, price, cost_share,
                              prem["leftover_value"], brand_costs))

        actual_schedule = np.array([s if isinstance(s, (list, np.ndarray)) else [0.0] * WEEKS
                                    for s in data["schedule"]], dtype=float)
        out = simulate(rate, shape, actual_schedule, prem_stock, lift, rng)
        rows.append(summarise("retailer_actual_markdowns", out, prem_stock, price, cost_share,
                              prem["leftover_value"], brand_costs))

        # Elevation: a discounting brand keeps only the last N weeks of its markdowns, re-buying for each step.
        elevation = []
        # Elevation in two separate moves: first cut markdowns (outlet kept), then close the outlet exit.
        asp_buy = buy(asp)
        for keep_weeks in (12, 6, 4, 2, 0):
            trimmed = actual_schedule.copy()
            trimmed[:, : WEEKS - keep_weeks] = 0
            out = simulate(rate, shape, trimmed, asp_buy, lift, rng)
            summary = summarise(f"markdowns_in_last_{keep_weeks}_weeks_outlet_kept", out, asp_buy, price, cost_share,
                                asp["leftover_value"], brand_costs)
            summary["weeks_with_markdowns_kept"], summary["outlet_exit"] = keep_weeks, True
            elevation.append(summary)
        out = simulate(rate, shape, np.zeros((len(data), WEEKS)), lux_stock, lift, rng)
        summary = summarise("no_markdowns_no_outlet_luxury", out, lux_stock, price, cost_share,
                            tiers["luxury"]["leftover_value"], brand_costs)
        summary["weeks_with_markdowns_kept"], summary["outlet_exit"] = 0, False
        elevation.append(summary)
        # The exit route is the lever, so how much does its value matter? (no markdowns, re-buying for each value)
        exit_values = []
        for value in EXIT_VALUES:
            stock = buy({**asp, "leftover_value": value})
            out = simulate(rate, shape, np.zeros((len(data), WEEKS)), stock, lift, rng)
            summary = summarise(f"no_markdowns_exit_value_{int(value * 100)}pct", out, stock, price, cost_share,
                                value, brand_costs)
            summary["exit_value_share_of_price"] = value
            exit_values.append(summary)
        markdown_choices = pd.Series([f"week {s}, {int(d * 100)}%" if d else "no markdown"
                                      for s, d in chosen.values()]).value_counts().head(5).to_dict()
        results[scenario] = {"policies": rows, "elevation": elevation, "exit_value_sensitivity": exit_values,
                             "premium_markdown_choices": markdown_choices}

    report = {
        "launches": int(data["product_id"].nunique()), "store_launch_pairs": len(data),
        "lifecycle_shape": [round(float(v), 3) for v in shape],
        "lift_points": [(b["depth"], b["lift"]) for b in effects["markdown_lift"]["by_depth"]],
        "scenarios": results,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "tier_dial.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    out = build_tier_dial()
    for scenario, res in out["scenarios"].items():
        print(f"\n== {scenario}")
        print(pd.DataFrame(res["policies"]).to_string(index=False))
        print(pd.DataFrame(res["elevation"])[["policy", "units_bought", "sell_through", "revenue_eur_m",
                                               "gross_profit_eur_m", "profit_margin"]].to_string(index=False))
        print(res["premium_markdown_choices"])
