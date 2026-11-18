"""Cut, ladder or hold: three answers to luxury price fatigue.

In 2026 houses split three ways: cut list prices (Gucci), add entry pieces under
the icons while holding their prices (Chanel, Dior), or hold (Hermès). This
script judges the three on the project's data:

1. Ladder evidence (real Visuelle behaviour): do clients whose first piece was
   entry-priced come back and grow as much as clients who started higher?
   Compared within the same boutique and month, with boutique-level bootstrap
   intervals. Spend is measured at list prices, before the luxury trade-up.
2. Cut economics: the volume a permanent cut must add to break even, against
   the sustained response measured on real shallow markdowns.
3. Ladder economics: an entry line on the luxury policy's base, with
   cannibalisation as an explicit assumption.

Writes ``reports/price_strategy.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from couture_atelier.paths import CONFIGS, REPORTS, WAREHOUSE

TIER_ORDER = {"entry": 0, "core": 1, "high": 2, "exceptional": 3}


def cohort_frame(con: duckdb.DuckDBPyConnection, cfg: dict) -> pd.DataFrame:
    c = cfg["cohort"]
    frame = con.execute(
        f"""
        with first as (
            select client_id, receipt_id, store_id, purchased_at from lux.receipt where prior_visits = 0
              and cast(purchased_at as date) between date '{c["first_visit_from"]}' and date '{c["first_visit_to"]}'
        ),
        first_tier as (
            select f.client_id, max(case p.price_tier when 'entry' then 0 when 'core' then 1 when 'high' then 2
                                       else 3 end) as tier_rank
            from first f join lux.purchase_value v using (client_id, receipt_id) join lux.product p using (product_id)
            group by 1
        ),
        later as (
            select f.client_id, count(r.receipt_id) as visits_2y, coalesce(sum(r.base_value_eur), 0) as list_spend_2y
            from first f left join lux.receipt r
              on r.client_id = f.client_id and r.purchased_at > f.purchased_at
             and r.purchased_at <= f.purchased_at + interval {int(c["window_days"])} day
            group by 1
        )
        select f.client_id, f.store_id, date_trunc('month', f.purchased_at)::date as first_month,
               t.tier_rank, l.visits_2y, l.list_spend_2y
        from first f join first_tier t using (client_id) join later l using (client_id)
        order by f.client_id
        """
    ).fetchdf()
    frame["first_tier"] = frame["tier_rank"].map({v: k for k, v in TIER_ORDER.items()})
    frame["returned_2y"] = frame["visits_2y"] > 0
    frame["top_decile_2y"] = frame["list_spend_2y"] >= frame["list_spend_2y"].quantile(0.9)
    return frame


def within_store_month_gap(frame: pd.DataFrame, outcome: str, draws: int, seed: int) -> dict:
    """Entry-first minus higher-first, compared inside each boutique x month and weighted by entry clients."""
    frame = frame.assign(entry=frame["first_tier"].eq("entry"), y=frame[outcome].astype(float))
    cells = frame.groupby(["store_id", "first_month", "entry"])["y"].agg(["sum", "count"]).unstack("entry")
    cells = cells.dropna()  # boutique-months with both kinds of first purchase
    stores = cells.index.get_level_values("store_id").to_numpy()
    sums, counts = cells["sum"].to_numpy(), cells["count"].to_numpy()

    def estimate(rows: np.ndarray) -> float:
        rate_entry = sums[rows, 1] / counts[rows, 1]
        rate_other = sums[rows, 0] / counts[rows, 0]
        return float(np.average(rate_entry - rate_other, weights=counts[rows, 1]))

    point = estimate(np.arange(len(cells)))
    unique = np.unique(stores)
    by_store = {s: np.flatnonzero(stores == s) for s in unique}
    rng = np.random.default_rng(seed)
    boot = [estimate(np.concatenate([by_store[s] for s in rng.choice(unique, len(unique))])) for _ in range(draws)]
    low, high = np.percentile(boot, [2.5, 97.5])
    return {"difference": round(point, 4), "ci_low": round(float(low), 4), "ci_high": round(float(high), 4),
            "boutique_months": len(cells), "entry_clients_compared": int(counts[:, 1].sum())}


def cut_economics(cfg: dict, effects: dict, luxury: dict, leftover_value: float) -> dict:
    """Profit change from a permanent list-price cut on the luxury policy's sales.

    Extra units come from stock already bought, so each one only gives up its exit value:
    the assumption most favourable to cutting.
    """
    shallow = next(row for row in effects["markdown_lift"]["by_depth"] if row["depth"] == "up to 20%")
    depth = shallow["mean_depth"]
    elasticity = {"sustained": np.log(shallow["lift_weeks_3_4"]) / np.log(1 - depth),
                  "first_two_weeks": np.log(shallow["lift"]) / np.log(1 - depth)}
    revenue = luxury["revenue_eur_m"]
    max_volume = 1 / luxury["sell_through"]  # cannot sell more than the buy
    rows = []
    for cut in cfg["cut"]["depths"]:
        breakeven = (1 - leftover_value) / (1 - cut - leftover_value)
        row = {"cut": cut, "breakeven_volume_gain": round(breakeven - 1, 4)}
        for name, e in elasticity.items():
            volume = min((1 - cut) ** e, max_volume)
            change = revenue * (volume * (1 - cut) - 1) - revenue * (volume - 1) * leftover_value
            row[f"volume_gain_{name}"] = round(volume - 1, 4)
            row[f"profit_change_{name}_eur_m"] = round(float(change), 1)
        rows.append(row)
    return {"elasticity": {k: round(float(v), 2) for k, v in elasticity.items()},
            "measured_on": f"{shallow['events']} shallow markdowns, mean depth {depth:.1%}",
            "lift_first_two_weeks": shallow["lift"], "lift_weeks_3_4": shallow["lift_weeks_3_4"], "cuts": rows}


def ladder_economics(cfg: dict, con: duckdb.DuckDBPyConnection, cohort: pd.DataFrame, luxury: dict) -> dict:
    prices = con.execute(
        """
        select p.price_tier, sum(v.qty * v.list_price_eur) / sum(v.qty) as mean_price,
               sum(v.qty) filter (where r.prior_visits = 0) / sum(v.qty) as first_visit_share
        from lux.purchase_value v join lux.product p using (product_id) join lux.receipt r using (client_id, receipt_id)
        group by 1
        """
    ).fetchdf().set_index("price_tier")
    m = cfg["gross_margin"]
    entry_price, core_price = prices.loc["entry", "mean_price"], prices.loc["core", "mean_price"]
    new_share = prices.loc["entry", "first_visit_share"]
    future = cohort.loc[cohort["first_tier"].eq("entry"), "list_spend_2y"].mean()
    units = cfg["ladder"]["entry_units_share_of_sales"] * luxury["units_sold"]
    rows = []
    for k in cfg["ladder"]["cannibalisation"]:
        now = units * ((1 - k) * m * entry_price + k * m * (entry_price - core_price))
        later = units * (1 - k) * new_share * m * future
        rows.append({"cannibalisation": k, "profit_now_eur_m": round(now / 1e6, 1),
                     "profit_next_two_years_eur_m": round(later / 1e6, 1),
                     "total_eur_m": round((now + later) / 1e6, 1)})
    return {"entry_units": int(units), "entry_mean_price_eur": round(float(entry_price)),
            "core_mean_price_eur": round(float(core_price)), "entry_purchases_by_new_clients": round(float(new_share), 3),
            "entry_first_client_list_spend_2y_eur": round(float(future)), "scenarios": rows}


def build_price_strategy(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "price_strategy.yml").read_text())
    dial = yaml.safe_load((CONFIGS / "tier_dial.yml").read_text())
    effects = json.loads((REPORTS / "discount_effects.json").read_text())
    tier_dial = json.loads((REPORTS / "tier_dial.json").read_text())
    luxury = next(p for p in tier_dial["scenarios"]["central"]["policies"] if p["policy"] == "luxury")
    with duckdb.connect(str(warehouse), read_only=True) as con:
        cohort = cohort_frame(con, cfg)
        ladder = ladder_economics(cfg, con, cohort, luxury)
    by_tier = cohort.groupby("first_tier").agg(
        clients=("client_id", "size"), returned_2y=("returned_2y", "mean"), visits_2y=("visits_2y", "mean"),
        list_spend_2y_eur=("list_spend_2y", "mean"), top_decile_2y=("top_decile_2y", "mean"),
    ).reindex(list(TIER_ORDER)).dropna().round(4).reset_index()
    draws, seed = cfg["cohort"]["bootstrap_draws"], cfg["seed"]
    report = {
        "cohort": {"first_visits": f"{cfg['cohort']['first_visit_from']} to {cfg['cohort']['first_visit_to']}",
                   "window_days": cfg["cohort"]["window_days"], "clients": len(cohort),
                   "by_first_tier": by_tier.to_dict("records"),
                   "entry_minus_higher_within_boutique_month": {
                       outcome: within_store_month_gap(cohort, outcome, draws, seed)
                       for outcome in ("returned_2y", "top_decile_2y")}},
        "cut": cut_economics(cfg, effects, luxury, dial["tiers"]["luxury"]["leftover_value"]),
        "ladder": ladder,
        "hold": {"profit_change_eur_m": 0.0, "note": "the luxury policy as it stands; its risk is the client "
                 "base, not this season's margin"},
        "luxury_policy_base": {k: luxury[k] for k in ("revenue_eur_m", "gross_profit_eur_m", "units_sold",
                                                       "sell_through")},
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "price_strategy.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_price_strategy(), indent=2))
