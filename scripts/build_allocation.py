"""Allocate scarce new pieces across boutiques and price what happens to unsold stock.

For every Jul-Nov 2019 launch, a scarce quantity (60% of forecast season
demand) is split across the stores that carried it. Policies are judged on
true full-price demand (recovered weekly demand x 12 weeks):

- the retailer's own opening allocation, rescaled to the scarce quantity
- equal split, store size, forecast mean
- optimal: each next unit to the store most likely to sell it, using the
  forecast's uncertainty (negative binomial from the 10th-90th percentiles)
- the same optimal rule ignoring uncertainty
- an oracle that knows true demand

Then: what to do with stock left at the end of the season now that the EU
bans destroying it, and how the ban changes the optimal buy depth.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import nbinom

from couture_atelier.allocation import (
    critical_ratio,
    demand_distribution,
    expected_sales,
    greedy_allocation,
    leftover_value,
    proportional_allocation,
)
from couture_atelier.paths import CONFIGS, REPORTS, WAREHOUSE

HISTORY_BEFORE = "2019-07-01"
SUPPLY_SWEEP = (0.6, 0.8, 1.0, 1.2)
MAX_SERVICE_LEVEL = 0.99


def load(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        f"""
        with volume as (
            select store_id, sum(units_sold) as store_units
            from marts.fct_store_product_week where week_start < date '{HISTORY_BEFORE}' group by 1
        )
        select f.*, p.list_price_eur, p.hero_piece, s.store_format, coalesce(v.store_units, 0) as store_units
        from lux.demand_forecast f
        join lux.product p using (product_id)
        join lux.store s using (store_id)
        left join volume v using (store_id)
        order by f.product_id, f.store_id
        """
    ).fetchdf()


def allocate_piece(rows: pd.DataFrame, cfg: dict) -> dict[str, np.ndarray]:
    s = cfg["scarcity"]
    weeks = s["season_weeks"]
    mean, q10, q90 = (rows[c].to_numpy() * weeks for c in ("forecast_mean", "forecast_q10", "forecast_q90"))
    n, p = demand_distribution(mean, q10, q90)
    n_flat, p_flat = demand_distribution(mean, mean, mean)  # same mean, almost no uncertainty
    total = max(1, round(float(s["units_share_of_forecast"] * mean.sum())))
    cap = np.full(len(rows), s["max_units_per_store"])
    minimum = np.where(rows["store_format"].eq("flagship"), s["min_units_flagship"], 0)
    if minimum.sum() > total:
        minimum = np.zeros_like(minimum)
    truth = np.round(rows["rate_recovered"].to_numpy() * weeks).astype(int)
    oracle = np.minimum(truth, cap)
    if oracle.sum() > total:  # with known demand any fill of true demand up to the total is optimal
        order = np.argsort(-truth)
        oracle = np.zeros_like(truth)
        left = total
        for i in order:
            oracle[i] = min(truth[i], cap[i], left)
            left -= oracle[i]
    return {
        "retailer_opening_share": proportional_allocation(total, rows["opening_units"].to_numpy(), cap),
        "equal_split": proportional_allocation(total, np.ones(len(rows)), cap),
        "store_size": proportional_allocation(total, rows["store_units"].to_numpy(), cap),
        "forecast_mean_share": proportional_allocation(total, mean, cap),
        "optimal_ignoring_uncertainty": greedy_allocation(total, n_flat, p_flat, cap, minimum),
        "optimal_with_uncertainty": greedy_allocation(total, n, p, cap, minimum),
        "oracle_true_demand": oracle,
        "_truth": truth, "_n": n, "_p": p, "_total": total,
    }


def simulate(data: pd.DataFrame, cfg: dict) -> list[dict]:
    records = []
    for product_id, rows in data.groupby("product_id", sort=True):
        rows = rows.reset_index(drop=True)
        alloc = allocate_piece(rows, cfg)
        truth, price = alloc["_truth"], float(rows["list_price_eur"].iloc[0])
        for policy, units in alloc.items():
            if policy.startswith("_"):
                continue
            sold = np.minimum(truth, units)
            unmet = np.maximum(truth - units, 0).sum()
            left = int(units.sum() - sold.sum())
            records.append({
                "product_id": product_id, "policy": policy, "hero_piece": bool(rows["hero_piece"].iloc[0]),
                "price": price, "units": int(units.sum()), "demand": int(truth.sum()), "sold": int(sold.sum()),
                "leftover": left, "unmet": int(unmet),
                # Mid-season rebalancing: leftover moved to boutiques with unmet demand for the same piece.
                "transferable": int(min(left, unmet)),
            })
    return records


def build_allocation(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "allocation.yml").read_text())
    econ = cfg["economics"]
    with duckdb.connect(str(warehouse), read_only=True) as con:
        data = load(con)

    records = simulate(data, cfg)
    results = pd.DataFrame(records)

    # How much does the allocation rule matter as supply approaches demand?
    sweep = []
    for share in SUPPLY_SWEEP:
        swept = {**cfg, "scarcity": {**cfg["scarcity"], "units_share_of_forecast": share}}
        frame = pd.DataFrame(simulate(data, swept))
        revenue = (frame["sold"] * frame["price"]).groupby(frame["policy"]).sum()
        left = frame.groupby("policy")["leftover"].sum()
        for policy in ("retailer_opening_share", "equal_split", "forecast_mean_share", "optimal_with_uncertainty",
                       "oracle_true_demand"):
            sweep.append({"supply_share_of_forecast": share, "policy": policy,
                          "revenue_eur_m": round(float(revenue[policy]) / 1e6, 2), "leftover_units": int(left[policy])})


    def summary(frame: pd.DataFrame) -> pd.DataFrame:
        agg = frame.groupby("policy").agg(units=("units", "sum"), demand=("demand", "sum"), sold=("sold", "sum"),
                                          leftover=("leftover", "sum"), transferable=("transferable", "sum"))
        revenue = (frame["sold"] * frame["price"]).groupby(frame["policy"]).sum()
        agg["full_price_sell_through"] = (agg["sold"] / agg["units"]).round(4)
        agg["demand_served"] = (agg["sold"] / agg["demand"]).round(4)
        agg["revenue_eur_m"] = (revenue / 1e6).round(2)
        agg["leftover_cost_eur_m"] = ((frame["leftover"] * frame["price"] * (1 - econ["gross_margin"]["value"]))
                                      .groupby(frame["policy"]).sum() / 1e6).round(2)
        return agg.sort_values("revenue_eur_m")

    overall = summary(results)
    heroes = summary(results[results["hero_piece"]]) if results["hero_piece"].any() else None

    # ---- End of season under the destruction ban --------------------------------------
    best = results[results["policy"] == "optimal_with_uncertainty"]
    margin = econ["gross_margin"]["value"]
    transfer_cost = econ["leftover_routes"]["transfer"]["cost"]
    after_transfer = best["leftover"] - best["transferable"]
    price = best["price"]
    transfer_value = float((best["transferable"] * price * (1 - transfer_cost)).sum())
    route_values = {r: float((after_transfer * leftover_value(econ, price, price * (1 - margin), r)).sum())
                    for r in ("destroy", "carry_over", "recycle", "certified_pre_owned")}
    leftover = {
        "leftover_units": int(best["leftover"].sum()),
        "moved_to_boutiques_with_unmet_demand": int(best["transferable"].sum()),
        "value_recovered_by_transfers_eur_m": round(transfer_value / 1e6, 2),
        "still_unsold_after_transfers": int(after_transfer.sum()),
        "net_value_of_remaining_stock_by_route_eur_m": {k: round(v / 1e6, 2) for k, v in route_values.items()},
        "inventory_cost_tied_up_eur_m": round(float((after_transfer * price * (1 - margin)).sum()) / 1e6, 2),
    }

    # ---- Buy depth: how the ban changes the newsvendor answer ---------------------------
    price_unit, cost_unit = 1.0, 1 - margin
    regimes = {
        "before_ban_destroy": leftover_value(econ, price_unit, cost_unit, "destroy"),
        "after_ban_carry_over": leftover_value(econ, price_unit, cost_unit, "carry_over"),
        "after_ban_recycle_only": leftover_value(econ, price_unit, cost_unit, "recycle"),
        "aspirational_tier_certified_pre_owned": leftover_value(econ, price_unit, cost_unit, "certified_pre_owned"),
    }
    sample = data.groupby("product_id").agg(mean=("forecast_mean", "sum"), q10=("forecast_q10", "sum"),
                                            q90=("forecast_q90", "sum")) * cfg["scarcity"]["season_weeks"]
    n_all, p_all = demand_distribution(sample["mean"].to_numpy(), sample["q10"].to_numpy(), sample["q90"].to_numpy())
    buy_depth = {}
    for regime, salvage in regimes.items():
        ratio = critical_ratio(price_unit, cost_unit, salvage)
        unbounded = ratio >= MAX_SERVICE_LEVEL  # leftover recovers its cost: overstocking is free
        quantity = nbinom.ppf(min(ratio, MAX_SERVICE_LEVEL), n_all, p_all)
        sold = expected_sales(quantity.astype(int), n_all, p_all)
        buy_depth[regime] = {
            "salvage_value_share_of_price": round(salvage, 3),
            "critical_ratio": round(ratio, 3),
            "units_bought_vs_mean_demand": round(float(quantity.sum() / sample["mean"].sum()), 3),
            "expected_unsold_share": round(float(1 - sold.sum() / quantity.sum()), 3),
            "overstock_costless_capped_at_99pct": bool(unbounded),
        }

    report = {
        "launches": int(results["product_id"].nunique()),
        "store_launch_pairs": len(data),
        "scarce_units_share_of_forecast": cfg["scarcity"]["units_share_of_forecast"],
        "store_allocation": overall.reset_index().to_dict(orient="records"),
        "store_allocation_hero_pieces": heroes.reset_index().to_dict(orient="records") if heroes is not None else None,
        "supply_sweep": sweep,
        "end_of_season": leftover,
        "buy_depth_by_regime": buy_depth,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "allocation.json").write_text(json.dumps(report, indent=2, default=float) + "\n")
    overall.to_csv(REPORTS / "allocation_policies.csv")
    return report


if __name__ == "__main__":
    out = build_allocation()
    print(pd.DataFrame(out["store_allocation"]).to_string(index=False))
    print(json.dumps({k: out[k] for k in ("end_of_season", "buy_depth_by_regime")}, indent=2))
