"""The luxury scarcity trap: what happens when demand is learned from sales capped by tight stock.

Same real weekly demand as build_demand.py (recovered Visuelle sales), but each
boutique gets a tight opening allocation (70% of a planner's estimate) and no
restock, as for a scarce luxury piece. Sales stop when stock runs out.

A planner who treats those capped sales as demand is compared with one who
recovers demand first. Both train the same launch forecast and are judged on
true demand for the Jul-Nov 2019 launches; each then allocates the next
launches from its forecast, and the lost sales and leftover stock are counted.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scripts.build_demand import SPLITS, WEEK_FEATURES, launch_frame, recovery_study, weekly_frame

from couture_atelier.client_models import DEFAULT_PARAMS, gbm
from couture_atelier.demand import category_codes, em_recover, recover, wape
from couture_atelier.paths import REPORTS, WAREHOUSE

ALLOCATION_SHARE = 0.7  # opening stock as a share of the planner's estimate: deliberately tight
WEEKS = 12
LAUNCH_FEATURES = ["category_code", "color_code", "fabric_code", "price_index", "release_month",
                   "stores_launched_pct", "store_rate_hist", "store_category_rate_hist"]
PARAMS = {**DEFAULT_PARAMS, "min_samples_leaf": 100}


def simulate_scarcity(weeks: pd.DataFrame, stock: pd.Series) -> pd.DataFrame:
    """Sales under a fixed opening stock and no restock, from true weekly demand."""
    weeks = weeks.sort_values(["product_id", "store_id", "week_index"]).copy()
    weeks["true_demand"] = np.round(weeks["recovered_units"])
    weeks["allocation"] = weeks.set_index(["product_id", "store_id"]).index.map(stock).to_numpy()
    weeks = weeks[weeks["allocation"].notna()].copy()  # only store-products with a planner allocation
    cum = weeks.groupby(["product_id", "store_id"])["true_demand"].cumsum()
    sold_cum = np.minimum(cum, weeks["allocation"])
    weeks["units_sold"] = sold_cum - sold_cum.groupby([weeks["product_id"], weeks["store_id"]]).shift(1).fillna(0)
    weeks["censored"] = cum >= weeks["allocation"]
    weeks["opening_units"] = weeks["allocation"]
    # Signals a planner would see in a scarce world are built from capped sales too.
    weeks["lag_units"] = weeks.groupby(["product_id", "store_id"])["units_sold"].shift(1).fillna(0)
    weeks["chain_units_other"] = (weeks.groupby(["product_id", "week_index"])["units_sold"].transform("sum")
                                  - weeks["units_sold"])
    weeks["store_week_units"] = weeks.groupby(["store_id", "week_start"])["units_sold"].transform("sum")
    return weeks


def planner_stock(frame: pd.DataFrame, share: float) -> pd.Series:
    estimate = frame["store_category_rate_hist"].fillna(frame["store_rate_hist"]).fillna(frame["rate_recovered"].mean())
    units = np.maximum(1, np.round(share * estimate * WEEKS))
    return pd.Series(units.to_numpy(), index=pd.MultiIndex.from_frame(frame[["product_id", "store_id"]]))


def prepare(frame: pd.DataFrame, vocab: dict) -> pd.DataFrame:
    frame, _ = category_codes(frame, ["category", "color", "fabric"], vocab)
    frame["split"] = np.select(
        [frame["release_date"] < SPLITS["train_before"], frame["release_date"] < SPLITS["valid_before"],
         frame["release_date"] < SPLITS["test_before"]], ["train", "valid", "test"], default="excluded")
    return frame


def run(warehouse: Path = WAREHOUSE) -> dict:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        real_weeks, _ = recovery_study(weekly_frame(con))
        truth = launch_frame(con, real_weeks)
        vocab = {c: sorted(truth.loc[truth["release_date"] < SPLITS["train_before"], c].unique())
                 for c in ("category", "color", "fabric")}
        truth = prepare(truth, vocab)

        scarce = simulate_scarcity(real_weeks, planner_stock(truth, ALLOCATION_SHARE))
        uncensored = scarce[~scarce["censored"]]
        model = gbm("regressor", {**DEFAULT_PARAMS, "min_samples_leaf": 200}, "poisson")
        model.fit(uncensored[WEEK_FEATURES], uncensored["units_sold"])
        scarce["expected_units"] = model.predict(scarce[WEEK_FEATURES])
        recovered = scarce.assign(
            recovered_units=recover(scarce["units_sold"], scarce["censored"], scarce["expected_units"], "conditional"))
        em_units, em_path = em_recover(
            scarce, WEEK_FEATURES,
            lambda X, y: gbm("regressor", {**DEFAULT_PARAMS, "min_samples_leaf": 200}, "poisson").fit(X, y))
        recovered_em = scarce.assign(recovered_units=em_units)
        naive = scarce.assign(recovered_units=scarce["units_sold"])
        worlds = {"treats_sales_as_demand": prepare(launch_frame(con, naive), vocab),
                  "recovers_demand_first": prepare(launch_frame(con, recovered), vocab),
                  "recovers_demand_with_em": prepare(launch_frame(con, recovered_em), vocab)}
        censored = scarce["censored"].to_numpy()
        true_censored = scarce["true_demand"].to_numpy()[censored].sum()
        recovery_check = {
            name: round(float(units[censored].sum() / true_censored - 1) * 100, 1)
            for name, units in (("sales_as_demand", scarce["units_sold"].to_numpy()),
                                ("one_pass_recovery", recovered["recovered_units"].to_numpy()),
                                ("em_recovery", em_units))
        }

    key = ["product_id", "store_id"]
    true_rate = truth.set_index(key)["rate_recovered"]
    report = {"allocation_share": ALLOCATION_SHARE,
              "censored_week_share_under_scarcity": round(float(scarce["censored"].mean()), 3),
              "true_demand_lost_to_stockouts_pct": round(float(
                  1 - scarce["units_sold"].sum() / scarce["true_demand"].sum()) * 100, 1),
              "recovery_bias_in_sold_out_weeks_pct": recovery_check,
              "em_mean_recovered_demand_by_iteration": [round(v, 3) for v in em_path],
              "true_mean_demand_in_sold_out_weeks": round(float(scarce["true_demand"].to_numpy()[censored].mean()), 3),
              "planners": {}}
    test_truth = truth[truth["split"] == "test"]
    for name, world in worlds.items():
        fit_rows = world[world["split"].isin(["train", "valid"])]
        forecaster = gbm("regressor", PARAMS, "poisson").fit(fit_rows[LAUNCH_FEATURES], fit_rows["rate_recovered"])
        test_rows = world[world["split"] == "test"].set_index(key)
        common = test_rows.index.intersection(test_truth.set_index(key).index)
        features = test_rows.loc[common, LAUNCH_FEATURES]  # each planner only sees its own (capped or recovered) history
        forecast = pd.Series(forecaster.predict(features), index=common)
        actual = true_rate.loc[common]

        # Allocate the test launches from the forecast (12 weeks of forecast demand), then replay true demand.
        allocation = np.maximum(1, np.round(forecast * WEEKS))
        test_weeks = real_weeks.set_index(key).loc[common].reset_index()
        replay = simulate_scarcity(test_weeks, allocation)
        demand = replay["true_demand"].sum()
        stock_total = float(allocation.sum())
        report["planners"][name] = {
            "forecast_bias_vs_true_demand_pct": round(float(forecast.sum() / actual.sum() - 1) * 100, 1),
            "forecast_wape_vs_true_demand": round(wape(actual, forecast), 4),
            "next_allocation_units": round(stock_total),
            "demand_lost_to_stockouts_pct": round(float(1 - replay["units_sold"].sum() / demand) * 100, 1),
            "stock_left_unsold_pct": round(float(1 - replay["units_sold"].sum() / stock_total) * 100, 1),
        }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "scarcity_scenario.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
