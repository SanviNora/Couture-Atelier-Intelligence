"""Do the client-model conclusions survive different luxury-layer assumptions?

Rebuilds the luxury layer under alternative calibration choices (VIC revenue
share target and trade-up cap), refits the selected client models with the
same settings, and re-tests on 2019. Each variant runs on a throwaway copy of
the warehouse; the main results are not touched.

Writes ``reports/sensitivity_client.csv``.
"""

from __future__ import annotations

import copy
import itertools
import json
import shutil
import time

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from scripts.build_luxury_layer import build_luxury_layer
from sklearn.metrics import average_precision_score, roc_auc_score

from couture_atelier.client_frames import (
    FEATURE_SETS,
    add_probabilistic,
    label_outcomes,
    spend_metrics,
    window_frame,
)
from couture_atelier.client_models import fit_return, fit_spend, gbm
from couture_atelier.luxury import load_calibration
from couture_atelier.paths import CONFIGS, DATA, REPORTS, WAREHOUSE

TEST = "2018-12-31"
VIC_TARGETS = [0.25, 0.30, 0.35]
CAPS = [6, 12, 24]


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def evaluate_variant(warehouse, selection: dict) -> dict:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        frames = {c: label_outcomes(add_probabilistic(window_frame(con, c))[0])
                  for c in [*selection["training_origins"], TEST]}
    train = pd.concat([frames[c] for c in selection["training_origins"]], ignore_index=True)
    test = frames[TEST]
    s_cfg, r_cfg, u_cfg = selection["spend"], selection["return"], selection["vic_upgrade"]
    spend = fit_spend(s_cfg["structure"], train, FEATURE_SETS[s_cfg["feature_set"]], s_cfg["params"])(test)
    p_return = fit_return(r_cfg["model"], train, FEATURE_SETS[r_cfg["feature_set"]], r_cfg["params"])(test)
    pool_train, pool_test = train[~train["is_vic_now"]], test[~test["is_vic_now"]]
    features = FEATURE_SETS[u_cfg["feature_set"]]
    upgrader = gbm("classifier", {**u_cfg["params"], "class_weight": u_cfg["class_weight"]})
    upgrade = upgrader.fit(pool_train[features], pool_train["upgraded"]).predict_proba(pool_test[features])[:, 1]

    actual = test["spend_next"]
    ml, naive = spend_metrics(actual, spend), spend_metrics(actual, test["spend_12m"])
    return {
        "spend_spearman_ml": ml["spearman"], "spend_spearman_naive": naive["spearman"],
        "top1_precision_ml": ml["top1pct_precision"], "top1_precision_naive": naive["top1pct_precision"],
        "total_bias_pct_ml": ml["total_bias_pct"],
        "return_auc_ml": round(float(roc_auc_score(test["returned_next"], p_return)), 4),
        "return_auc_recency": round(float(roc_auc_score(test["returned_next"], -test["recency_weeks"])), 4),
        "upgrade_ap_ml": round(float(average_precision_score(pool_test["upgraded"], upgrade)), 4),
        "upgrade_ap_naive": round(float(average_precision_score(pool_test["upgraded"], pool_test["spend_12m"])), 4),
        "visits_spearman_ml_return_prob": round(float(spearmanr(p_return, test["visits_next"]).statistic), 4),
    }


def run() -> pd.DataFrame:
    selection = yaml.safe_load((CONFIGS / "client_models.yml").read_text())
    base_cfg = load_calibration()
    work = DATA / "sensitivity"
    work.mkdir(exist_ok=True)
    rows = []
    for target, cap in itertools.product(VIC_TARGETS, CAPS):
        cfg = copy.deepcopy(base_cfg)
        cfg["targets"]["vic_revenue_share"]["value"] = target
        cfg["assumptions"]["trade_up"]["cap_prior_visits"] = cap
        warehouse = work / f"vic{int(target * 100)}_cap{cap}.duckdb"
        shutil.copy(WAREHOUSE, warehouse)
        try:
            layer = build_luxury_layer(warehouse, cfg, write_report=False)
            result = evaluate_variant(warehouse, selection)
        finally:
            warehouse.unlink(missing_ok=True)
        conc = layer["concentration"]
        row = {"vic_share_target": target, "trade_up_cap": cap, "trade_up_beta": conc["trade_up_beta"],
               "top_0_1pct_share": conc["cross_check_top_0_1pct_share"],
               "top_tier_annual_spend_eur": conc["cross_check_top_0_1pct_mean_annual_spend_eur"],
               "is_baseline": target == 0.30 and cap == 12, **result}
        rows.append(row)
        log(json.dumps(row))
        pd.DataFrame(rows).to_csv(REPORTS / "sensitivity_client.csv", index=False)
    shutil.rmtree(work, ignore_errors=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = run()
    with np.printoptions(linewidth=200):
        print(table.to_string(index=False))
