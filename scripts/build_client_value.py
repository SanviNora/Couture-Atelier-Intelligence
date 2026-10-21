"""VIC client intelligence: next-year value, tiers, VIC upgrade odds, return and win-back.

Refits the setup chosen by ``experiment_client_models.py`` (``configs/client_models.yml``)
on every client at the four 2017 training cut-offs, then evaluates once:

- 2018-12-31 -> 2019: main test window
- 2019-12-31 -> 2020: COVID stress test, reported separately

Finally scores every client as of 2019-12-31 into ``lux.client_value``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from couture_atelier.client_frames import (
    FEATURE_SETS,
    add_probabilistic,
    label_outcomes,
    spend_metrics,
    window_frame,
)
from couture_atelier.client_models import fit_return, fit_spend, gbm
from couture_atelier.evaluation import paired_bootstrap
from couture_atelier.luxury import client_tier, load_calibration
from couture_atelier.paths import CONFIGS, PROCESSED, REPORTS, WAREHOUSE

TEST, COVID = "2018-12-31", "2019-12-31"
CACHE = PROCESSED / "client_frames"


def load_frame(con: duckdb.DuckDBPyConnection, cutoff: str) -> tuple[pd.DataFrame, dict]:
    frame, params = add_probabilistic(window_frame(con, cutoff))
    frame = label_outcomes(frame)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(CACHE / f"{cutoff}.parquet", index=False)
    return frame, params


def decile_table(actual: pd.Series, pred: np.ndarray, model: str) -> pd.DataFrame:
    pred = pd.Series(pred, index=actual.index)
    decile = pd.qcut(pred.rank(method="first"), 10, labels=range(1, 11))
    table = pd.DataFrame({"decile": decile, "pred": pred, "actual": actual}).groupby("decile", observed=True).agg(
        n=("pred", "size"), predicted_mean_eur=("pred", "mean"), actual_mean_eur=("actual", "mean")
    ).reset_index()
    table["model"] = model
    return table


def spearman(y: np.ndarray, score: np.ndarray) -> float:
    return float(spearmanr(score, y).statistic)


def top1_precision(y: np.ndarray, score: np.ndarray) -> float:
    k = max(1, int(len(y) * 0.01))
    return len(set(np.argsort(-score)[:k]) & set(np.argsort(-y)[:k])) / k


def precision_at_k(y: np.ndarray, score: np.ndarray) -> float:
    k = int(y.sum())
    return float(y[np.argsort(-score)[:k]].mean())


def build_client_value(warehouse: Path = WAREHOUSE) -> dict[str, object]:
    selection = yaml.safe_load((CONFIGS / "client_models.yml").read_text())
    tiers = load_calibration()["assumptions"]["client_tiers"]
    with duckdb.connect(str(warehouse), read_only=True) as con:
        loaded = {cutoff: load_frame(con, cutoff) for cutoff in [*selection["training_origins"], TEST, COVID]}
    frames = {cutoff: frame for cutoff, (frame, _) in loaded.items()}
    fitted_params = {cutoff: params for cutoff, (_, params) in loaded.items() if cutoff in (TEST, COVID)}
    train = pd.concat([frames[c] for c in selection["training_origins"]], ignore_index=True)

    spend_cfg, return_cfg, upgrade_cfg = selection["spend"], selection["return"], selection["vic_upgrade"]
    spend_features = FEATURE_SETS[spend_cfg["feature_set"]]
    predict_spend = fit_spend(spend_cfg["structure"], train, spend_features, spend_cfg["params"])
    recalibration = spend_cfg["validation_recalibration_factor"]
    return_features = FEATURE_SETS[return_cfg["feature_set"]]
    predict_return = fit_return(return_cfg["model"], train, return_features, return_cfg["params"])
    upgrade_features = FEATURE_SETS[upgrade_cfg["feature_set"]]
    pool_train = train[~train["is_vic_now"]]
    upgrader = gbm("classifier", {**upgrade_cfg["params"], "class_weight": upgrade_cfg["class_weight"]}).fit(
        pool_train[upgrade_features], pool_train["upgraded"]
    )

    report: dict[str, object] = {"selection": selection, "fitted_params": fitted_params, "evaluation": {}}
    deciles = []
    for name, cutoff in (("test", TEST), ("covid", COVID)):
        frame = frames[cutoff]
        actual = frame["spend_next"]
        ml = predict_spend(frame)
        preds = {
            "naive_last_12m": frame["spend_12m"].to_numpy(),
            "bgnbd_gg": frame["pred_spend_bg"].to_numpy(),
            "mbgnbd_gg": frame["pred_spend_mbg"].to_numpy(),
            "selected_ml": ml,
            "selected_ml_recalibrated": ml * recalibration,
        }
        spend = {model: spend_metrics(actual, pred) for model, pred in preds.items()}

        pool = frame[~frame["is_vic_now"]]
        upgraded = pool["upgraded"].to_numpy()
        upgrade_prob = upgrader.predict_proba(pool[upgrade_features])[:, 1]
        naive_upgrade = pool["spend_12m"].to_numpy()
        upgrade = {
            "candidates": len(pool), "upgraders": int(upgraded.sum()), "base_rate": round(float(upgraded.mean()), 4),
            "auc_ml": round(float(roc_auc_score(upgraded, upgrade_prob)), 3),
            "auc_naive_spend": round(float(roc_auc_score(upgraded, naive_upgrade)), 3),
            "avg_precision_ml": round(float(average_precision_score(upgraded, upgrade_prob)), 3),
            "avg_precision_naive_spend": round(float(average_precision_score(upgraded, naive_upgrade)), 3),
            "precision_at_k_ml": round(precision_at_k(upgraded, upgrade_prob), 3),
            "precision_at_k_naive_spend": round(precision_at_k(upgraded, naive_upgrade), 3),
        }

        p_return = predict_return(frame)
        returned = frame["returned_next"]
        lapsed = frame["lapsed_now"].to_numpy()
        lapse = {
            "return_rate": round(float(returned.mean()), 4),
            "return_auc_ml": round(float(roc_auc_score(returned, p_return)), 3),
            "return_auc_mbgnbd_p_alive": round(float(roc_auc_score(returned, frame["p_alive_mbg"])), 3),
            "return_auc_bgnbd_p_alive": round(float(roc_auc_score(returned, frame["p_alive_bg"])), 3),
            "return_auc_recency_only": round(float(roc_auc_score(returned, -frame["recency_weeks"])), 3),
            "brier_ml": round(float(brier_score_loss(returned, p_return)), 4),
            "lapsed_clients": int(lapsed.sum()),
            "lapsed_win_back_rate": round(float(returned[lapsed].mean()), 4),
            "lapsed_win_back_auc_ml": round(float(roc_auc_score(returned[lapsed], p_return[lapsed])), 3),
            "lapsed_win_back_auc_recency_only": round(
                float(roc_auc_score(returned[lapsed], -frame.loc[lapsed, "recency_weeks"])), 3),
        }
        # Visits come straight from Visuelle, with no modelled euros: a check on real behaviour only.
        visits_next = frame["visits_next"].to_numpy()
        real_behaviour = {
            f"visits_spearman_{name}": round(spearman(visits_next, score), 3)
            for name, score in (("naive_visits_12m", frame["visits_12m"].to_numpy()),
                                ("bgnbd_expected", frame["expected_visits_bg"].to_numpy()),
                                ("mbgnbd_expected", frame["expected_visits_mbg"].to_numpy()),
                                ("ml_return_probability", p_return))
        }
        real_behaviour["visits_total_bias_pct_mbgnbd"] = round(
            float(frame["expected_visits_mbg"].sum() / visits_next.sum() - 1) * 100, 1)
        real_behaviour["visits_total_bias_pct_bgnbd"] = round(
            float(frame["expected_visits_bg"].sum() / visits_next.sum() - 1) * 100, 1)
        y_spend, y_return = actual.to_numpy(), returned.to_numpy()
        uncertainty = {
            "spend_spearman_ml_minus_naive": paired_bootstrap(spearman, y_spend, ml, preds["naive_last_12m"]),
            "top1pct_precision_ml_minus_naive": paired_bootstrap(top1_precision, y_spend, ml, preds["naive_last_12m"]),
            "return_auc_ml_minus_recency": paired_bootstrap(
                roc_auc_score, y_return, p_return, -frame["recency_weeks"].to_numpy()),
            "upgrade_avg_precision_ml_minus_naive": paired_bootstrap(
                average_precision_score, upgraded, upgrade_prob, naive_upgrade),
        }
        report["evaluation"][name] = {"clients": len(frame), "spend": spend, "vic_upgrade": upgrade, "lapse": lapse,
                                      "real_behaviour_visits": real_behaviour, "paired_bootstrap_95ci": uncertainty}
        if name == "test":
            deciles += [decile_table(actual, preds[m], m) for m in ("naive_last_12m", "mbgnbd_gg", "selected_ml")]

    # Score every client as of the latest cut-off for clienteling and the dashboard.
    latest = frames[COVID].copy()
    latest["pred_spend_next_eur"] = predict_spend(latest) * recalibration
    latest["p_return_next"] = predict_return(latest)
    latest["value_tier"] = client_tier(latest["spend_12m"], tiers)
    latest.loc[latest["lapsed_now"], "value_tier"] = "lapsed"
    pool = ~latest["is_vic_now"]
    latest["vic_upgrade_prob"] = np.nan
    latest.loc[pool, "vic_upgrade_prob"] = upgrader.predict_proba(latest.loc[pool, upgrade_features])[:, 1]
    latest["as_of_date"] = pd.Timestamp(COVID)
    latest["data_provenance"] = "derived"
    columns = ["client_id", "as_of_date", "value_tier", "spend_12m", "visits_12m", "recency_weeks",
               "p_alive_mbg", "p_return_next", "expected_visits_mbg", "pred_spend_next_eur", "vic_upgrade_prob",
               "full_price_share_12m", "hero_share_12m", "categories_12m", "data_provenance"]
    with duckdb.connect(str(warehouse)) as con:
        con.register("scored", latest[columns])
        con.execute("create or replace table lux.client_value as select * from scored")
    report["tiers_as_of"] = COVID
    report["tier_counts"] = latest["value_tier"].value_counts().to_dict()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "client_value.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    pd.concat(deciles).round(1).to_csv(REPORTS / "client_value_deciles.csv", index=False)
    return report


if __name__ == "__main__":
    result = build_client_value()
    print(json.dumps(result["evaluation"], indent=2))
