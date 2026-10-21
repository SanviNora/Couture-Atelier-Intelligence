"""VIC client intelligence: next-year value, tiers, VIC upgrade odds, lapse and win-back.

Three chronological windows. Features use only data up to each cut-off; the
following 12 months are the outcome.

- 2017-12-31 -> 2018: training window for the machine-learning challengers
- 2018-12-31 -> 2019: main test window
- 2019-12-31 -> 2020: COVID stress test, reported separately

Spend predictions compared: last-year spend (naive), BG/NBD x Gamma-Gamma,
and a gradient-boosting challenger trained on the earlier window.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

from couture_atelier.clv import fit_bgnbd, fit_gamma_gamma
from couture_atelier.luxury import client_tier, load_calibration
from couture_atelier.paths import REPORTS, WAREHOUSE

WINDOWS = {"train": "2017-12-31", "test": "2018-12-31", "covid": "2019-12-31"}
WEEKS_PER_YEAR = 365 / 7
SEED = 42
FEATURES = [
    "recency_weeks", "tenure_weeks_capped", "visits_12m", "spend_12m", "mean_visit_value_12m",
    "max_visit_value_12m", "last_visit_value", "products_12m", "stores_12m", "full_price_share_12m",
    "hero_share_12m", "lifetime_visits", "expected_visits_bgnbd", "p_alive",
    "visits_90d", "spend_90d", "spend_180d", "spend_momentum",
]
VIC_SHARE = 0.01
SPEND_SCALE = 1_000  # model spend in thousands of euros for a well-conditioned Poisson fit


def window_frame(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    """One row per client with history up to the cut-off and the next 12 months' outcome."""
    return con.execute(
        f"""
        with days as (
            select client_id, cast(purchased_at as date) as day, sum(value_eur) as value_eur
            from lux.receipt group by all
        ),
        hist as (select * from days where day <= date '{cutoff}'),
        future as (
            select client_id, count(*) as visits_next, sum(value_eur) as spend_next
            from days where day > date '{cutoff}' and day <= date '{cutoff}' + interval 365 day
            group by 1
        ),
        first_last as (
            select client_id, min(day) as first_day, max(day) as last_day, count(*) as lifetime_visits,
                   arg_max(value_eur, day) as last_visit_value
            from hist group by 1
        ),
        repeat_value as (
            select client_id, avg(value_eur) as monetary
            from (select *, row_number() over (partition by client_id order by day) as n from hist)
            where n > 1 group by 1
        ),
        recent as (
            select client_id, count(*) as visits_12m, sum(value_eur) as spend_12m,
                   avg(value_eur) as mean_visit_value_12m, max(value_eur) as max_visit_value_12m,
                   count(*) filter (where day > date '{cutoff}' - interval 90 day) as visits_90d,
                   coalesce(sum(value_eur) filter (where day > date '{cutoff}' - interval 90 day), 0) as spend_90d,
                   coalesce(sum(value_eur) filter (where day > date '{cutoff}' - interval 180 day), 0) as spend_180d
            from hist where day > date '{cutoff}' - interval 365 day group by 1
        ),
        lines as (
            select v.client_id, v.product_id, v.store_id, v.qty, p.hero_piece,
                   w.on_discount
            from lux.purchase_value v
            join lux.product p using (product_id)
            left join marts.fct_store_product_week w
              on w.product_id = v.product_id and w.store_id = v.store_id
             and w.week_start = cast(date_trunc('week', v.purchased_at) as date)
            where cast(v.purchased_at as date) <= date '{cutoff}'
              and cast(v.purchased_at as date) > date '{cutoff}' - interval 365 day
        ),
        line_features as (
            select client_id,
                   count(distinct product_id) as products_12m,
                   count(distinct store_id) as stores_12m,
                   sum(qty) filter (where on_discount = false) / sum(qty) as full_price_share_12m,
                   sum(qty) filter (where hero_piece) / sum(qty) as hero_share_12m
            from lines group by 1
        )
        select f.client_id,
               f.lifetime_visits - 1 as frequency,
               date_diff('day', f.first_day, f.last_day) / 7.0 as recency_span_weeks,
               date_diff('day', f.first_day, date '{cutoff}') / 7.0 as tenure_weeks,
               date_diff('day', f.last_day, date '{cutoff}') / 7.0 as recency_weeks,
               least(date_diff('day', f.first_day, date '{cutoff}') / 7.0, 52) as tenure_weeks_capped,
               f.lifetime_visits, f.last_visit_value,
               coalesce(m.monetary, 0) as monetary,
               coalesce(r.visits_12m, 0) as visits_12m, coalesce(r.spend_12m, 0) as spend_12m,
               coalesce(r.mean_visit_value_12m, 0) as mean_visit_value_12m,
               coalesce(r.max_visit_value_12m, 0) as max_visit_value_12m,
               coalesce(r.visits_90d, 0) as visits_90d, coalesce(r.spend_90d, 0) as spend_90d,
               coalesce(r.spend_180d, 0) as spend_180d,
               coalesce(r.spend_180d, 0) / greatest(coalesce(r.spend_12m, 0), 1) as spend_momentum,
               coalesce(l.products_12m, 0) as products_12m, coalesce(l.stores_12m, 0) as stores_12m,
               coalesce(l.full_price_share_12m, 0) as full_price_share_12m,
               coalesce(l.hero_share_12m, 0) as hero_share_12m,
               coalesce(n.visits_next, 0) as visits_next, coalesce(n.spend_next, 0) as spend_next
        from first_last f
        left join repeat_value m using (client_id)
        left join recent r using (client_id)
        left join line_features l using (client_id)
        left join future n using (client_id)
        """
    ).fetchdf()


def add_probabilistic(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = frame["frequency"].to_numpy(float)
    t_x = frame["recency_span_weeks"].to_numpy(float)
    T = frame["tenure_weeks"].to_numpy(float)
    bg = fit_bgnbd(x, t_x, T)
    gg = fit_gamma_gamma(x, frame["monetary"].to_numpy(float))
    frame = frame.copy()
    frame["p_alive"] = bg.p_alive(x, t_x, T)
    frame["expected_visits_bgnbd"] = bg.expected_visits(WEEKS_PER_YEAR, x, t_x, T)
    frame["pred_spend_probabilistic"] = frame["expected_visits_bgnbd"] * gg.expected_spend_per_visit(
        x, frame["monetary"].to_numpy(float)
    )
    params = {"bgnbd": {k: round(float(v), 4) for k, v in vars(bg).items()},
              "gamma_gamma": {k: round(float(v), 4) for k, v in vars(gg).items()}}
    return frame, params


def label_vic(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    now_cut = frame["spend_12m"].quantile(1 - VIC_SHARE)
    next_cut = frame["spend_next"].quantile(1 - VIC_SHARE)
    frame["is_vic_now"] = frame["spend_12m"] >= now_cut
    frame["is_vic_next"] = frame["spend_next"] >= next_cut
    frame["upgraded"] = ~frame["is_vic_now"] & frame["is_vic_next"]
    frame["lapsed_now"] = frame["visits_12m"] == 0
    frame["returned_next"] = frame["visits_next"] > 0
    return frame


def spend_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    order = pred.sort_values(ascending=False).index
    k = max(1, int(len(pred) * VIC_SHARE))
    top_pred = order[:k]
    top_actual = actual.sort_values(ascending=False).index[:k]
    return {
        "total_bias_pct": round(float(pred.sum() / actual.sum() - 1) * 100, 1),
        "mae_eur": round(float((pred - actual).abs().mean()), 1),
        "spearman": round(float(spearmanr(pred, actual).statistic), 3),
        "top1pct_spend_capture": round(float(actual.loc[top_pred].sum() / actual.sum()), 3),
        "top1pct_precision": round(float(len(set(top_pred) & set(top_actual)) / k), 3),
    }


def decile_table(actual: pd.Series, pred: pd.Series, model: str) -> pd.DataFrame:
    decile = pd.qcut(pred.rank(method="first"), 10, labels=range(1, 11))
    table = pd.DataFrame({"decile": decile, "pred": pred, "actual": actual}).groupby("decile", observed=True).agg(
        n=("pred", "size"), predicted_mean_eur=("pred", "mean"), actual_mean_eur=("actual", "mean")
    ).reset_index()
    table["model"] = model
    return table


def build_client_value(warehouse: Path = WAREHOUSE) -> dict[str, object]:
    cfg = load_calibration()["assumptions"]
    with duckdb.connect(str(warehouse)) as con:
        frames, params = {}, {}
        for name, cutoff in WINDOWS.items():
            frame, fitted = add_probabilistic(window_frame(con, cutoff))
            frames[name], params[name] = label_vic(frame), fitted

    train = frames["train"]
    regressor = HistGradientBoostingRegressor(loss="poisson", max_iter=300, learning_rate=0.06,
                                              l2_regularization=1.0, random_state=SEED)
    regressor.fit(train[FEATURES], train["spend_next"] / SPEND_SCALE)
    spend_ceiling = float(train["spend_next"].max())

    def predict_spend(frame: pd.DataFrame) -> np.ndarray:
        return np.clip(regressor.predict(frame[FEATURES]) * SPEND_SCALE, 0, spend_ceiling)

    returner = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                              random_state=SEED)
    returner.fit(train[FEATURES], train["returned_next"])
    upgrade_pool = train[~train["is_vic_now"]]
    classifier = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, l2_regularization=1.0,
                                                random_state=SEED)
    classifier.fit(upgrade_pool[FEATURES], upgrade_pool["upgraded"])

    report: dict[str, object] = {"windows": WINDOWS, "fitted_params": params, "evaluation": {}}
    deciles = []
    for name in ("test", "covid"):
        frame = frames[name]
        frame["pred_spend_naive"] = frame["spend_12m"]
        frame["pred_spend_ml"] = predict_spend(frame)
        actual = frame["spend_next"]
        spend = {model: spend_metrics(actual, frame[f"pred_spend_{model}"])
                 for model in ("naive", "probabilistic", "ml")}
        pool = frame[~frame["is_vic_now"]]
        upgrade_prob = classifier.predict_proba(pool[FEATURES])[:, 1]
        k = int(pool["upgraded"].sum())
        upgrade = {
            "candidates": len(pool),
            "upgraders": k,
            "base_rate": round(float(pool["upgraded"].mean()), 4),
            "auc_ml": round(float(roc_auc_score(pool["upgraded"], upgrade_prob)), 3),
            "auc_naive_spend": round(float(roc_auc_score(pool["upgraded"], pool["spend_12m"])), 3),
            "avg_precision_ml": round(float(average_precision_score(pool["upgraded"], upgrade_prob)), 3),
            "avg_precision_naive_spend": round(float(average_precision_score(pool["upgraded"], pool["spend_12m"])), 3),
            "precision_at_k_ml": round(float(pool["upgraded"].to_numpy()[np.argsort(-upgrade_prob)[:k]].mean()), 3),
        }
        frame["p_return_ml"] = returner.predict_proba(frame[FEATURES])[:, 1]
        lapsed = frame[frame["lapsed_now"]]
        returned, lapsed_returned = frame["returned_next"], lapsed["returned_next"]
        lapse = {
            "return_rate": round(float(returned.mean()), 4),
            "return_auc_ml": round(float(roc_auc_score(returned, frame["p_return_ml"])), 3),
            "return_auc_bgnbd_p_alive": round(float(roc_auc_score(returned, frame["p_alive"])), 3),
            "return_auc_recency_only": round(float(roc_auc_score(returned, -frame["recency_weeks"])), 3),
            "brier_ml": round(float(((frame["p_return_ml"] - returned) ** 2).mean()), 4),
            "lapsed_clients": len(lapsed),
            "lapsed_win_back_rate": round(float(lapsed_returned.mean()), 4),
            "lapsed_win_back_auc_ml": round(float(roc_auc_score(lapsed_returned, lapsed["p_return_ml"])), 3),
            "lapsed_win_back_auc_recency_only": round(float(roc_auc_score(lapsed_returned, -lapsed["recency_weeks"])), 3),
        }
        report["evaluation"][name] = {"clients": len(frame), "spend": spend, "vic_upgrade": upgrade, "lapse": lapse}
        if name == "test":
            deciles += [decile_table(actual, frame[f"pred_spend_{m}"], m) for m in ("naive", "probabilistic", "ml")]

    # Score every client as of the latest cut-off for the clienteling and dashboard phases.
    latest = frames["covid"]
    latest["pred_spend_next_eur"] = predict_spend(latest)
    latest["p_return_next"] = returner.predict_proba(latest[FEATURES])[:, 1]
    latest["value_tier"] = client_tier(latest["spend_12m"], cfg["client_tiers"])
    latest.loc[latest["lapsed_now"], "value_tier"] = "lapsed"
    pool = ~latest["is_vic_now"]
    latest["vic_upgrade_prob"] = np.nan
    latest.loc[pool, "vic_upgrade_prob"] = classifier.predict_proba(latest.loc[pool, FEATURES])[:, 1]
    latest["as_of_date"] = pd.Timestamp(WINDOWS["covid"])
    latest["data_provenance"] = "derived"
    columns = ["client_id", "as_of_date", "value_tier", "spend_12m", "visits_12m", "recency_weeks", "p_alive",
               "p_return_next", "expected_visits_bgnbd", "pred_spend_next_eur", "vic_upgrade_prob", "full_price_share_12m",
               "hero_share_12m", "data_provenance"]
    with duckdb.connect(str(warehouse)) as con:
        con.register("scored", latest[columns])
        con.execute("create or replace table lux.client_value as select * from scored")
    report["tiers_as_of"] = WINDOWS["covid"]
    report["tier_counts"] = latest["value_tier"].value_counts().to_dict()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "client_value.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    pd.concat(deciles).round(1).to_csv(REPORTS / "client_value_deciles.csv", index=False)
    return report


if __name__ == "__main__":
    print(json.dumps(build_client_value(), indent=2, default=str))
