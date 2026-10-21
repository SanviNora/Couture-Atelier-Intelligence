"""Client feature frames at a cut-off date, shared by experiments and the final pipeline.

A frame has one row per client with purchases up to the cut-off. Features use
only that history; the outcome columns (``visits_next``, ``spend_next``) cover
the following 365 days.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from couture_atelier.clv import fit_bgnbd, fit_gamma_gamma, fit_mbgnbd

WEEKS_PER_YEAR = 365 / 7
VIC_SHARE = 0.01

BASE_FEATURES = [
    "recency_weeks", "tenure_weeks_capped", "visits_12m", "spend_12m", "mean_visit_value_12m",
    "max_visit_value_12m", "last_visit_value", "products_12m", "stores_12m", "full_price_share_12m",
    "hero_share_12m", "lifetime_visits", "expected_visits_bg", "p_alive_bg",
]
MOMENTUM_FEATURES = ["visits_90d", "spend_90d", "spend_180d", "spend_momentum"]
EXTRA_FEATURES = [
    "expected_visits_mbg", "p_alive_mbg", "spend_per_visit_gg", "pred_spend_mbg",
    "categories_12m", "ticket_trend", "holiday_share", "mean_gap_weeks", "cutoff_month",
]
FEATURE_SETS = {
    "base": BASE_FEATURES,
    "base+momentum": BASE_FEATURES + MOMENTUM_FEATURES,
    "full": BASE_FEATURES + MOMENTUM_FEATURES + EXTRA_FEATURES,
}


def window_frame(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    frame = con.execute(
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
        ordered as (
            select *, row_number() over (partition by client_id order by day) as n,
                      row_number() over (partition by client_id order by day desc) as n_desc
            from hist
        ),
        first_last as (
            select client_id, min(day) as first_day, max(day) as last_day, count(*) as lifetime_visits,
                   arg_max(value_eur, day) as last_visit_value,
                   avg(value_eur) as mean_visit_value,
                   avg(value_eur) filter (where n_desc <= 3) as last3_mean,
                   avg(value_eur) filter (where n > 1) as monetary,
                   count(*) filter (where month(day) in (11, 12)) / count(*) as holiday_share
            from ordered group by 1
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
            select v.client_id, v.product_id, v.store_id, v.qty, p.hero_piece, p.category, w.on_discount
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
                   count(distinct category) as categories_12m,
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
               f.lifetime_visits, f.last_visit_value, f.holiday_share,
               f.last3_mean / f.mean_visit_value as ticket_trend,
               coalesce(f.monetary, 0) as monetary,
               coalesce(r.visits_12m, 0) as visits_12m, coalesce(r.spend_12m, 0) as spend_12m,
               coalesce(r.mean_visit_value_12m, 0) as mean_visit_value_12m,
               coalesce(r.max_visit_value_12m, 0) as max_visit_value_12m,
               coalesce(r.visits_90d, 0) as visits_90d, coalesce(r.spend_90d, 0) as spend_90d,
               coalesce(r.spend_180d, 0) as spend_180d,
               coalesce(r.spend_180d, 0) / greatest(coalesce(r.spend_12m, 0), 1) as spend_momentum,
               coalesce(l.products_12m, 0) as products_12m, coalesce(l.stores_12m, 0) as stores_12m,
               coalesce(l.categories_12m, 0) as categories_12m,
               coalesce(l.full_price_share_12m, 0) as full_price_share_12m,
               coalesce(l.hero_share_12m, 0) as hero_share_12m,
               coalesce(n.visits_next, 0) as visits_next, coalesce(n.spend_next, 0) as spend_next
        from first_last f
        left join recent r using (client_id)
        left join line_features l using (client_id)
        left join future n using (client_id)
        order by f.client_id
        """
    ).fetchdf()
    frame["mean_gap_weeks"] = np.where(
        frame["frequency"] > 0, frame["recency_span_weeks"] / frame["frequency"].clip(lower=1), frame["tenure_weeks"]
    )
    frame["cutoff_month"] = pd.Timestamp(cutoff).month
    frame["cutoff"] = cutoff
    return frame


def add_probabilistic(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit BG/NBD, MBG/NBD and Gamma-Gamma on history only (no outcomes are used)."""
    x = frame["frequency"].to_numpy(float)
    t_x = frame["recency_span_weeks"].to_numpy(float)
    T = frame["tenure_weeks"].to_numpy(float)
    monetary = frame["monetary"].to_numpy(float)
    bg, mbg, gg = fit_bgnbd(x, t_x, T), fit_mbgnbd(x, t_x, T), fit_gamma_gamma(x, monetary)
    frame = frame.copy()
    frame["p_alive_bg"] = bg.p_alive(x, t_x, T)
    frame["expected_visits_bg"] = bg.expected_visits(WEEKS_PER_YEAR, x, t_x, T)
    frame["p_alive_mbg"] = mbg.p_alive(x, t_x, T)
    frame["expected_visits_mbg"] = mbg.expected_visits(WEEKS_PER_YEAR, x, t_x, T)
    frame["spend_per_visit_gg"] = gg.expected_spend_per_visit(x, monetary)
    frame["pred_spend_bg"] = frame["expected_visits_bg"] * frame["spend_per_visit_gg"]
    frame["pred_spend_mbg"] = frame["expected_visits_mbg"] * frame["spend_per_visit_gg"]
    params = {name: {k: round(float(v), 4) for k, v in vars(model).items()}
              for name, model in (("bgnbd", bg), ("mbgnbd", mbg), ("gamma_gamma", gg))}
    return frame, params


def label_outcomes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["is_vic_now"] = frame["spend_12m"] >= frame["spend_12m"].quantile(1 - VIC_SHARE)
    frame["is_vic_next"] = frame["spend_next"] >= frame["spend_next"].quantile(1 - VIC_SHARE)
    frame["upgraded"] = ~frame["is_vic_now"] & frame["is_vic_next"]
    frame["lapsed_now"] = frame["visits_12m"] == 0
    frame["returned_next"] = frame["visits_next"] > 0
    return frame


def spend_metrics(actual: pd.Series, pred: pd.Series | np.ndarray) -> dict[str, float]:
    pred = pd.Series(np.asarray(pred, dtype=float), index=actual.index)
    k = max(1, int(len(pred) * VIC_SHARE))
    top_pred = pred.sort_values(ascending=False).index[:k]
    top_actual = actual.sort_values(ascending=False).index[:k]
    return {
        "total_bias_pct": round(float(pred.sum() / actual.sum() - 1) * 100, 1),
        "mae_eur": round(float((pred - actual).abs().mean()), 1),
        "spearman": round(float(spearmanr(pred, actual).statistic), 3),
        "top1pct_spend_capture": round(float(actual.loc[top_pred].sum() / actual.sum()), 3),
        "top1pct_precision": round(float(len(set(top_pred) & set(top_actual)) / k), 3),
    }
