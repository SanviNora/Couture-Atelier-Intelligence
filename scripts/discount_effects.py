"""What discounting does, measured on real Visuelle data.

A. Markdown lift. The same piece starts its markdown in different weeks in
   different stores. For each piece and start week, stores that start
   discounting are compared with stores selling the same piece that have not
   started yet (or never do), over the same weeks: a staggered
   difference-in-differences. Lift = actual sales in the two weeks after the
   start / the sales expected from the comparison stores' trend. A placebo on
   the two weeks before the start checks that treated and comparison stores
   were trending alike. 95% intervals by bootstrapping pieces.

B. Do discounts attract weaker clients? Clients whose first purchase was on
   discount are compared with clients whose first purchase was at full price,
   matched exactly on store, category, month and price tier, over the next
   12 months: return rate and spend.

Writes ``reports/discount_effects.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from couture_atelier.paths import REPORTS, WAREHOUSE

DEPTH_BINS = [0.0, 0.2, 0.35, 0.55, 1.0]
DEPTH_LABELS = ["up to 20%", "20-35%", "35-55%", "over 55%"]
BOOTSTRAPS = 300
FIRST_PURCHASE_BEFORE = "2019-03-01"  # 12-month follow-up ends before the 2020 lockdowns


def markdown_events(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    weeks = con.execute(
        "select product_id, store_id, week_index, units_sold, discount, on_discount from marts.fct_stock_week"
    ).fetchdf()
    start = weeks[weeks["on_discount"]].groupby(["product_id", "store_id"])["week_index"].min().rename("start")
    weeks = weeks.merge(start, on=["product_id", "store_id"], how="left")
    weeks["start"] = weeks["start"].fillna(99).astype(int)
    rows = []
    for (product_id, k), treated in weeks[weeks["start"].between(4, 9)].groupby(["product_id", "start"]):
        same_piece = weeks[(weeks["product_id"] == product_id) & (weeks["start"] > k + 1)]
        if same_piece.empty:
            continue
        # Comparison stores still at full price four weeks on, to measure whether the lift fades.
        late_piece = weeks[(weeks["product_id"] == product_id) & (weeks["start"] > k + 3)]
        depth = treated.loc[treated["week_index"] == k, "discount"].mean()

        def window(frame: pd.DataFrame, lo: int, hi: int) -> float:
            return float(frame.loc[frame["week_index"].between(lo, hi), "units_sold"].sum())

        rows.append({
            "product_id": product_id, "start": k, "depth": depth, "treated_stores": treated["store_id"].nunique(),
            "t_pre": window(treated, k - 2, k - 1), "t_post": window(treated, k, k + 1),
            "t_pre2": window(treated, k - 4, k - 3),
            "c_pre": window(same_piece, k - 2, k - 1), "c_post": window(same_piece, k, k + 1),
            "c_pre2": window(same_piece, k - 4, k - 3),
            "t_late": window(treated, k + 2, k + 3) if k <= 8 else np.nan,
            "l_pre": window(late_piece, k - 2, k - 1) if not late_piece.empty else np.nan,
            "l_late": window(late_piece, k + 2, k + 3) if not late_piece.empty else np.nan,
        })
    return pd.DataFrame(rows)


def lift(events: pd.DataFrame, placebo: bool = False) -> float:
    """Actual over counterfactual, pooled: sum(treated after) / sum(treated before x comparison growth)."""
    if placebo:  # pretend the markdown started two weeks earlier: should show no effect
        before, after, c_before, c_after = "t_pre2", "t_pre", "c_pre2", "c_pre"
    else:
        before, after, c_before, c_after = "t_pre", "t_post", "c_pre", "c_post"
    usable = events[(events[c_before] > 0) & (events[before] > 0)]
    expected = usable[before] * usable[c_after] / usable[c_before]
    return float(usable[after].sum() / expected.sum())


def late_lift(events: pd.DataFrame) -> float:
    """Lift in weeks 3-4 after the start, against stores still at full price then."""
    usable = events.dropna(subset=["t_late", "l_pre", "l_late"])
    usable = usable[(usable["l_pre"] > 0) & (usable["t_pre"] > 0)]
    expected = usable["t_pre"] * usable["l_late"] / usable["l_pre"]
    return float(usable["t_late"].sum() / expected.sum())


def bootstrap(events: pd.DataFrame, placebo: bool = False) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    products = events["product_id"].unique()
    grouped = {p: g for p, g in events.groupby("product_id")}
    draws = []
    for _ in range(BOOTSTRAPS):
        pick = rng.choice(products, len(products))
        draws.append(lift(pd.concat([grouped[p] for p in pick]), placebo))
    low, high = np.percentile(draws, [2.5, 97.5])
    return round(float(low), 3), round(float(high), 3)


def first_purchase_study(con: duckdb.DuckDBPyConnection) -> dict:
    first = con.execute(
        f"""
        with lines as (
            select f.client_id, f.receipt_id, f.purchased_at, f.store_id, f.product_id, p.category, lp.price_tier,
                   w.on_discount
            from marts.fct_purchase f
            join lux.client c using (client_id)
            join marts.dim_product p using (product_id)
            join lux.product lp using (product_id)
            left join marts.fct_stock_week w
              on w.product_id = f.product_id and w.store_id = f.store_id
             and w.week_start = cast(date_trunc('week', f.purchased_at) as date)
        ),
        first_visit as (
            select * from lines
            qualify dense_rank() over (partition by client_id order by purchased_at, receipt_id) = 1
        ),
        labelled as (
            select client_id, any_value(purchased_at) as first_at, any_value(store_id) as store_id,
                   any_value(category) as category, any_value(price_tier) as price_tier,
                   bool_and(on_discount is not null) as known,
                   bool_or(coalesce(on_discount, false)) as discounted
            from first_visit group by 1
        ),
        later as (
            select l.client_id, count(distinct r.receipt_id) as visits_next, coalesce(sum(r.value_eur), 0) as spend_next
            from labelled l
            left join lux.receipt r on r.client_id = l.client_id
             and r.purchased_at > l.first_at and r.purchased_at <= l.first_at + interval 365 day
            group by 1
        )
        select l.*, strftime(l.first_at, '%Y-%m') as month, k.visits_next, k.spend_next
        from labelled l join later k using (client_id)
        where l.known and l.first_at < date '{FIRST_PURCHASE_BEFORE}'
        """
    ).fetchdf()
    first["returned"] = first["visits_next"] > 0
    strata = ["store_id", "category", "month", "price_tier"]
    cells = first.groupby(strata + ["discounted"]).agg(n=("client_id", "size"), returned=("returned", "mean"),
                                                        spend=("spend_next", "mean")).reset_index()
    wide = cells.pivot_table(index=strata, columns="discounted", values=["n", "returned", "spend"]).dropna()
    weights = wide[("n", True)]  # effect on clients who did start on a discount
    diff_return = ((wide[("returned", True)] - wide[("returned", False)]) * weights).sum() / weights.sum()
    full_spend = (wide[("spend", False)] * weights).sum() / weights.sum()
    diff_spend = ((wide[("spend", True)] - wide[("spend", False)]) * weights).sum() / weights.sum()
    discounted_share = float(first["discounted"].mean())
    return {
        "clients_with_known_first_purchase": len(first),
        "share_first_purchase_on_discount": round(discounted_share, 3),
        "matched_discount_starters": int(weights.sum()),
        "matched_cells": len(wide),
        "raw_return_rate": {"discount_start": round(float(first.loc[first.discounted, "returned"].mean()), 4),
                            "full_price_start": round(float(first.loc[~first.discounted, "returned"].mean()), 4)},
        "matched_difference_in_return_rate": round(float(diff_return), 4),
        "matched_full_price_starter_spend_next_12m_eur": round(float(full_spend)),
        "matched_difference_in_spend_next_12m_eur": round(float(diff_spend)),
        "matched_difference_in_spend_pct": round(float(diff_spend / full_spend) * 100, 1),
    }


def run(warehouse: Path = WAREHOUSE) -> dict:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        events = markdown_events(con)
        clients = first_purchase_study(con)
    events["depth_bin"] = pd.cut(events["depth"], DEPTH_BINS, labels=DEPTH_LABELS, include_lowest=True)
    by_depth = []
    for label, group in events.groupby("depth_bin", observed=True):
        low, high = bootstrap(group)
        p_low, p_high = bootstrap(group, placebo=True)
        by_depth.append({"depth": label, "events": len(group), "mean_depth": round(float(group["depth"].mean()), 3),
                         "lift": round(lift(group), 3), "ci_low": low, "ci_high": high,
                         "lift_weeks_3_4": round(late_lift(group), 3),
                         "placebo_lift": round(lift(group, placebo=True), 3), "placebo_ci": [p_low, p_high]})
    low, high = bootstrap(events)
    report = {
        "markdown_lift": {"events": len(events), "pieces": int(events["product_id"].nunique()),
                          "overall_lift": round(lift(events), 3), "overall_ci": [low, high],
                          "overall_placebo": round(lift(events, placebo=True), 3),
                          "by_depth": by_depth},
        "discount_acquired_clients": clients,
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "discount_effects.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
