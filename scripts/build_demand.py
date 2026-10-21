"""Demand: recover sales lost to stockouts, then forecast store-level demand for new pieces.

1. Weekly demand model on weeks where stock lasted (Poisson gradient boosting).
2. Recovery methods validated by artificial censoring: weeks where demand is
   known are capped at a random stock level, then recovered.
3. Recovery applied to the real weeks where a store sold out.
4. Launch forecast: for each store and new piece, the full-price weekly demand
   rate (mean recovered weekly demand over weeks without discount), with
   10th/50th/90th percentiles. Split by release date: train before 2019,
   validate on H1 2019, test once on Jul-Nov 2019 launches (ends before the
   2020 lockdowns).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from couture_atelier.client_models import DEFAULT_PARAMS, gbm
from couture_atelier.demand import (
    artificial_censoring,
    category_codes,
    coverage,
    pinball_loss,
    recover,
    wape,
)
from couture_atelier.paths import REPORTS, SILHOUETTE_DIR, WAREHOUSE

SPLITS = {"train_before": "2019-01-01", "valid_before": "2019-07-01", "test_before": "2019-12-01"}
MIN_FULL_PRICE_WEEKS = 4
IMAGE_COMPONENTS = 16
QUANTILES = (0.1, 0.5, 0.9)
WAPE_TIE = 0.005
WEEK_FEATURES = ["week_index", "discount", "price_index", "category_code", "lag_units", "chain_units_other",
                 "store_week_units", "opening_units"]


# ---- 1-3. Weekly demand and recovery -------------------------------------------------------
def weekly_frame(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    frame = con.execute(
        """
        select s.*, p.category, p.release_date,
               coalesce(lag(s.units_sold) over (partition by s.product_id, s.store_id order by s.week_index), 0)
                   as lag_units,
               sum(s.units_sold) over (partition by s.product_id, s.week_index) - s.units_sold as chain_units_other,
               sum(s.units_sold) over (partition by s.store_id, s.week_start) as store_week_units
        from marts.fct_stock_week s
        join marts.dim_product p using (product_id)
        order by s.product_id, s.store_id, s.week_index
        """
    ).fetchdf()
    frame, _ = category_codes(frame, ["category"])
    return frame


def recovery_study(weeks: pd.DataFrame, seed: int = 7) -> tuple[pd.DataFrame, dict]:
    """Validate recovery on held-out store-products that never sold out, then recover real sell-outs."""
    pairs = weeks.groupby(["product_id", "store_id"])["censored"].transform("max") == 0
    clean = weeks[pairs & ~weeks["supply_gap"]]
    rng = np.random.default_rng(seed)
    pair_ids = clean[["product_id", "store_id"]].drop_duplicates()
    held = pair_ids.sample(frac=0.2, random_state=seed)
    held_mask = clean.set_index(["product_id", "store_id"]).index.isin(held.set_index(["product_id", "store_id"]).index)
    train, check = clean[~held_mask], clean[held_mask]

    model = gbm("regressor", {**DEFAULT_PARAMS, "min_samples_leaf": 200}, "poisson")
    model.fit(train[WEEK_FEATURES], train["units_sold"])

    validation = {}
    true_demand = check["units_sold"].to_numpy()
    mu = model.predict(check[WEEK_FEATURES])
    for share in (0.1, 0.3, 0.5):
        observed, censored = artificial_censoring(true_demand, mu, share, seed=int(rng.integers(1_000_000)))
        truth = true_demand[censored]
        row = {"weeks_censored": int(censored.sum())}
        for method in ("naive", "max", "conditional"):
            est = recover(observed, censored, mu, method)[censored]
            row[method] = {"total_bias_pct": round(float(est.sum() / truth.sum() - 1) * 100, 1),
                           "mae_units": round(float(np.abs(est - truth).mean()), 3)}
        validation[f"censor_{int(share * 100)}pct_of_weeks"] = row

    weeks = weeks.copy()
    weeks["expected_units"] = model.predict(weeks[WEEK_FEATURES])
    weeks["recovered_units"] = recover(weeks["units_sold"], weeks["censored"], weeks["expected_units"], "conditional")
    real = {
        "censored_weeks": int(weeks["censored"].sum()),
        "censored_share": round(float(weeks["censored"].mean()), 4),
        "store_products_ever_censored": round(float(
            weeks.groupby(["product_id", "store_id"])["censored"].max().mean()), 4),
        "units_sold": int(weeks["units_sold"].sum()),
        "recovered_units": round(float(weeks["recovered_units"].sum())),
        "demand_added_pct": round(float(weeks["recovered_units"].sum() / weeks["units_sold"].sum() - 1) * 100, 2),
    }
    return weeks, {"validation_artificial_censoring": validation, "applied_to_real_sell_outs": real}


# ---- 4. Launch forecast --------------------------------------------------------------------
def image_features(products: pd.DataFrame, train_ids: set[int]) -> pd.DataFrame:
    catalogue = pd.read_parquet(SILHOUETTE_DIR / "data" / "processed" / "visuelle_products.parquet")
    vectors = np.load(SILHOUETTE_DIR / "artifacts" / "embeddings" / "visuelle" / "marqo_fashion_siglip.npy")
    ids = catalogue["external_code"].to_numpy()
    fit_rows = np.isin(ids, list(train_ids))
    pca = PCA(n_components=IMAGE_COMPONENTS, random_state=0).fit(vectors[fit_rows])
    comps = pd.DataFrame(pca.transform(vectors), columns=[f"image_pc{i}" for i in range(IMAGE_COMPONENTS)])
    comps["product_id"] = ids
    return products.merge(comps, on="product_id", how="left")


def trend_features(con: duckdb.DuckDBPyConnection, products: pd.DataFrame) -> pd.DataFrame:
    """Search interest for the piece's category, colour and fabric in the 4 weeks before launch."""
    trends = con.execute("select * from raw.gtrends order by date").fetchdf()
    trends["date"] = pd.to_datetime(trends["date"])
    rolled = trends.set_index("date").rolling(4, min_periods=1).mean()
    out = products.copy()
    for column in ("category", "color", "fabric"):
        values = []
        for tag, release in zip(out[column], out["release_date"], strict=True):
            if tag not in rolled.columns:
                values.append(np.nan)
                continue
            before = rolled.loc[rolled.index < pd.Timestamp(release), tag]
            values.append(float(before.iloc[-1]) if len(before) else np.nan)
        out[f"trend_{column}"] = values
    return out


def launch_frame(con: duckdb.DuckDBPyConnection, weeks: pd.DataFrame) -> pd.DataFrame:
    full_price = weeks[~weeks["on_discount"]]
    target = full_price.groupby(["product_id", "store_id"]).agg(
        full_price_weeks=("week_index", "size"),
        rate_recovered=("recovered_units", "mean"),
        rate_sold=("units_sold", "mean"),
        opening_units=("opening_units", "first"),
    ).reset_index()
    target = target[target["full_price_weeks"] >= MIN_FULL_PRICE_WEEKS]
    products = con.execute(
        "select product_id, category, color, fabric, price_index, release_date, stores_launched "
        "from marts.dim_product where has_attributes"
    ).fetchdf()
    frame = target.merge(products, on="product_id")
    frame["release_date"] = pd.to_datetime(frame["release_date"])
    frame["release_month"] = frame["release_date"].dt.month
    frame["release_period"] = frame["release_date"].dt.to_period("M")
    # Launch breadth drifts upward over time (33 stores per piece in 2017-18, 41-44 in 2019), so the
    # model uses each piece's breadth relative to other launches in the same quarter.
    pieces = frame.drop_duplicates("product_id")[["product_id", "stores_launched", "release_date"]].copy()
    pieces["stores_launched_pct"] = pieces.groupby(pieces["release_date"].dt.to_period("Q"))[
        "stores_launched"].rank(pct=True)
    frame = frame.merge(pieces[["product_id", "stores_launched_pct"]], on="product_id")

    # Store history known at launch: full-price rates of the store's pieces launched 3-12 months earlier.
    monthly = frame.groupby(["store_id", "release_period"])["rate_recovered"].agg(["sum", "count"]).reset_index()
    monthly_cat = frame.groupby(["store_id", "category", "release_period"])["rate_recovered"].agg(
        ["sum", "count"]).reset_index()

    def history(keys: list[str], table: pd.DataFrame, name: str) -> pd.Series:
        merged = frame[keys + ["release_period"]].reset_index().merge(table, on=keys, suffixes=("", "_hist"))
        lag = (merged["release_period"] - merged["release_period_hist"]).apply(lambda d: d.n)
        merged = merged[(lag >= 3) & (lag <= 12)]
        agg = merged.groupby("index")[["sum", "count"]].sum()
        return (agg["sum"] / agg["count"]).rename(name)

    frame = frame.join(history(["store_id"], monthly, "store_rate_hist"))
    frame = frame.join(history(["store_id", "category"], monthly_cat, "store_category_rate_hist"))
    return frame


def build_demand(warehouse: Path = WAREHOUSE) -> dict:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        weeks = weekly_frame(con)
        weeks, recovery = recovery_study(weeks)
        frame = launch_frame(con, weeks)
        frame = trend_features(con, frame)

    train_ids = set(frame.loc[frame["release_date"] < SPLITS["train_before"], "product_id"])
    frame = image_features(frame, train_ids)
    frame, _ = category_codes(frame, ["category", "color", "fabric"],
                              {c: sorted(frame.loc[frame["release_date"] < SPLITS["train_before"], c].unique())
                               for c in ("category", "color", "fabric")})
    split = np.select(
        [frame["release_date"] < SPLITS["train_before"], frame["release_date"] < SPLITS["valid_before"],
         frame["release_date"] < SPLITS["test_before"]], ["train", "valid", "test"], default="excluded")
    frame["split"] = split

    base = ["category_code", "color_code", "fabric_code", "price_index", "release_month", "stores_launched_pct",
            "store_rate_hist", "store_category_rate_hist"]
    image = [f"image_pc{i}" for i in range(IMAGE_COMPONENTS)]
    trend = ["trend_category", "trend_color", "trend_fabric"]
    feature_sets = {
        "base": base,
        "base+image": base + image,
        "base+trends": base + trend,
        "base+image+trends": base + image + trend,
        "base+image+trends+opening_units": base + image + trend + ["opening_units"],
    }
    train, valid, test = (frame[frame["split"] == s] for s in ("train", "valid", "test"))
    train_valid = pd.concat([train, valid])  # refit on everything before the test launches
    target = "rate_recovered"

    def evaluate(rows: pd.DataFrame, pred: np.ndarray) -> dict:
        return {"wape": round(wape(rows[target], pred), 4),
                "bias_pct": round(float(pred.sum() / rows[target].sum() - 1) * 100, 1)}

    results: dict[str, dict] = {}
    for name, rows, history in (("valid", valid, train), ("test", test, train_valid)):
        store_cat = history.groupby(["store_id", "category"])[target].mean()
        keys = list(zip(rows["store_id"], rows["category"], strict=True))
        fallback = rows["store_rate_hist"].fillna(history[target].mean()).to_numpy()
        naive = np.array([store_cat.get(k, np.nan) for k in keys])
        results.setdefault("store_category_average", {})[name] = evaluate(rows, np.where(np.isnan(naive), fallback, naive))

    params = {**DEFAULT_PARAMS, "min_samples_leaf": 100}
    for fs, features in feature_sets.items():
        on_train = gbm("regressor", params, "poisson").fit(train[features], train[target])
        on_train_valid = gbm("regressor", params, "poisson").fit(train_valid[features], train_valid[target])
        results[f"gbm|{fs}"] = {"valid": evaluate(valid, on_train.predict(valid[features])),
                                "test": evaluate(test, on_train_valid.predict(test[features]))}
    # The retailer's own opening allocation is reported but not selectable: using it would piggyback on the
    # merchant's forecast. Feature sets are listed simplest first; take the simplest within WAPE_TIE of the best.
    candidates = [k for k in results if k.startswith("gbm") and "opening_units" not in k]
    best_wape = min(results[k]["valid"]["wape"] for k in candidates)
    chosen = next(k for k in candidates if results[k]["valid"]["wape"] <= best_wape + WAPE_TIE)
    features = feature_sets[chosen.split("|")[1]]

    # Quantile forecasts for the chosen feature set; the interval is judged on the test launches.
    quantile_preds = {}
    for q in QUANTILES:
        model = gbm("regressor", {**params, "quantile": q}, "quantile").fit(train_valid[features], train_valid[target])
        quantile_preds[q] = model.predict(test[features])
    y = test[target].to_numpy()
    interval = {
        "coverage_q10_q90": round(coverage(y, quantile_preds[0.1], quantile_preds[0.9]), 3),
        **{f"pinball_q{int(q * 100)}": round(pinball_loss(y, quantile_preds[q], q), 4) for q in QUANTILES},
    }

    # Training on raw sales instead of recovered demand: how much does it matter here?
    raw_model = gbm("regressor", params, "poisson").fit(train_valid[features], train_valid["rate_sold"])
    raw_vs_recovered = {
        "trained_on_raw_sales": evaluate(test, raw_model.predict(test[features])),
        "trained_on_recovered_demand": results[chosen]["test"],
    }

    mean_model = gbm("regressor", params, "poisson").fit(train_valid[features], train_valid[target])
    out = test[["product_id", "store_id", "category", "release_date", "opening_units", target]].copy()
    out["forecast_mean"] = mean_model.predict(test[features])
    for q in QUANTILES:
        out[f"forecast_q{int(q * 100)}"] = quantile_preds[q]
    out["data_provenance"] = "derived"
    with duckdb.connect(str(warehouse)) as con:
        con.register("w", weeks[["product_id", "store_id", "week_index", "units_sold", "censored",
                                 "expected_units", "recovered_units"]])
        con.execute("create or replace table marts.fct_demand_week as select * from w")
        con.register("f", out)
        con.execute("create or replace table lux.demand_forecast as select * from f")

    report = {
        "recovery": recovery,
        "launch_forecast": {
            "target": "full-price weekly demand rate per store and piece (recovered)",
            "rows": {s: int((frame["split"] == s).sum()) for s in ("train", "valid", "test")},
            "models": results, "chosen_on_validation": chosen,
            "quantile_interval_test": interval,
            "raw_sales_vs_recovered_target_test": raw_vs_recovered,
        },
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "demand.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_demand(), indent=2))
