"""Assemble the public demo bundle: aggregates, model outputs, synthetic examples and public data only.

The dashboard reads nothing else, so the local and hosted versions are the
same. Every file is declared in MANIFEST.json with its provenance, and
``couture_atelier.public_bundle.check_bundle`` must pass before the bundle is
written (no real client identifiers, raw timestamps or groups under 10).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from couture_atelier.client_frames import FEATURE_SETS
from couture_atelier.paths import PROCESSED, PUBLIC_BUNDLE, REPORTS, ROOT, WAREHOUSE
from couture_atelier.public_bundle import MIN_GROUP_SIZE, check_bundle

REPORT_FILES = {
    "luxury_layer.json": "model_output", "client_value.json": "model_output", "clienteling.json": "model_output",
    "recommendations.json": "model_output", "demand.json": "model_output", "scarcity_scenario.json": "model_output",
    "allocation.json": "model_output", "client_allocation.json": "model_output",
    "discount_effects.json": "model_output", "tier_dial.json": "model_output",
    "sensitivity_client.csv": "model_output", "client_experiments.csv": "model_output",
    "client_feature_importance.csv": "model_output", "clienteling_policies.csv": "model_output",
    "allocation_policies.csv": "model_output", "pilot_size_value.csv": "model_output",
    "client_value_deciles.csv": "aggregate", "market_pulse.csv": "public_source",
}
DRIFT_FEATURES = ["spend_12m", "visits_12m", "recency_weeks", "lifetime_visits", "products_12m",
                  "full_price_share_12m", "p_alive_mbg", "expected_visits_mbg"]
DRIFT_WINDOWS = {"2017-12-31": "Dec 2017 (training)", "2018-12-31": "Dec 2018 (test)",
                 "2019-12-31": "Dec 2019 (before COVID)"}
DEMO_BOOK_SIZE = 40
SEED = 11


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index on reference deciles (0.1 = watch, 0.25 = act)."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    ref = np.histogram(np.clip(reference, edges[0], edges[-1]), edges)[0] / len(reference)
    cur = np.histogram(np.clip(current, edges[0], edges[-1]), edges)[0] / len(current)
    ref, cur = np.clip(ref, 1e-4, None), np.clip(cur, 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def drift_table() -> pd.DataFrame:
    """Input drift between consecutive scoring dates, plus the training-to-test shift behind the spend bias."""
    frames = {c: pd.read_parquet(PROCESSED / "client_frames" / f"{c}.parquet") for c in DRIFT_WINDOWS}
    comparisons = [("2017-12-31", "2018-12-31"), ("2018-12-31", "2019-12-31")]
    rows = []
    for ref, cur in comparisons:
        label = f"{DRIFT_WINDOWS[ref]} -> {DRIFT_WINDOWS[cur]}"
        for feature in DRIFT_FEATURES:
            value = psi(frames[ref][feature].to_numpy(float), frames[cur][feature].to_numpy(float))
            rows.append({"comparison": label, "feature": feature, "psi": round(value, 3), "n": len(frames[cur]),
                         "status": "act" if value >= 0.25 else "watch" if value >= 0.1 else "stable"})
    return pd.DataFrame(rows)


def advisor_aggregates(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    actions = con.execute(
        """
        select a.*, case when a.spend_12m >= quantile_cont(a.spend_12m, 0.99) over () then 'VIC'
                         when a.spend_12m >= quantile_cont(a.spend_12m, 0.90) over () then 'core'
                         when a.spend_12m > 0 then 'aspirational' else 'lapsed' end as tier
        from lux.clienteling_actions a
        """
    ).fetchdf()
    reasons = (actions.groupby(["tier", "reason"]).size().rename("n").reset_index())
    reasons = reasons[reasons["n"] >= MIN_GROUP_SIZE]
    eligible = actions[actions["reason"].ne("no personalisation consent") & actions["reason"].ne(
        "not active in the last two years")]
    agree = (eligible.groupby(["tier", "action", "challenger_action"]).size().rename("n").reset_index())
    agree = agree[agree["n"] >= MIN_GROUP_SIZE]
    return reasons, agree


def synthetic_book(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Illustrative advisor book of synthetic clients, drawn from tier-level aggregates only.

    Each attribute is sampled independently from per-tier deciles (each decile
    pools many real clients), so no synthetic row corresponds to a real client.
    """
    actions = con.execute(
        """
        select spend_12m, recency_weeks, predicted_uplift_eur,
               case when spend_12m >= quantile_cont(spend_12m, 0.99) over () then 'VIC'
                    when spend_12m >= quantile_cont(spend_12m, 0.90) over () then 'core'
                    else 'aspirational' end as tier
        from lux.clienteling_actions where consent_status = 'consented' and recency_weeks <= 104
        """
    ).fetchdf()
    categories = con.execute(
        "select category, count(*) as n from marts.fct_purchase join marts.dim_product using (product_id) "
        "group by 1 order by 2 desc limit 8"
    ).fetchdf()
    rng = np.random.default_rng(SEED)
    tier_share = {"VIC": 0.05, "core": 0.25, "aspirational": 0.70}
    rows = []
    for i in range(DEMO_BOOK_SIZE):
        tier = rng.choice(list(tier_share), p=list(tier_share.values()))
        pool = actions[actions["tier"] == tier]
        deciles = {c: np.quantile(pool[c], np.linspace(0.05, 0.95, 10)) for c in
                   ("spend_12m", "recency_weeks", "predicted_uplift_eur")}
        rows.append({
            "client": f"Client {i + 1:03d}", "tier": tier,
            "spend_last_12m_eur": round(float(rng.choice(deciles["spend_12m"])), -2),
            "weeks_since_last_visit": int(rng.choice(deciles["recency_weeks"])),
            "predicted_contact_value_eur": round(float(rng.choice(deciles["predicted_uplift_eur"])), -1),
            "favourite_category": rng.choice(categories["category"], p=categories["n"] / categories["n"].sum()),
        })
    book = pd.DataFrame(rows)
    contacts = max(1, round(DEMO_BOOK_SIZE * 0.16))  # 40 contacts for a book of about 250 clients
    book["champion_action"] = np.where(book["spend_last_12m_eur"].rank(ascending=False, method="first") <= contacts,
                                       "contact", "no action")
    book["challenger_action"] = np.where(
        (book["predicted_contact_value_eur"].rank(ascending=False, method="first") <= contacts)
        & (book["predicted_contact_value_eur"] > 0), "contact", "no action")
    return book.sort_values("spend_last_12m_eur", ascending=False)


def planner_aggregates(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """
        select category, count(*) as n,
               sum(rate_recovered) * 12 as true_season_demand, sum(forecast_mean) * 12 as forecast_season_demand,
               avg((rate_recovered between forecast_q10 and forecast_q90)::int) as interval_coverage
        from lux.demand_forecast group by 1 having count(*) >= 10 order by 3 desc
        """
    ).fetchdf()


def gaps_log() -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^\| (G\d+) \| (.+?) \| (.+?) \| (Fixed(?: \(partly\))?|Flagged|Accepted) \| (.+?) \| (.+) \|$")
    for line in (ROOT / "docs" / "gaps_and_decisions.md").read_text().splitlines():
        match = pattern.match(line)
        if match:
            rows.append(dict(zip(["id", "area", "finding", "status", "handled", "interview_question"],
                                 match.groups(), strict=True)))
    return pd.DataFrame(rows)


def build_public_bundle(target: Path = PUBLIC_BUNDLE) -> dict:
    staging = target.with_name(target.name + "_staging")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    manifest = []

    def add(name: str, provenance: str, description: str) -> None:
        manifest.append({"path": name, "provenance": provenance, "description": description})

    for name, provenance in REPORT_FILES.items():
        shutil.copy(REPORTS / name, staging / name)
        add(name, provenance, "copied from reports/")
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        reasons, agree = advisor_aggregates(con)
        reasons.to_csv(staging / "advisor_reasons.csv", index=False)
        add("advisor_reasons.csv", "aggregate", "clients by value tier and champion reason (groups of 10+)")
        agree.to_csv(staging / "champion_challenger_overlap.csv", index=False)
        add("champion_challenger_overlap.csv", "aggregate", "eligible clients by tier and champion/challenger action")
        synthetic_book(con).to_csv(staging / "demo_advisor_book.csv", index=False)
        add("demo_advisor_book.csv", "synthetic", "illustrative advisor book drawn from tier-level deciles")
        planner_aggregates(con).to_csv(staging / "forecast_by_category.csv", index=False)
        add("forecast_by_category.csv", "aggregate", "launch forecast vs true demand by category")
    drift_table().to_csv(staging / "drift.csv", index=False)
    add("drift.csv", "aggregate", "population stability index of model inputs between consecutive scoring dates")
    gaps_log().to_csv(staging / "gaps_log.csv", index=False)
    add("gaps_log.csv", "documentation", "docs/gaps_and_decisions.md as a table")
    (staging / "MANIFEST.json").write_text(json.dumps({"files": manifest}, indent=2) + "\n")

    problems = check_bundle(staging)
    if problems:
        raise ValueError(f"Public bundle failed the guard: {problems}")
    shutil.rmtree(target, ignore_errors=True)
    staging.rename(target)
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    return {"files": len(manifest), "size_kb": round(size / 1024), "features_monitored": len(DRIFT_FEATURES),
            "feature_sets": list(FEATURE_SETS)}


if __name__ == "__main__":
    print(json.dumps(build_public_bundle(), indent=2))
