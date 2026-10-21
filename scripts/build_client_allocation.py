"""Which clients are offered a scarce hero piece: an optimisation with relationship rules.

For each Jul-Nov 2019 hero-piece launch, each boutique's units (from the
optimal store allocation at 60% of forecast) are offered to its clients. An
integer programme per boutique chooses the offers that maximise expected
revenue subject to:

- at most the boutique's units of each piece
- at most one hero piece per client per season
- consent to personal contact; clients who bought the same piece twice or more
  (a reseller signal) are excluded
- at least 30% of offers go to rising clients: not yet VIC, but spending in the
  top 10% and accelerating, so the future VIC pipeline keeps growing

Acceptance probabilities are a documented assumption (category affinity and
engagement). Compared with value-only greedy offers and "VICs first".
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import Bounds, LinearConstraint, milp
from scripts.build_allocation import allocate_piece, load

from couture_atelier.paths import CONFIGS, PROCESSED, REPORTS, WAREHOUSE

DEPLOY = "2019-06-30"
RELATIONSHIP_SWEEP = (0.0, 0.01, 0.02, 0.05, 0.10)


def client_pool(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_parquet(PROCESSED / "clienteling_frames" / f"{DEPLOY}_90d.parquet")
    homes = con.execute("select client_id, home_store_id from lux.client").fetchdf()
    resellers = con.execute(
        f"""
        select distinct client_id from (
            select client_id, product_id, sum(qty) as bought from marts.fct_purchase
            where purchased_at <= date '{DEPLOY}' group by all) where bought >= 2
        """
    ).fetchdf()["client_id"]
    affinity = con.execute(
        f"""
        with counts as (
            select f.client_id, p.category, count(*) as n
            from marts.fct_purchase f join marts.dim_product p using (product_id)
            where f.purchased_at <= date '{DEPLOY}' group by all
        )
        select client_id, category, n * 1.0 / sum(n) over (partition by client_id) as share from counts
        """
    ).fetchdf()
    frame = frame.merge(homes, on="client_id")
    vic_cut = frame["spend_12m"].quantile(0.99)
    top10_cut = frame["spend_12m"].quantile(0.90)
    frame["is_vic"] = frame["spend_12m"] >= vic_cut
    frame["is_rising"] = (~frame["is_vic"] & (frame["spend_12m"] >= top10_cut) & (frame["spend_momentum"] >= 0.6))
    frame["is_reseller_risk"] = frame["client_id"].isin(resellers)
    frame["eligible"] = frame["consent_status"].eq("consented") & ~frame["is_reseller_risk"] & (frame["visits_12m"] > 0)
    return frame, affinity


def acceptance(clients: pd.DataFrame, affinity: pd.DataFrame, category: str) -> np.ndarray:
    """Assumed chance a client buys the piece when offered: category affinity and engagement."""
    share = affinity[affinity["category"] == category].set_index("client_id")["share"]
    cat = clients["client_id"].map(share).fillna(0).to_numpy()
    return np.clip(0.15 + 0.35 * cat + 0.25 * clients["p_alive_mbg"].to_numpy(), 0, 0.9)


def _milp(value: np.ndarray, rows: list, lower: list, upper: list):
    return milp(c=-value, constraints=LinearConstraint(np.array(rows), lower, upper),
                integrality=np.ones(len(value)), bounds=Bounds(0, 1))


def solve_store(cands: pd.DataFrame, units: dict, reserve: float, objective: str) -> np.ndarray:
    """Two-stage integer programme for one boutique.

    Rules: at most the boutique's units of each piece, at most one hero piece
    per client, and at least ``reserve`` of the units offered to rising clients
    (as many as can be). Stage 1 finds the largest number of offers the rules
    allow, so no unit is left unoffered to satisfy a rule; stage 2 maximises
    value while making that many offers.
    """
    n = len(cands)
    rows, lower, upper = [], [], []
    for piece, cap in units.items():
        rows.append((cands["product_id"] == piece).to_numpy(float))
        lower.append(0)
        upper.append(cap)
    for idx in cands.groupby("client_id").indices.values():
        if len(idx) > 1:
            row = np.zeros(n)
            row[idx] = 1
            rows.append(row)
            lower.append(0)
            upper.append(1)
    rising = cands["is_rising"].to_numpy(float)
    rising_target = int(np.ceil(reserve * sum(units.values())))
    # The largest number of rising clients the unit and one-per-client rules allow.
    most_rising = round(-_milp(rising, rows, lower, upper).fun) if rising_target else 0
    rows.append(rising)
    lower.append(min(rising_target, most_rising))
    upper.append(np.inf)
    count = _milp(np.ones(n), rows, lower, upper)
    rows.append(np.ones(n))
    lower.append(round(-count.fun))
    upper.append(np.inf)
    result = _milp(cands[objective].to_numpy(), rows, lower, upper)
    if result.x is None:
        raise RuntimeError(f"Client allocation infeasible: {result.message}")
    return np.round(result.x).astype(bool)


def greedy(cands: pd.DataFrame, units: dict, order: list[str]) -> np.ndarray:
    chosen = np.zeros(len(cands), dtype=bool)
    left = dict(units)
    used: set = set()
    for i in cands.sort_values(order, ascending=False).index:
        row = cands.loc[i]
        if left[row["product_id"]] > 0 and row["client_id"] not in used:
            chosen[cands.index.get_loc(i)] = True
            left[row["product_id"]] -= 1
            used.add(row["client_id"])
    return chosen


def build_client_allocation(warehouse: Path = WAREHOUSE) -> dict:
    alloc_cfg = yaml.safe_load((CONFIGS / "allocation.yml").read_text())
    rules = alloc_cfg["client_allocation"]
    with duckdb.connect(str(warehouse), read_only=True) as con:
        launches = load(con)
        clients, affinity = client_pool(con)
    heroes = launches[launches["hero_piece"]]
    store_units: dict[int, dict] = {}
    for product_id, rows in heroes.groupby("product_id"):
        rows = rows.reset_index(drop=True)
        units = allocate_piece(rows, alloc_cfg)["optimal_with_uncertainty"]
        for store, u in zip(rows["store_id"], units, strict=True):
            if u > 0:
                store_units.setdefault(store, {})[product_id] = int(u)
    categories = heroes.drop_duplicates("product_id").set_index("product_id")["category"]
    prices = heroes.drop_duplicates("product_id").set_index("product_id")["list_price_eur"]

    outcomes = {"revenue_only": [], "vic_first": [], "relationship_value": [],
                "relationship_value_with_pipeline_reserve": []}
    sweep = {k: [] for k in RELATIONSHIP_SWEEP}
    for store, units in store_units.items():
        local = clients[(clients["home_store_id"] == store) & clients["eligible"]]
        if local.empty:
            continue
        frames = []
        for piece, u in units.items():
            cand = local.assign(product_id=piece, price=prices[piece],
                                p_accept=acceptance(local, affinity, categories[piece]))
            cand["expected_revenue"] = cand["p_accept"] * cand["price"]
            cand["relationship_value"] = cand["p_accept"] * rules["relationship_value_share"] * cand["spend_12m"]
            cand["total_value"] = cand["expected_revenue"] + cand["relationship_value"]
            shortlist = pd.concat([
                cand.nlargest(u * rules["candidates_per_unit"], "expected_revenue"),
                cand.nlargest(u * rules["candidates_per_unit"], "total_value"),
                cand[cand["is_rising"]].nlargest(u * rules["candidates_per_unit"], "total_value"),
            ]).drop_duplicates(["client_id", "product_id"])
            frames.append(shortlist)
        cands = pd.concat(frames, ignore_index=True)
        picks = {
            "revenue_only": greedy(cands, units, ["expected_revenue"]),
            "vic_first": greedy(cands.assign(v=cands["is_vic"].astype(int)), units, ["v", "expected_revenue"]),
            "relationship_value": solve_store(cands, units, 0.0, "total_value"),
            "relationship_value_with_pipeline_reserve": solve_store(
                cands, units, rules["reserved_share_rising_clients"], "total_value"),
        }
        for name, chosen in picks.items():
            outcomes[name].append(cands[chosen])
        # How the choice moves with the relationship-value assumption (pipeline reserve kept).
        for kappa in RELATIONSHIP_SWEEP:
            alt = cands.assign(total_value=cands["expected_revenue"]
                               + cands["p_accept"] * kappa * cands["spend_12m"])
            sweep[kappa].append(alt[solve_store(alt, units, rules["reserved_share_rising_clients"], "total_value")])

    report = {"hero_launches": int(heroes["product_id"].nunique()), "boutiques": len(store_units),
              "units_offered": int(sum(sum(u.values()) for u in store_units.values())),
              "eligible_clients": int(clients["eligible"].sum()),
              "excluded_reseller_risk": int(clients["is_reseller_risk"].sum()),
              "rising_clients": int(clients["is_rising"].sum()), "policies": {}}
    for name, parts in outcomes.items():
        offers = pd.concat(parts)
        report["policies"][name] = {
            "offers": len(offers),
            "expected_revenue_eur_m": round(float(offers["expected_revenue"].sum()) / 1e6, 3),
            "expected_relationship_value_eur_m": round(float(offers["relationship_value"].sum()) / 1e6, 3),
            "expected_total_value_eur_m": round(float(offers["total_value"].sum()) / 1e6, 3),
            "expected_units_sold": round(float(offers["p_accept"].sum()), 1),
            "share_offers_to_vic": round(float(offers["is_vic"].mean()), 3),
            "share_offers_to_rising": round(float(offers["is_rising"].mean()), 3),
            "max_offers_per_client": int(offers.groupby("client_id").size().max()),
        }
    report["relationship_value_sensitivity"] = [
        {"relationship_value_share": kappa,
         "expected_revenue_now_eur_m": round(float(pd.concat(parts)["expected_revenue"].sum()) / 1e6, 3),
         "expected_units_sold_now": round(float(pd.concat(parts)["p_accept"].sum()), 1),
         "share_offers_to_vic": round(float(pd.concat(parts)["is_vic"].mean()), 3),
         "share_offers_to_rising": round(float(pd.concat(parts)["is_rising"].mean()), 3)}
        for kappa, parts in sweep.items()]
    base = report["policies"]["relationship_value"]["expected_total_value_eur_m"]
    for policy in report["policies"].values():
        policy["total_value_vs_unconstrained_optimum"] = round(policy["expected_total_value_eur_m"] / base - 1, 4)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "client_allocation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_client_allocation(), indent=2))
