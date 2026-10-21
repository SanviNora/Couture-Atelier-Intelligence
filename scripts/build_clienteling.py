"""Clienteling: which clients each advisor should contact, judged by the sales a contact adds.

1. Pilot (2018-06-30): half of consented clients are contacted at random.
   Their next-90-day spend is real if not contacted; contact effects follow
   the documented rule in configs/clienteling.yml.
2. Uplift models are trained on 80% of the pilot and chosen by Qini area on
   the other 20%, using observed outcomes only (as a real team would).
3. Deployment (2019-06-30): every policy gets the same advisor capacity and is
   scored by the true incremental euros of the clients it contacts.
4. Power analysis for the live experiment that would confirm the result.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm

from couture_atelier.client_frames import FEATURE_SETS, add_probabilistic, window_frame
from couture_atelier.client_models import DEFAULT_PARAMS, fit_return, gbm
from couture_atelier.clienteling import (
    allocate,
    contact_effect,
    fit_t_learner,
    fit_t_learner_logistic,
    fit_transformed_outcome,
    fit_x_learner,
    qini_area,
    simulate_contacted_spend,
    true_incremental_value,
)
from couture_atelier.luxury import stable_uniform
from couture_atelier.paths import CONFIGS, PROCESSED, REPORTS, WAREHOUSE

FEATURES = FEATURE_SETS["full"]
CORE = ["lifetime_visits", "recency_weeks", "spend_12m", "visits_12m", "mean_visit_value_12m",
        "p_alive_mbg", "expected_visits_mbg", "tenure_weeks_capped"]
PARAMS = {**DEFAULT_PARAMS, "min_samples_leaf": 300}
LARGE_LEAVES = {**DEFAULT_PARAMS, "min_samples_leaf": 2000, "learning_rate": 0.03, "max_leaf_nodes": 15}
QINI_BOOTSTRAPS = 100


def visit_value_proxy(frame: pd.DataFrame) -> np.ndarray:
    """Expected value of one extra visit, from history only."""
    mean_value = frame["mean_visit_value_12m"].where(frame["mean_visit_value_12m"] > 0, frame["last_visit_value"])
    return (0.8 * mean_value).to_numpy()


# Uplift learners on the binary outcome "visited in the next 90 days", simplest first.
LEARNERS = {
    "t_logistic_core": lambda d: fit_t_learner_logistic(d, CORE, "visit_obs"),
    "t_logistic_full": lambda d: fit_t_learner_logistic(d, FEATURES, "visit_obs"),
    "t_gbm_full": lambda d: fit_t_learner(d, FEATURES, "visit_obs", PARAMS),
    "t_gbm_full_large_leaves": lambda d: fit_t_learner(d, FEATURES, "visit_obs", LARGE_LEAVES),
    "transformed_outcome_core": lambda d: fit_transformed_outcome(d, CORE, "visit_obs", LARGE_LEAVES),
    "x_learner_spend": lambda d: fit_x_learner(d, FEATURES, "spend_obs", PARAMS),
}


def frame_at(con: duckdb.DuckDBPyConnection, cutoff: str, horizon: int) -> pd.DataFrame:
    cache = PROCESSED / "clienteling_frames" / f"{cutoff}_{horizon}d.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    frame, _ = add_probabilistic(window_frame(con, cutoff))
    outcome = con.execute(
        f"""
        select client_id, count(distinct cast(purchased_at as date)) as visits_h, sum(value_eur) as spend_h
        from lux.receipt
        where cast(purchased_at as date) > date '{cutoff}'
          and cast(purchased_at as date) <= date '{cutoff}' + interval {horizon} day
        group by 1
        """
    ).fetchdf()
    clients = con.execute("select client_id, advisor_id, consent_status from lux.client").fetchdf()
    frame = frame.merge(outcome, on="client_id", how="left").merge(clients, on="client_id", how="inner")
    frame[["visits_h", "spend_h"]] = frame[["visits_h", "spend_h"]].fillna(0)
    frame = frame.sort_values("client_id").reset_index(drop=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache, index=False)
    return frame


def eligible_mask(frame: pd.DataFrame, cfg: dict) -> np.ndarray:
    e = cfg["eligibility"]
    consented = frame["consent_status"].eq("consented") if e["require_consent"] else True
    return (consented & (frame["recency_weeks"] <= e["active_within_weeks"])).to_numpy()


def run_pilot(pilot: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    pilot = pilot[eligible_mask(pilot, cfg)].copy()
    rng = np.random.default_rng(cfg["pilot"]["seed"])
    pilot["treated"] = rng.random(len(pilot)) < cfg["pilot"]["treated_share"]
    effect = contact_effect(pilot, cfg)
    contacted = simulate_contacted_spend(effect, pilot["spend_h"], seed=cfg["pilot"]["seed"] + 1)
    pilot["spend_obs"] = np.where(pilot["treated"], contacted, pilot["spend_h"])
    pilot["visit_obs"] = pilot["spend_obs"] > 0
    pilot["true_tau"] = effect["tau"]
    return pilot


def holdout_needed(delta: float, variance: float, n_total: int, alpha: float = 0.05, power: float = 0.8) -> float | None:
    """Smallest holdout share (1-50%) whose comparison detects ``delta``; None if even 50% is not enough."""
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    target = (delta / z) ** 2
    for share in np.arange(0.01, 0.51, 0.01):
        if variance * (1 / (n_total * (1 - share)) + 1 / (n_total * share)) <= target:
            return round(float(share), 2)
    return None


def design_effect(selected: pd.DataFrame, column: str) -> tuple[float, float]:
    """Variance inflation when whole advisors (not single clients) are held out."""
    values = selected[column].astype(float)
    between = float(selected.groupby("advisor_id")[column].mean().var())
    icc = max(0.0, between / (between + float(values.var())))
    cluster_size = float(selected.groupby("advisor_id").size().mean())
    return round(icc, 4), round(1 + (cluster_size - 1) * icc, 2)


def power_analysis(selected: pd.DataFrame, effect: pd.DataFrame, true_value: pd.Series) -> dict:
    """Holdout needed to confirm the contact effect among the clients the policy would contact.

    Two candidate success metrics: euros spent (what the business cares about)
    and the share of clients who visit (far less noisy). CUPED uses last-12-month
    behaviour as a covariate to remove predictable variance.
    """
    n = len(selected)
    visited = (selected["spend_h"] > 0).astype(float)
    tau = effect["tau"]
    visit_delta = float(np.where(tau > 0, tau * (1 - visited), tau * visited).mean())
    out = {"contacts": n}
    for metric, values, delta, covariate in (
        ("spend_eur", selected["spend_h"], float(true_value.mean()), selected["spend_12m"]),
        ("visit_rate", visited, visit_delta, (selected["visits_12m"] > 0).astype(float) + selected["visits_12m"]),
    ):
        variance = float(values.var())
        rho = float(np.corrcoef(values, covariate)[0, 1])
        frame = selected.assign(_metric=values)
        icc, deff = design_effect(frame, "_metric")
        out[metric] = {
            "true_effect_per_contact": round(delta, 4),
            "sd": round(variance**0.5, 4),
            "holdout_share_client_randomised": holdout_needed(delta, variance, n),
            "cuped_correlation": round(rho, 3),
            "holdout_share_client_randomised_cuped": holdout_needed(delta, variance * (1 - rho**2), n),
            "advisor_icc": icc,
            "design_effect_advisor_randomised": deff,
            "holdout_share_advisor_randomised_cuped": holdout_needed(delta, variance * (1 - rho**2) * deff, n),
        }
    return out


def build_clienteling(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "clienteling.yml").read_text())
    horizon = cfg["horizon_days"]
    with duckdb.connect(str(warehouse), read_only=True) as con:
        pilot_frame = frame_at(con, cfg["cutoffs"]["pilot"], horizon)
        deploy = frame_at(con, cfg["cutoffs"]["deploy"], horizon)

    # ---- Pilot and model choice on observed outcomes only ------------------------------
    pilot = run_pilot(pilot_frame, cfg)
    holdout = stable_uniform(pilot["client_id"], seed=5) < 0.2
    fit_part, check_part = pilot[~holdout], pilot[holdout]
    treated_check = check_part["treated"].to_numpy()
    visits_check = check_part["visit_obs"].astype(float).to_numpy()
    rng_boot = np.random.default_rng(0)
    boots = [rng_boot.integers(0, len(check_part), len(check_part)) for _ in range(QINI_BOOTSTRAPS)]
    qini = {}
    for name, make in LEARNERS.items():
        score = make(fit_part)(check_part)
        if name == "x_learner_spend":  # euro-scale learner: rank on the same visit outcome for comparability
            score = score / np.maximum(visit_value_proxy(check_part), 1)
        draws = [qini_area(score[i], treated_check[i], visits_check[i]) for i in boots]
        qini[name] = {"qini": round(qini_area(score, treated_check, visits_check), 1),
                      "ci_low": round(float(np.percentile(draws, 2.5)), 1),
                      "ci_high": round(float(np.percentile(draws, 97.5)), 1)}
    best = max(qini, key=lambda n: qini[n]["qini"])
    # Learners are listed simplest first; take the simplest one statistically indistinguishable from the best.
    chosen = next(n for n in LEARNERS if qini[n]["ci_high"] >= qini[best]["qini"])

    # ---- Refit on the full pilot and score the deployment population -------------------
    uplift = {name: make(pilot)(deploy) for name, make in LEARNERS.items()}
    value_at_stake = visit_value_proxy(deploy)
    scorers = {name: (u if name == "x_learner_spend" else u * value_at_stake) for name, u in uplift.items()}
    control = pilot[~pilot["treated"]].rename(columns={"visit_obs": "returned_next"})
    p_return = fit_return("logistic", control, FEATURES, {})(deploy)
    treated = pilot[pilot["treated"]]
    response = gbm("regressor", PARAMS, "poisson").fit(treated[FEATURES], treated["spend_obs"] / 1000)

    effect = contact_effect(deploy, cfg)
    true_value = true_incremental_value(effect, deploy["spend_h"])
    eligible = eligible_mask(deploy, cfg)
    capacity = cfg["capacity"]["contacts_per_advisor"]
    rng = np.random.default_rng(3)
    policies = {
        "random": (rng.random(len(deploy)), False),
        "status_quo_best_clients": (deploy["spend_12m"].to_numpy(), False),
        "response_model": (response.predict(deploy[FEATURES]), False),
        "churn_risk_x_value": (deploy["spend_12m"].to_numpy() * (1 - p_return), False),
        **{f"uplift_{n}": (s, False) for n, s in scorers.items()},
        f"uplift_{chosen}_no_contact_if_no_lift": (scorers[chosen], True),
        "oracle_true_effect": (true_value.to_numpy(), True),
    }
    results, selections = [], {}
    for name, (score, positive_only) in policies.items():
        chosen_mask = allocate(np.asarray(score, dtype=float), deploy["advisor_id"], eligible, capacity, positive_only)
        selections[name] = chosen_mask
        picked = true_value[chosen_mask]
        results.append({
            "policy": name,
            "contacts": int(chosen_mask.sum()),
            "true_incremental_eur": round(float(picked.sum())),
            "eur_per_contact": round(float(picked.mean()), 1) if len(picked) else 0.0,
            "share_do_not_disturb": round(float(effect.loc[chosen_mask, "do_not_disturb"].mean()), 4),
            "share_harmful_contacts": round(float((effect.loc[chosen_mask, "tau"] < 0).mean()), 4),
        })
    table = pd.DataFrame(results)
    oracle = table.loc[table.policy == "oracle_true_effect", "true_incremental_eur"].iloc[0]
    table["share_of_oracle"] = (table["true_incremental_eur"] / oracle).round(3)
    status_quo = table.loc[table.policy == "status_quo_best_clients", "true_incremental_eur"].iloc[0]
    table["vs_status_quo"] = (table["true_incremental_eur"] / status_quo).round(2)

    final_name = f"uplift_{chosen}_no_contact_if_no_lift"
    final_mask = selections[final_name]
    recovery = {n: round(float(np.corrcoef(u[eligible], effect["tau"][eligible])[0, 1]), 3) for n, u in uplift.items()}
    explainable = effect["tau"] - cfg["contact_effect"]["individual_noise_sd"] * effect["trait"]
    ceiling = round(float(np.corrcoef(explainable[eligible], effect["tau"][eligible])[0, 1]), 3)

    # Does the ranking hold at other advisor capacities?
    sweep = []
    for cap in (10, 20, 40, 80):
        oracle_cap = true_value[allocate(true_value.to_numpy(), deploy["advisor_id"], eligible, cap, True)].sum()
        for name in ("random", "status_quo_best_clients", f"uplift_{chosen}_no_contact_if_no_lift"):
            score, positive_only = policies[name]
            mask = allocate(np.asarray(score, dtype=float), deploy["advisor_id"], eligible, cap, positive_only)
            sweep.append({"contacts_per_advisor": cap, "policy": name,
                          "share_of_oracle": round(float(true_value[mask].sum() / oracle_cap), 3),
                          "share_harmful_contacts": round(float((effect.loc[mask, "tau"] < 0).mean()), 4)})
    report = {
        "cutoffs": cfg["cutoffs"], "horizon_days": horizon, "capacity_per_advisor": capacity,
        "pilot": {"clients": len(pilot), "treated": int(pilot["treated"].sum()),
                  "observed_lift_eur_per_client": round(float(
                      pilot.loc[pilot.treated, "spend_obs"].mean() - pilot.loc[~pilot.treated, "spend_obs"].mean()), 1),
                  "true_mean_tau": round(float(pilot["true_tau"].mean()), 4)},
        "model_choice_qini_on_pilot_holdout": qini, "best_qini": best, "chosen_uplift_model": chosen,
        "selection_rule": "simplest learner whose Qini 95% interval reaches the best learner's Qini",
        "deploy": {"clients": len(deploy), "eligible": int(eligible.sum()),
                   "true_effect_correlation_by_learner": recovery,
                   "true_effect_correlation_ceiling_from_features": ceiling},
        "capacity_sweep": sweep,
        "policies": table.to_dict(orient="records"),
        "power_analysis": power_analysis(deploy[final_mask], effect[final_mask], true_value[final_mask]),
    }

    actions = deploy.loc[:, ["client_id", "advisor_id", "consent_status", "spend_12m", "recency_weeks"]].copy()
    actions["predicted_uplift_eur"] = scorers[chosen]
    actions["action"] = np.where(final_mask, "contact", "no_action")
    actions["reason"] = np.select(
        [~eligible & deploy["consent_status"].ne("consented").to_numpy(), ~eligible, final_mask,
         actions["predicted_uplift_eur"].to_numpy() <= 0],
        ["no personalisation consent", "not active in the last two years", "contact expected to add sales",
         "contact not expected to help; leave undisturbed"],
        default="below this advisor's capacity cut-off",
    )
    actions["as_of_date"] = pd.Timestamp(cfg["cutoffs"]["deploy"])
    actions["data_provenance"] = "derived"
    with duckdb.connect(str(warehouse)) as con:
        con.register("a", actions)
        con.execute("create or replace table lux.clienteling_actions as select * from a")
    report["action_counts"] = actions["reason"].value_counts().to_dict()

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "clienteling.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    table.to_csv(REPORTS / "clienteling_policies.csv", index=False)
    return report


if __name__ == "__main__":
    result = build_clienteling()
    print(json.dumps({k: v for k, v in result.items() if k != "policies"}, indent=2, default=str))
    print(pd.DataFrame(result["policies"]).to_string(index=False))
