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
    bagged,
    contact_effect,
    fit_s_learner_logistic,
    fit_t_learner,
    fit_t_learner_logistic,
    fit_transformed_outcome,
    fit_x_learner,
    qini_area,
    simulate_contacted_spend,
    true_incremental_value,
    uplift_stability,
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
    "s_logistic_core": lambda d: fit_s_learner_logistic(d, CORE, "visit_obs"),
    "bagged_t_logistic_core": bagged(lambda d: fit_t_learner_logistic(d, CORE, "visit_obs")),
}
TRUST_THRESHOLD = 0.5  # minimum ranking agreement across bootstrap refits before uplift targeting is trusted
PILOT_DRAWS = 3        # independent randomisations of the pilot, to report how much results depend on one draw


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


def choose_learner(pilot: pd.DataFrame) -> dict:
    """Pick an uplift learner using the pilot only.

    Qini on a 20% pilot holdout identifies the learners that are statistically
    indistinguishable from the best; among those, the one whose client ranking
    is most stable across bootstrap refits is chosen. If even that one is
    unstable, uplift targeting is not trusted for this pilot.
    """
    holdout = stable_uniform(pilot["client_id"], seed=5) < 0.2
    fit_part, check_part = pilot[~holdout], pilot[holdout]
    treated_check = check_part["treated"].to_numpy()
    visits_check = check_part["visit_obs"].astype(float).to_numpy()
    rng_boot = np.random.default_rng(0)
    boots = [rng_boot.integers(0, len(check_part), len(check_part)) for _ in range(QINI_BOOTSTRAPS)]
    qini, stability = {}, {}
    for name, make in LEARNERS.items():
        score = make(fit_part)(check_part)
        if name == "x_learner_spend":  # euro-scale learner: rank on the same visit outcome for comparability
            score = score / np.maximum(visit_value_proxy(check_part), 1)
        draws = [qini_area(score[i], treated_check[i], visits_check[i]) for i in boots]
        qini[name] = {"qini": round(qini_area(score, treated_check, visits_check), 1),
                      "ci_low": round(float(np.percentile(draws, 2.5)), 1),
                      "ci_high": round(float(np.percentile(draws, 97.5)), 1)}
    best = max(qini, key=lambda n: qini[n]["qini"])
    tied = [n for n in LEARNERS if qini[n]["ci_high"] >= qini[best]["qini"]]
    for name in tied:
        stability[name] = round(uplift_stability(LEARNERS[name], pilot), 3)
    chosen = max(tied, key=lambda n: stability[n])
    return {"qini": qini, "best_qini": best, "indistinguishable_from_best": tied, "stability": stability,
            "chosen": chosen, "trusted": stability[chosen] >= TRUST_THRESHOLD}


def build_clienteling(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "clienteling.yml").read_text())
    horizon = cfg["horizon_days"]
    with duckdb.connect(str(warehouse), read_only=True) as con:
        pilot_frame = frame_at(con, cfg["cutoffs"]["pilot"], horizon)
        deploy = frame_at(con, cfg["cutoffs"]["deploy"], horizon)

    effect = contact_effect(deploy, cfg)
    true_value = true_incremental_value(effect, deploy["spend_h"])
    eligible = eligible_mask(deploy, cfg)
    capacity = cfg["capacity"]["contacts_per_advisor"]
    value_at_stake = visit_value_proxy(deploy)
    status_quo_score = deploy["spend_12m"].to_numpy()

    def share(score: np.ndarray, positive_only: bool, cap: int = capacity) -> float:
        oracle_value = true_value[allocate(true_value.to_numpy(), deploy["advisor_id"], eligible, cap, True)].sum()
        mask = allocate(np.asarray(score, dtype=float), deploy["advisor_id"], eligible, cap, positive_only)
        return round(float(true_value[mask].sum() / oracle_value), 3)

    # ---- How much does the result depend on one pilot randomisation? -------------------
    robustness = []
    primary = None
    for draw in range(PILOT_DRAWS):
        draw_cfg = {**cfg, "pilot": {**cfg["pilot"], "seed": cfg["pilot"]["seed"] + 100 * draw}}
        pilot = run_pilot(pilot_frame, draw_cfg)
        choice = choose_learner(pilot)
        uplift = LEARNERS[choice["chosen"]](pilot)(deploy)
        score = uplift if choice["chosen"] == "x_learner_spend" else uplift * value_at_stake
        uplift_share = share(score, True)
        status_quo_share = share(status_quo_score, False)
        robustness.append({
            "draw": draw, "chosen": choice["chosen"], "stability": choice["stability"][choice["chosen"]],
            "trusted": choice["trusted"],
            "true_effect_correlation": round(float(np.corrcoef(uplift[eligible], effect["tau"][eligible])[0, 1]), 3),
            "uplift_share_of_oracle": uplift_share, "status_quo_share_of_oracle": status_quo_share,
            "with_trust_check_share_of_oracle": uplift_share if choice["trusted"] else status_quo_share,
        })
        if draw == 0:
            primary = {"pilot": pilot, "choice": choice}

    # ---- Full comparison on the primary pilot draw ------------------------------------
    pilot, choice = primary["pilot"], primary["choice"]
    chosen = choice["chosen"]
    uplift = {name: make(pilot)(deploy) for name, make in LEARNERS.items()}
    scorers = {name: (u if name == "x_learner_spend" else u * value_at_stake) for name, u in uplift.items()}
    control = pilot[~pilot["treated"]].rename(columns={"visit_obs": "returned_next"})
    p_return = fit_return("logistic", control, FEATURES, {})(deploy)
    treated = pilot[pilot["treated"]]
    response = gbm("regressor", PARAMS, "poisson").fit(treated[FEATURES], treated["spend_obs"] / 1000)
    # Champion / challenger: calling best clients stays the deployed policy; a trusted uplift policy runs as a
    # randomised challenger arm and replaces it only if a live test on visit rate shows a gain.
    champion = "status_quo_best_clients"
    challenger = f"uplift_{chosen}_no_contact_if_no_lift" if choice["trusted"] else None

    rng = np.random.default_rng(3)
    policies = {
        "random": (rng.random(len(deploy)), False),
        "status_quo_best_clients": (status_quo_score, False),
        "response_model": (response.predict(deploy[FEATURES]), False),
        "churn_risk_x_value": (status_quo_score * (1 - p_return), False),
        **{f"uplift_{n}": (s, False) for n, s in scorers.items()},
        f"uplift_{chosen}_no_contact_if_no_lift": (scorers[chosen], True),
        "oracle_true_effect": (true_value.to_numpy(), True),
    }
    results, selections = [], {}
    for name, (score, positive_only) in policies.items():
        mask = allocate(np.asarray(score, dtype=float), deploy["advisor_id"], eligible, capacity, positive_only)
        selections[name] = mask
        picked = true_value[mask]
        results.append({
            "policy": name,
            "contacts": int(mask.sum()),
            "true_incremental_eur": round(float(picked.sum())),
            "eur_per_contact": round(float(picked.mean()), 1) if len(picked) else 0.0,
            "share_do_not_disturb": round(float(effect.loc[mask, "do_not_disturb"].mean()), 4),
            "share_harmful_contacts": round(float((effect.loc[mask, "tau"] < 0).mean()), 4),
        })
    table = pd.DataFrame(results)
    oracle = table.loc[table.policy == "oracle_true_effect", "true_incremental_eur"].iloc[0]
    table["share_of_oracle"] = (table["true_incremental_eur"] / oracle).round(3)
    status_quo = table.loc[table.policy == "status_quo_best_clients", "true_incremental_eur"].iloc[0]
    table["vs_status_quo"] = (table["true_incremental_eur"] / status_quo).round(2)

    champion_mask = selections[champion]
    test_policy = challenger or champion
    challenger_mask = selections[test_policy]
    recovery = {n: round(float(np.corrcoef(u[eligible], effect["tau"][eligible])[0, 1]), 3) for n, u in uplift.items()}
    explainable = effect["tau"] - cfg["contact_effect"]["individual_noise_sd"] * effect["trait"]
    ceiling = round(float(np.corrcoef(explainable[eligible], effect["tau"][eligible])[0, 1]), 3)
    sweep = []
    for cap in (10, 20, 40, 80):
        for name in ("random", "status_quo_best_clients", f"uplift_{chosen}_no_contact_if_no_lift"):
            score, positive_only = policies[name]
            mask = allocate(np.asarray(score, dtype=float), deploy["advisor_id"], eligible, cap, positive_only)
            sweep.append({"contacts_per_advisor": cap, "policy": name, "share_of_oracle": share(score, positive_only, cap),
                          "share_harmful_contacts": round(float((effect.loc[mask, "tau"] < 0).mean()), 4)})
    robust = pd.DataFrame(robustness)
    report = {
        "cutoffs": cfg["cutoffs"], "horizon_days": horizon, "capacity_per_advisor": capacity,
        "pilot": {"clients": len(pilot), "treated": int(pilot["treated"].sum()),
                  "observed_lift_eur_per_client": round(float(
                      pilot.loc[pilot.treated, "spend_obs"].mean() - pilot.loc[~pilot.treated, "spend_obs"].mean()), 1),
                  "true_mean_tau": round(float(pilot["true_tau"].mean()), 4)},
        "model_choice": {**choice, "selection_rule": (
            "among learners whose Qini 95% interval reaches the best Qini, the one with the most stable client "
            f"ranking across bootstrap refits; trusted only if that stability is at least {TRUST_THRESHOLD}")},
        "champion_policy": champion,
        "challenger_policy": challenger,
        "deploy": {"clients": len(deploy), "eligible": int(eligible.sum()),
                   "true_effect_correlation_by_learner": recovery,
                   "true_effect_correlation_ceiling_from_features": ceiling},
        "robustness_across_pilot_draws": {
            "draws": robustness,
            "mean_uplift_share": round(float(robust["uplift_share_of_oracle"].mean()), 3),
            "worst_uplift_share": round(float(robust["uplift_share_of_oracle"].min()), 3),
            "mean_with_trust_check": round(float(robust["with_trust_check_share_of_oracle"].mean()), 3),
            "worst_with_trust_check": round(float(robust["with_trust_check_share_of_oracle"].min()), 3),
            "status_quo_share": float(robust["status_quo_share_of_oracle"].iloc[0]),
        },
        "capacity_sweep": sweep,
        "policies": table.to_dict(orient="records"),
        "power_analysis_for_challenger_contacts": power_analysis(
            deploy[challenger_mask], effect[challenger_mask], true_value[challenger_mask]),
    }

    actions = deploy.loc[:, ["client_id", "advisor_id", "consent_status", "spend_12m", "recency_weeks"]].copy()
    actions["predicted_uplift_eur"] = scorers[chosen]
    actions["action"] = np.where(champion_mask, "contact", "no_action")
    actions["challenger_action"] = np.where(challenger_mask, "contact", "no_action")
    actions["reason"] = np.select(
        [~eligible & deploy["consent_status"].ne("consented").to_numpy(), ~eligible, champion_mask],
        ["no personalisation consent", "not active in the last two years", "among this advisor's most valuable clients"],
        default="below this advisor's capacity cut-off",
    )
    actions["predicted_harmful"] = actions["predicted_uplift_eur"] <= 0
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
