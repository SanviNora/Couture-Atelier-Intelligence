import numpy as np
import pandas as pd
import pytest
import yaml

from couture_atelier.clienteling import (
    allocate,
    bagged,
    contact_effect,
    fit_s_learner_logistic,
    fit_t_learner_logistic,
    qini_area,
    simulate_contacted_spend,
    true_incremental_value,
    uplift_stability,
)
from couture_atelier.paths import CONFIGS

CFG = yaml.safe_load((CONFIGS / "clienteling.yml").read_text())


def clients(n: int = 2_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "client_id": np.arange(n),
        "lifetime_visits": rng.integers(1, 20, n),
        "recency_weeks": rng.uniform(0, 60, n),
        "spend_12m": rng.lognormal(8, 1, n),
        "mean_visit_value_12m": rng.lognormal(7.5, 0.5, n),
        "last_visit_value": rng.lognormal(7.5, 0.5, n),
    })


def test_contact_helps_most_for_engaged_clients_who_are_drifting() -> None:
    frame = pd.DataFrame({
        "client_id": [1, 2, 3], "lifetime_visits": [10, 10, 10], "recency_weeks": [1.0, 13.0, 40.0],
        "spend_12m": [100.0, 100.0, 100.0], "mean_visit_value_12m": [500.0] * 3, "last_visit_value": [500.0] * 3,
    })
    no_noise = {**CFG, "contact_effect": {**CFG["contact_effect"], "individual_noise_sd": 0.0}}

    tau = contact_effect(frame, no_noise)["tau"]

    assert tau.iloc[1] > tau.iloc[0] and tau.iloc[1] > tau.iloc[2]


def test_recent_top_clients_are_do_not_disturb() -> None:
    frame = clients()
    frame.loc[0, ["spend_12m", "recency_weeks"]] = [1e9, 1.0]
    no_noise = {**CFG, "contact_effect": {**CFG["contact_effect"], "individual_noise_sd": 0.0}}

    effect = contact_effect(frame, no_noise)

    assert effect.loc[0, "do_not_disturb"]
    assert effect.loc[0, "tau"] < 0


def test_client_trait_is_stable_across_cut_offs() -> None:
    first, later = clients(seed=1), clients(seed=2)

    assert np.allclose(contact_effect(first, CFG)["trait"], contact_effect(later, CFG)["trait"])


def test_simulated_contacts_add_the_true_expected_value() -> None:
    frame = clients(200_000)
    effect = contact_effect(frame, CFG)
    no_contact = pd.Series(np.where(np.arange(len(frame)) % 2 == 0, 0.0, 800.0))

    contacted = simulate_contacted_spend(effect, no_contact, seed=4)

    expected = true_incremental_value(effect, no_contact).mean()
    assert (contacted - no_contact).mean() == pytest.approx(expected, rel=0.05)


def test_allocation_respects_capacity_eligibility_and_no_harm_rule() -> None:
    advisors = pd.Series(["A"] * 5 + ["B"] * 5)
    scores = np.array([5, 4, 3, -1, 2, 9, 8, -2, -3, 7], dtype=float)
    eligible = np.array([True, True, False, True, True, True, True, True, True, False])

    chosen = allocate(scores, advisors, eligible, capacity=2)
    cautious = allocate(scores, advisors, eligible, capacity=4, positive_only=True)

    assert chosen.tolist() == [True, True, False, False, False, True, True, False, False, False]
    assert not cautious[[3, 7, 8]].any()
    assert not cautious[[2, 9]].any()


def test_qini_rewards_a_scorer_that_knows_the_uplift() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    uplift = rng.uniform(0, 0.3, n)
    treated = rng.random(n) < 0.5
    outcome = (rng.random(n) < 0.2 + treated * uplift).astype(float)

    assert qini_area(uplift, treated, outcome) > qini_area(rng.random(n), treated, outcome)


def test_logistic_t_learner_finds_who_responds() -> None:
    rng = np.random.default_rng(3)
    n = 40_000
    frame = pd.DataFrame({"engaged": rng.integers(0, 2, n).astype(float), "noise": rng.normal(size=n) ** 2})
    frame["treated"] = rng.random(n) < 0.5
    p = 0.2 + frame["treated"] * frame["engaged"] * 0.2
    frame["visit_obs"] = rng.random(n) < p

    uplift = fit_t_learner_logistic(frame, ["engaged", "noise"], "visit_obs")(frame)

    assert uplift[frame["engaged"] == 1].mean() > uplift[frame["engaged"] == 0].mean() + 0.1


def responsive_pilot(n: int = 40_000, effect: float = 0.2, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({"engaged": rng.integers(0, 2, n).astype(float), "noise": rng.normal(size=n) ** 2})
    frame["treated"] = rng.random(n) < 0.5
    frame["visit_obs"] = rng.random(n) < 0.2 + frame["treated"] * frame["engaged"] * effect
    frame["client_id"] = np.arange(n)
    return frame


def test_s_learner_and_bagging_find_who_responds() -> None:
    pilot = responsive_pilot()
    for make in (lambda d: fit_s_learner_logistic(d, ["engaged", "noise"], "visit_obs"),
                 bagged(lambda d: fit_t_learner_logistic(d, ["engaged", "noise"], "visit_obs"), copies=3)):
        uplift = make(pilot)(pilot)
        assert uplift[pilot["engaged"] == 1].mean() > uplift[pilot["engaged"] == 0].mean() + 0.1


def test_stability_is_high_for_real_signal_and_low_for_noise() -> None:
    def make(d: pd.DataFrame):
        return fit_t_learner_logistic(d, ["engaged", "noise"], "visit_obs")

    signal = uplift_stability(make, responsive_pilot(effect=0.2), refits=3, probe=5_000)
    noise = uplift_stability(make, responsive_pilot(effect=0.0), refits=3, probe=5_000)

    assert signal > 0.8
    assert noise < signal
