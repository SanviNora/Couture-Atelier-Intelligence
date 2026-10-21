"""Clienteling: who an advisor should contact, judged by the sales a contact adds.

Real data gives each client's behaviour with no contact. The effect of a
contact is simulated from a documented rule (``configs/clienteling.yml``), so
uplift models can be judged against the true effect.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from couture_atelier.client_models import SPEND_SCALE, gbm
from couture_atelier.luxury import stable_uniform

Scorer = Callable[[pd.DataFrame], np.ndarray]


def contact_effect(frame: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """True effect of one contact: extra-visit probability ``tau`` and the value of that visit.

    The client-level noise is a fixed trait (keyed by client id), so the same
    client has the same unexplained responsiveness at every cut-off.
    """
    c = cfg["contact_effect"]
    engagement = frame["lifetime_visits"].clip(upper=c["engagement_visits_cap"]) / c["engagement_visits_cap"]
    drift = (frame["recency_weeks"] / (2 * c["persuadable_recency_weeks"])).clip(0, 1)
    persuadable = engagement * 4 * drift * (1 - drift)
    top_clients = frame["spend_12m"] >= frame["spend_12m"].quantile(0.99)
    do_not_disturb = top_clients & (frame["recency_weeks"] < c["do_not_disturb_weeks"])
    trait = norm.ppf(np.clip(stable_uniform(frame["client_id"], c["seed"]), 1e-6, 1 - 1e-6))
    tau = (c["base"] + c["persuadable_peak"] * persuadable - c["do_not_disturb_penalty"] * do_not_disturb
           + c["individual_noise_sd"] * trait)
    visit_value = frame["mean_visit_value_12m"].where(frame["mean_visit_value_12m"] > 0, frame["last_visit_value"])
    return pd.DataFrame({
        "tau": tau.clip(lower=c["floor"], upper=0.5),
        "visit_value_eur": c["basket_share_of_mean_visit"] * visit_value,
        "do_not_disturb": do_not_disturb,
        "persuadable": persuadable,
        "trait": trait,
    }, index=frame.index)


def true_incremental_value(effect: pd.DataFrame, spend_no_contact: pd.Series) -> pd.Series:
    """Expected euros a contact adds (negative for do-not-disturb clients)."""
    gain = effect["tau"] * effect["visit_value_eur"]
    loss = effect["tau"] * np.minimum(spend_no_contact, effect["visit_value_eur"])
    return gain.where(effect["tau"] >= 0, loss)


def simulate_contacted_spend(
    effect: pd.DataFrame, spend_no_contact: pd.Series, seed: int
) -> pd.Series:
    """Spend over the horizon if contacted: one extra visit with probability tau,
    or, for a negative tau, the loss of one visit's value with probability |tau|."""
    rng = np.random.default_rng(seed)
    draw = rng.random(len(effect)) < effect["tau"].abs().to_numpy()
    extra = np.where(effect["tau"] >= 0, draw * effect["visit_value_eur"], 0.0)
    lost = np.where(effect["tau"] < 0, draw * np.minimum(spend_no_contact, effect["visit_value_eur"]), 0.0)
    return spend_no_contact + extra - lost


def fit_t_learner(pilot: pd.DataFrame, features: list[str], outcome: str, params: dict) -> Scorer:
    """Separate outcome models for contacted and not-contacted clients; uplift is the difference."""
    loss = "poisson" if outcome.startswith("spend") else None
    scale = SPEND_SCALE if outcome.startswith("spend") else 1.0
    treated, control = pilot[pilot["treated"]], pilot[~pilot["treated"]]
    if loss:
        mu1 = gbm("regressor", params, loss).fit(treated[features], treated[outcome] / scale)
        mu0 = gbm("regressor", params, loss).fit(control[features], control[outcome] / scale)
        return lambda f: (mu1.predict(f[features]) - mu0.predict(f[features])) * scale
    mu1 = gbm("classifier", params).fit(treated[features], treated[outcome])
    mu0 = gbm("classifier", params).fit(control[features], control[outcome])
    return lambda f: mu1.predict_proba(f[features])[:, 1] - mu0.predict_proba(f[features])[:, 1]


def fit_t_learner_logistic(pilot: pd.DataFrame, features: list[str], outcome: str) -> Scorer:
    """T-learner with two logistic regressions on log-scaled features (binary outcome)."""
    def transform(frame: pd.DataFrame) -> pd.DataFrame:
        return np.log1p(frame[features].clip(lower=0))

    treated, control = pilot[pilot["treated"]], pilot[~pilot["treated"]]
    mu1 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(transform(treated), treated[outcome])
    mu0 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(transform(control), control[outcome])
    return lambda f: mu1.predict_proba(transform(f))[:, 1] - mu0.predict_proba(transform(f))[:, 1]


def fit_s_learner_logistic(pilot: pd.DataFrame, features: list[str], outcome: str, strength: float = 0.1) -> Scorer:
    """One regularised logistic model with treatment and treatment x feature interactions.

    Sharing one model between contacted and not-contacted clients makes the
    uplift estimate less sensitive to noise than the difference of two models.
    """
    scaler = StandardScaler().fit(np.log1p(pilot[features].clip(lower=0)))

    def design(frame: pd.DataFrame, treated: np.ndarray) -> np.ndarray:
        x = scaler.transform(np.log1p(frame[features].clip(lower=0)))
        return np.hstack([x, treated[:, None], x * treated[:, None]])

    model = LogisticRegression(C=strength, max_iter=3000).fit(
        design(pilot, pilot["treated"].to_numpy(dtype=float)), pilot[outcome])
    return lambda f: (model.predict_proba(design(f, np.ones(len(f))))[:, 1]
                      - model.predict_proba(design(f, np.zeros(len(f))))[:, 1])


def bagged(make: Callable[[pd.DataFrame], Scorer], copies: int = 10, seed: int = 0) -> Callable[[pd.DataFrame], Scorer]:
    """Average a learner over bootstrap resamples of the pilot to reduce its variance."""
    def fit(pilot: pd.DataFrame) -> Scorer:
        rng = np.random.default_rng(seed)
        scorers = [make(pilot.iloc[rng.integers(0, len(pilot), len(pilot))]) for _ in range(copies)]
        return lambda f: np.mean([s(f) for s in scorers], axis=0)
    return fit


def uplift_stability(make: Callable[[pd.DataFrame], Scorer], pilot: pd.DataFrame, refits: int = 4,
                     probe: int = 20_000, seed: int = 1) -> float:
    """Pilot-only trust check: how consistently the learner ranks clients across bootstrap refits.

    Mean pairwise Spearman correlation of predicted uplift on a fixed sample of
    pilot clients. Low agreement means the ranking mostly reflects noise in
    this particular pilot.
    """
    rng = np.random.default_rng(seed)
    sample = pilot.sample(min(probe, len(pilot)), random_state=0)
    preds = [make(pilot.iloc[rng.integers(0, len(pilot), len(pilot))])(sample) for _ in range(refits)]
    pairs = [spearmanr(preds[i], preds[j]).statistic for i in range(refits) for j in range(i + 1, refits)]
    return float(np.mean(pairs))


def fit_transformed_outcome(pilot: pd.DataFrame, features: list[str], outcome: str, params: dict) -> Scorer:
    """Regress Y * (T - e) / (e (1 - e)), whose expectation is the uplift, on the features."""
    share = pilot["treated"].mean()
    target = pilot[outcome].astype(float) * (pilot["treated"] - share) / (share * (1 - share))
    model = gbm("regressor", params, "squared_error").fit(pilot[features], target)
    return lambda f: model.predict(f[features])


def fit_x_learner(pilot: pd.DataFrame, features: list[str], outcome: str, params: dict) -> Scorer:
    """X-learner (Kunzel et al. 2019) with a known 50/50 assignment probability."""
    treated, control = pilot[pilot["treated"]], pilot[~pilot["treated"]]
    scale = SPEND_SCALE if outcome.startswith("spend") else 1.0
    mu1 = gbm("regressor", params, "poisson").fit(treated[features], treated[outcome] / scale)
    mu0 = gbm("regressor", params, "poisson").fit(control[features], control[outcome] / scale)
    d1 = treated[outcome] / scale - mu0.predict(treated[features])
    d0 = mu1.predict(control[features]) - control[outcome] / scale
    tau1 = gbm("regressor", params, "squared_error").fit(treated[features], d1)
    tau0 = gbm("regressor", params, "squared_error").fit(control[features], d0)
    share = pilot["treated"].mean()
    return lambda f: (share * tau0.predict(f[features]) + (1 - share) * tau1.predict(f[features])) * scale


def allocate(scores: np.ndarray, advisor: pd.Series, eligible: np.ndarray, capacity: int,
             positive_only: bool = False) -> np.ndarray:
    """Each advisor contacts their highest-scoring eligible clients, up to capacity.

    With ``positive_only`` a client is never contacted when the score says the
    contact would not help, even if the advisor has capacity left.
    """
    allowed = eligible & (scores > 0 if positive_only else True)
    frame = pd.DataFrame({"advisor": advisor.to_numpy(), "score": scores, "allowed": allowed})
    frame = frame[frame["allowed"]]
    rank = frame.groupby("advisor")["score"].rank(ascending=False, method="first")
    chosen = np.zeros(len(scores), dtype=bool)
    chosen[frame.index[rank <= capacity]] = True
    return chosen


def qini_area(uplift_score: np.ndarray, treated: np.ndarray, outcome: np.ndarray, bins: int = 20) -> float:
    """Area between the model's Qini curve and random targeting, on observed outcomes.

    Clients are sorted by predicted uplift; at each cut the curve is the
    treated-minus-control outcome difference scaled to the treated count.
    """
    order = np.argsort(-uplift_score)
    t, y = treated[order], outcome[order]
    n = len(y)
    cuts = np.linspace(0, n, bins + 1).astype(int)[1:]
    curve = []
    for cut in cuts:
        tt, yy = t[:cut], y[:cut]
        n_t, n_c = tt.sum(), (~tt).sum()
        if n_t == 0 or n_c == 0:
            curve.append(0.0)
            continue
        curve.append(yy[tt].sum() - yy[~tt].sum() * n_t / n_c)
    curve = np.array(curve)
    random_line = curve[-1] * cuts / n
    return float(np.trapezoid(curve - random_line, cuts / n))
