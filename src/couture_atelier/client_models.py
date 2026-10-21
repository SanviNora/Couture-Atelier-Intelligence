"""Model builders shared by the client-model experiments and the final pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
SPEND_SCALE = 1_000  # model spend in thousands of euros for a well-conditioned fit
DEFAULT_PARAMS = {"learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 20,
                  "l2_regularization": 1.0, "max_features": 1.0}
SEARCH_SPACE = {
    "learning_rate": [0.03, 0.05, 0.08, 0.12],
    "max_leaf_nodes": [15, 31, 63, 127],
    "min_samples_leaf": [20, 100, 300, 1000],
    "l2_regularization": [0.0, 0.5, 2.0, 8.0],
    "max_features": [0.5, 0.8, 1.0],
}
Predictor = Callable[[pd.DataFrame], np.ndarray]


def gbm(kind: str, params: dict, loss: str | None = None):
    common = {"max_iter": 1000, "early_stopping": True, "validation_fraction": 0.1, "n_iter_no_change": 25,
              "random_state": SEED, **params}
    if kind == "regressor":
        return HistGradientBoostingRegressor(loss=loss or "poisson", **common)
    return HistGradientBoostingClassifier(**common)


def sample_configs(n: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    return [{k: v[rng.integers(len(v))] for k, v in SEARCH_SPACE.items()} for _ in range(n)]


def fit_direct_spend(train: pd.DataFrame, features: list[str], params: dict, loss: str = "poisson") -> Predictor:
    """One model for next-year spend in euros."""
    model = gbm("regressor", params, loss)
    if loss == "squared_error":  # on log1p, rescaled so fitted totals match training totals
        target = np.log1p(train["spend_next"])
        model.fit(train[features], target)
        fitted = np.expm1(model.predict(train[features])).clip(min=0)
        smear = float(np.expm1(target).sum() / fitted.sum())
        return lambda frame: np.expm1(model.predict(frame[features])).clip(min=0) * smear
    model.fit(train[features], train["spend_next"] / SPEND_SCALE)
    ceiling = float(train["spend_next"].max())
    return lambda frame: np.clip(model.predict(frame[features]) * SPEND_SCALE, 0, ceiling)


def fit_hurdle_spend(train: pd.DataFrame, features: list[str], params: dict, loss: str = "gamma") -> Predictor:
    """P(returns) x E[spend | returns]: suits spend that is zero for most clients."""
    classifier = gbm("classifier", params).fit(train[features], train["returned_next"])
    buyers = train[train["returned_next"]]
    regressor = gbm("regressor", params, loss).fit(buyers[features], buyers["spend_next"] / SPEND_SCALE)
    ceiling = float(buyers["spend_next"].max())

    def predict(frame: pd.DataFrame) -> np.ndarray:
        p = classifier.predict_proba(frame[features])[:, 1]
        return p * np.clip(regressor.predict(frame[features]) * SPEND_SCALE, 0, ceiling)

    return predict


STRUCTURES = {
    "gbm_poisson": (fit_direct_spend, "poisson"),
    "gbm_log_squared": (fit_direct_spend, "squared_error"),
    "hurdle_gamma": (fit_hurdle_spend, "gamma"),
    "hurdle_poisson": (fit_hurdle_spend, "poisson"),
}


def fit_spend(structure: str, train: pd.DataFrame, features: list[str], params: dict) -> Predictor:
    maker, loss = STRUCTURES[structure]
    return maker(train, features, params, loss)


def fit_return(model: str, train: pd.DataFrame, features: list[str], params: dict) -> Predictor:
    """Probability a client returns within the next 12 months."""
    if model == "logistic":
        def transform(frame: pd.DataFrame) -> pd.DataFrame:
            return np.log1p(frame[features].clip(lower=0))

        pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        pipeline.fit(transform(train), train["returned_next"])
        return lambda frame: pipeline.predict_proba(transform(frame))[:, 1]
    classifier = gbm("classifier", params).fit(train[features], train["returned_next"])
    return lambda frame: classifier.predict_proba(frame[features])[:, 1]
