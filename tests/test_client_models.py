import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from couture_atelier.client_models import fit_return, fit_spend, sample_configs
from couture_atelier.evaluation import paired_bootstrap


def toy_clients(n: int = 4_000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    visits = rng.poisson(2, n)
    returned = rng.random(n) < 1 / (1 + np.exp(-(visits - 2)))
    spend = np.where(returned, rng.gamma(2, 500, n) * (1 + visits), 0.0)
    return pd.DataFrame({"visits_12m": visits, "noise": rng.normal(size=n),
                         "returned_next": returned, "spend_next": spend})


def test_bootstrap_flags_a_real_improvement_and_not_noise() -> None:
    rng = np.random.default_rng(1)
    y = rng.random(5_000) < 0.4
    good = y + rng.normal(0, 0.8, 5_000)
    random_a, random_b = rng.random(5_000), rng.random(5_000)

    real = paired_bootstrap(roc_auc_score, y, good, random_a, n_boot=100)
    noise = paired_bootstrap(roc_auc_score, y, random_a, random_b, n_boot=100)

    assert real["significant"] and real["ci_low"] > 0
    assert not noise["significant"]


def test_return_models_learn_signal() -> None:
    clients = toy_clients()
    for model in ("logistic", "gbm"):
        predict = fit_return(model, clients, ["visits_12m", "noise"], {"max_iter": 50} if model == "gbm" else {})
        probabilities = predict(clients)
        assert ((probabilities >= 0) & (probabilities <= 1)).all()
        assert roc_auc_score(clients["returned_next"], probabilities) > 0.7


def test_hurdle_spend_is_non_negative_and_ordered_by_signal() -> None:
    clients = toy_clients()
    predict = fit_spend("hurdle_gamma", clients, ["visits_12m", "noise"], {"max_iter": 50})
    pred = predict(clients)

    assert (pred >= 0).all()
    assert pred[clients["visits_12m"] >= 4].mean() > pred[clients["visits_12m"] <= 1].mean()


def test_search_configs_are_reproducible() -> None:
    assert sample_configs(5, seed=3) == sample_configs(5, seed=3)
