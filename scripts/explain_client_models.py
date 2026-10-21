"""What drives each selected client model?

Permutation importance on a fixed sample of 2019 test clients: shuffle one
feature at a time and measure how much the model's test metric drops (spend:
Spearman; VIC upgrade: average precision; return: AUC). The logistic return
model also reports its standardised coefficients.

Reads the cached client frames written by ``build_client_value.py``.
Writes ``reports/client_feature_importance.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from couture_atelier.client_frames import FEATURE_SETS
from couture_atelier.client_models import fit_return, fit_spend, gbm
from couture_atelier.paths import CONFIGS, PROCESSED, REPORTS

TEST = "2018-12-31"
SAMPLE = 60_000
REPEATS = 3
SEED = 42


def permutation_drop(predict, frame: pd.DataFrame, features: list[str], metric) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = metric(predict(frame))
    rows = []
    for feature in features:
        drops = []
        for _ in range(REPEATS):
            shuffled = frame.copy()
            shuffled[feature] = rng.permutation(shuffled[feature].to_numpy())
            drops.append(base - metric(predict(shuffled)))
        rows.append({"feature": feature, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def run() -> pd.DataFrame:
    selection = yaml.safe_load((CONFIGS / "client_models.yml").read_text())
    cache = PROCESSED / "client_frames"
    train = pd.concat([pd.read_parquet(cache / f"{c}.parquet") for c in selection["training_origins"]],
                      ignore_index=True)
    test = pd.read_parquet(cache / f"{TEST}.parquet").sample(SAMPLE, random_state=SEED).reset_index(drop=True)
    tables = []

    s_cfg = selection["spend"]
    s_features = FEATURE_SETS[s_cfg["feature_set"]]
    spend = fit_spend(s_cfg["structure"], train, s_features, s_cfg["params"])
    table = permutation_drop(spend, test, s_features,
                             lambda pred: spearmanr(pred, test["spend_next"]).statistic)
    tables.append(table.assign(model="spend (hurdle)", metric="Spearman drop"))

    r_cfg = selection["return"]
    r_features = FEATURE_SETS[r_cfg["feature_set"]]
    returner = fit_return(r_cfg["model"], train, r_features, r_cfg["params"])
    table = permutation_drop(returner, test, r_features, lambda p: roc_auc_score(test["returned_next"], p))
    tables.append(table.assign(model=f"return ({r_cfg['model']})", metric="AUC drop"))

    u_cfg = selection["vic_upgrade"]
    u_features = FEATURE_SETS[u_cfg["feature_set"]]
    pool_train = train[~train["is_vic_now"]]
    pool_test = test[~test["is_vic_now"]].reset_index(drop=True)
    upgrader = gbm("classifier", {**u_cfg["params"], "class_weight": u_cfg["class_weight"]})
    upgrader.fit(pool_train[u_features], pool_train["upgraded"])
    table = permutation_drop(lambda f: upgrader.predict_proba(f[u_features])[:, 1], pool_test, u_features,
                             lambda p: average_precision_score(pool_test["upgraded"], p))
    tables.append(table.assign(model="VIC upgrade (gbm)", metric="avg. precision drop"))

    result = pd.concat(tables, ignore_index=True)
    result["importance"] = result["importance"].round(4)
    result["std"] = result["std"].round(4)
    REPORTS.mkdir(exist_ok=True)
    result.to_csv(REPORTS / "client_feature_importance.csv", index=False)
    return result


if __name__ == "__main__":
    out = run()
    for model, group in out.groupby("model"):
        print(f"\n{model}\n" + group.head(8)[["feature", "importance", "std"]].to_string(index=False))
