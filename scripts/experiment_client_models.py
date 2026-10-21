"""Model selection for client value, return and VIC upgrade, using pre-test data only.

Training rows: clients at four 2017 cut-offs (outcome windows all end before
2019). Validation: a fixed 20% of clients, scored at the 2017-12-31 cut-off and
never used for training. The 2019 test window is not touched here; the final
pipeline (``build_client_value.py``) refits the selected setup and tests once.

Every configuration tried is logged to ``reports/client_experiments.csv``; the
winners are written to ``configs/client_models.yml``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from couture_atelier.client_frames import (
    FEATURE_SETS,
    VIC_SHARE,
    add_probabilistic,
    label_outcomes,
    spend_metrics,
    window_frame,
)
from couture_atelier.client_models import DEFAULT_PARAMS as DEFAULT
from couture_atelier.client_models import (
    fit_direct_spend,
    fit_hurdle_spend,
    fit_return,
    gbm,
    sample_configs,
)
from couture_atelier.luxury import stable_uniform
from couture_atelier.paths import CONFIGS, PROCESSED, REPORTS, WAREHOUSE

ORIGINS = ["2017-06-30", "2017-08-31", "2017-10-31", "2017-12-31"]
VALIDATION_ORIGIN = "2017-12-31"
SPEARMAN_TIE = 0.005  # spend models this close on ranking count as tied; then prefer top-1% precision
AUC_TIE = 0.002  # if a simpler return model is this close to the best, use the simpler one
CACHE = PROCESSED / "client_frames"


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_frame(con: duckdb.DuckDBPyConnection, cutoff: str) -> pd.DataFrame:
    path = CACHE / f"{cutoff}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    frame, _ = add_probabilistic(window_frame(con, cutoff))
    frame = label_outcomes(frame)
    CACHE.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def run(warehouse: Path = WAREHOUSE) -> dict:
    with duckdb.connect(str(warehouse), read_only=True) as con:
        frames = {cutoff: load_frame(con, cutoff) for cutoff in ORIGINS}
    log("frames ready: " + ", ".join(f"{c}={len(f):,}" for c, f in frames.items()))

    def is_val(frame: pd.DataFrame) -> np.ndarray:
        return stable_uniform(frame["client_id"], seed=7) < 0.2

    pooled = pd.concat([f[~is_val(f)] for f in frames.values()], ignore_index=True)
    single = frames[VALIDATION_ORIGIN][~is_val(frames[VALIDATION_ORIGIN])]
    val = frames[VALIDATION_ORIGIN][is_val(frames[VALIDATION_ORIGIN])].reset_index(drop=True)
    log(f"pooled train {len(pooled):,} rows, single-origin train {len(single):,}, validation {len(val):,}")
    rows: list[dict] = []

    # ---- Next-year spend -------------------------------------------------------------
    actual = val["spend_next"]
    spend_preds: dict[str, np.ndarray] = {
        "naive_last_12m": val["spend_12m"].to_numpy(),
        "bgnbd_gg": val["pred_spend_bg"].to_numpy(),
        "mbgnbd_gg": val["pred_spend_mbg"].to_numpy(),
    }
    for name, pred in spend_preds.items():
        rows.append({"task": "spend", "model": name, **spend_metrics(actual, pred)})
    for data_name, data in (("single", single), ("pooled", pooled)):
        for fs_name, features in FEATURE_SETS.items():
            name = f"gbm_poisson|{data_name}|{fs_name}"
            spend_preds[name] = fit_direct_spend(data, features, DEFAULT)(val)
            rows.append({"task": "spend", "model": name, **spend_metrics(actual, spend_preds[name])})
            log(f"spend {name}: {rows[-1]['spearman']}")
    full = FEATURE_SETS["full"]
    for name, maker, loss in (
        ("gbm_log_squared|pooled|full", fit_direct_spend, "squared_error"),
        ("hurdle_gamma|pooled|full", fit_hurdle_spend, "gamma"),
        ("hurdle_poisson|pooled|full", fit_hurdle_spend, "poisson"),
    ):
        spend_preds[name] = maker(pooled, full, DEFAULT, loss)(val)
        rows.append({"task": "spend", "model": name, **spend_metrics(actual, spend_preds[name])})
        log(f"spend {name}: {rows[-1]['spearman']}")

    def pick_spend(candidates: list[dict]) -> dict:
        """Best ranking, with near-ties broken by how many of next year's VICs are found."""
        top = max(r["spearman"] for r in candidates)
        tied = [r for r in candidates if r["spearman"] >= top - SPEARMAN_TIE]
        return max(tied, key=lambda r: (r["top1pct_precision"], r["spearman"]))

    structure_rows = [r for r in rows if r["task"] == "spend" and r["model"].endswith("|pooled|full")]
    best_structure = pick_spend(structure_rows)["model"]
    structure = best_structure.split("|")[0]
    log(f"tuning spend structure {structure}")
    maker = fit_hurdle_spend if structure.startswith("hurdle") else fit_direct_spend
    loss = {"gbm_poisson": "poisson", "gbm_log_squared": "squared_error",
            "hurdle_gamma": "gamma", "hurdle_poisson": "poisson"}[structure]
    params_by_model = {best_structure: DEFAULT}
    for i, params in enumerate(sample_configs(16, seed=1)):
        name = f"{structure}|pooled|full|search{i:02d}"
        spend_preds[name] = maker(pooled, full, params, loss)(val)
        metrics = spend_metrics(actual, spend_preds[name])
        rows.append({"task": "spend", "model": name, "params": json.dumps(params), **metrics})
        params_by_model[name] = params
        log(f"spend {name}: {metrics['spearman']} top1 {metrics['top1pct_precision']} {params}")
    chosen = pick_spend([r for r in rows if r["model"] in params_by_model])
    best_spend = {"model": chosen["model"], "params": params_by_model[chosen["model"]],
                  "spearman": chosen["spearman"], "top1pct_precision": chosen["top1pct_precision"]}
    best_ml_pred = spend_preds[best_spend["model"]]
    for weight in (0.25, 0.5, 0.75):
        blend_rank = weight * rankdata(best_ml_pred) + (1 - weight) * rankdata(spend_preds["mbgnbd_gg"])
        rows.append({"task": "spend", "model": f"rank_blend|ml_weight={weight}",
                     **spend_metrics(actual, blend_rank / len(blend_rank) * actual.mean() * 2)})
    recalibration = float(actual.sum() / best_ml_pred.sum())
    rows.append({"task": "spend", "model": f"{best_spend['model']}|recalibrated",
                 **spend_metrics(actual, best_ml_pred * recalibration)})

    # ---- Returns next year ----------------------------------------------------------
    returned = val["returned_next"]

    def return_row(name: str, score: np.ndarray, **extra) -> dict:
        row = {"task": "return", "model": name, "auc": round(float(roc_auc_score(returned, score)), 4), **extra}
        if score.min() >= 0 and score.max() <= 1:
            row["brier"] = round(float(brier_score_loss(returned, score)), 4)
        return row

    rows.append(return_row("recency_only", -val["recency_weeks"].to_numpy()))
    rows.append(return_row("bgnbd_p_alive", val["p_alive_bg"].to_numpy()))
    rows.append(return_row("mbgnbd_p_alive", val["p_alive_mbg"].to_numpy()))
    rows.append(return_row("logistic|pooled|full", fit_return("logistic", pooled, full, {})(val)))
    for data_name, data in (("single", single), ("pooled", pooled)):
        for fs_name, features in FEATURE_SETS.items():
            model = gbm("classifier", DEFAULT).fit(data[features], data["returned_next"])
            rows.append(return_row(f"gbm|{data_name}|{fs_name}", model.predict_proba(val[features])[:, 1]))
            log(f"return {rows[-1]['model']}: {rows[-1]['auc']}")
    best_gbm = {"params": DEFAULT, "auc": next(r["auc"] for r in rows if r["model"] == "gbm|pooled|full")}
    for i, params in enumerate(sample_configs(12, seed=2)):
        model = gbm("classifier", params).fit(pooled[full], pooled["returned_next"])
        row = return_row(f"gbm|pooled|full|search{i:02d}", model.predict_proba(val[full])[:, 1],
                         params=json.dumps(params))
        rows.append(row)
        log(f"return {row['model']}: {row['auc']}")
        if row["auc"] > best_gbm["auc"]:
            best_gbm = {"params": params, "auc": row["auc"]}
    logistic_auc = next(r["auc"] for r in rows if r["model"] == "logistic|pooled|full")
    if logistic_auc >= best_gbm["auc"] - AUC_TIE:
        best_return = {"model": "logistic", "params": {}, "auc": logistic_auc,
                       "note": f"within {AUC_TIE} AUC of the best gradient boosting ({best_gbm['auc']}); simpler model kept"}
    else:
        best_return = {"model": "gbm", **best_gbm}

    # ---- VIC upgrade among clients who are not VICs yet -----------------------------
    pool_train = pooled[~pooled["is_vic_now"]]
    pool_val = val[~val["is_vic_now"]]
    upgraded = pool_val["upgraded"]
    k = int(upgraded.sum())

    def upgrade_row(name: str, score: np.ndarray, **extra) -> dict:
        top = np.argsort(-score)[:k]
        return {"task": "vic_upgrade", "model": name,
                "auc": round(float(roc_auc_score(upgraded, score)), 4),
                "avg_precision": round(float(average_precision_score(upgraded, score)), 4),
                "precision_at_k": round(float(upgraded.to_numpy()[top].mean()), 4), "positives": k, **extra}

    rows.append(upgrade_row("naive_spend_12m", pool_val["spend_12m"].to_numpy()))
    rows.append(upgrade_row("naive_spend_90d", pool_val["spend_90d"].to_numpy()))
    rows.append(upgrade_row("best_spend_model", best_ml_pred[~val["is_vic_now"].to_numpy()]))
    for fs_name, features in FEATURE_SETS.items():
        for weighting in ("none", "balanced"):
            params = {**DEFAULT, "class_weight": None if weighting == "none" else "balanced"}
            model = gbm("classifier", params).fit(pool_train[features], pool_train["upgraded"])
            rows.append(upgrade_row(f"gbm|pooled|{fs_name}|weights={weighting}",
                                    model.predict_proba(pool_val[features])[:, 1]))
            log(f"upgrade {rows[-1]['model']}: AP {rows[-1]['avg_precision']}")
    best_upgrade = max((r for r in rows if r["task"] == "vic_upgrade" and r["model"].startswith("gbm")),
                       key=lambda r: r["avg_precision"])
    upgrade_features = best_upgrade["model"].split("|")[2]
    upgrade_weights = best_upgrade["model"].split("weights=")[1]
    best_upgrade_cfg = {"feature_set": upgrade_features, "class_weight": upgrade_weights, "params": DEFAULT,
                        "avg_precision": best_upgrade["avg_precision"]}
    for i, params in enumerate(sample_configs(12, seed=3)):
        full_params = {**params, "class_weight": None if upgrade_weights == "none" else "balanced"}
        features = FEATURE_SETS[upgrade_features]
        model = gbm("classifier", full_params).fit(pool_train[features], pool_train["upgraded"])
        row = upgrade_row(f"gbm|pooled|{upgrade_features}|search{i:02d}", model.predict_proba(pool_val[features])[:, 1],
                          params=json.dumps(params))
        rows.append(row)
        log(f"upgrade {row['model']}: AP {row['avg_precision']}")
        if row["avg_precision"] > best_upgrade_cfg["avg_precision"]:
            best_upgrade_cfg.update(params=params, avg_precision=row["avg_precision"])

    # ---- Save ----------------------------------------------------------------------
    REPORTS.mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(REPORTS / "client_experiments.csv", index=False)
    spend_model = best_spend["model"]
    selection = {
        "selected_on": f"validation clients at {VALIDATION_ORIGIN}; 2019 test not used",
        "training_origins": ORIGINS,
        "spend": {"structure": structure, "loss": loss, "feature_set": "full",
                  "params": best_spend["params"], "validation_spearman": best_spend["spearman"],
                  "validation_top1pct_precision": best_spend["top1pct_precision"],
                  "selection_rule": f"best Spearman; ties within {SPEARMAN_TIE} broken by top-1% precision",
                  "validation_recalibration_factor": round(recalibration, 4), "selected_model": spend_model},
        "return": {"model": best_return["model"], "feature_set": "full", "params": best_return["params"],
                   "validation_auc": best_return["auc"], "note": best_return.get("note", "")},
        "vic_upgrade": {"feature_set": upgrade_features,
                        "class_weight": None if upgrade_weights == "none" else "balanced",
                        "params": best_upgrade_cfg["params"],
                        "validation_avg_precision": best_upgrade_cfg["avg_precision"]},
        "vic_share": VIC_SHARE,
    }
    (CONFIGS / "client_models.yml").write_text(
        "# Written by scripts/experiment_client_models.py. Do not edit by hand.\n"
        + yaml.safe_dump(json.loads(json.dumps(selection, default=float)), sort_keys=False)
    )
    log("done")
    return selection


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, default=str))
