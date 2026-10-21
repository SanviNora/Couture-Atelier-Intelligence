"""Is a bigger clienteling pilot worth funding?

Re-runs the uplift pipeline with pilots of 25%, 50% and 100% of one quarter's
eligible clients, and 200% (two quarters pooled: March and June 2018). For each
size and three random draws, it measures how well the uplift model recovers the
true contact effect at deployment and how much incremental revenue its policy
captures at 40 and 80 contacts per advisor.

Writes ``reports/pilot_size_value.csv``.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd
import yaml
from scripts.build_clienteling import (
    CORE,
    FEATURES,
    PARAMS,
    eligible_mask,
    frame_at,
    run_pilot,
    visit_value_proxy,
)

from couture_atelier.clienteling import (
    allocate,
    contact_effect,
    fit_t_learner,
    fit_t_learner_logistic,
    true_incremental_value,
)
from couture_atelier.luxury import stable_uniform
from couture_atelier.paths import CONFIGS, REPORTS, WAREHOUSE

EXTRA_PILOT = "2018-03-31"
SIZES = (0.25, 0.5, 1.0, 2.0)
DRAWS = 3


def run() -> pd.DataFrame:
    cfg = yaml.safe_load((CONFIGS / "clienteling.yml").read_text())
    with duckdb.connect(str(WAREHOUSE), read_only=True) as con:
        june = frame_at(con, cfg["cutoffs"]["pilot"], cfg["horizon_days"])
        march = frame_at(con, EXTRA_PILOT, cfg["horizon_days"])
        deploy = frame_at(con, cfg["cutoffs"]["deploy"], cfg["horizon_days"])
    effect = contact_effect(deploy, cfg)
    true_value = true_incremental_value(effect, deploy["spend_h"])
    eligible = eligible_mask(deploy, cfg)
    value_at_stake = visit_value_proxy(deploy)
    oracle = {cap: true_value[allocate(true_value.to_numpy(), deploy["advisor_id"], eligible, cap, True)].sum()
              for cap in (40, 80)}

    rows = []
    for size in SIZES:
        for draw in range(DRAWS):
            seed_cfg = {**cfg, "pilot": {**cfg["pilot"], "seed": cfg["pilot"]["seed"] + 100 * draw}}
            if size <= 1.0:
                keep = stable_uniform(june["client_id"], seed=50 + draw) < size
                pilot = run_pilot(june[keep], seed_cfg)
            else:
                pilot = pd.concat([run_pilot(june, seed_cfg),
                                   run_pilot(march, {**seed_cfg, "pilot": {**seed_cfg["pilot"],
                                                                           "seed": seed_cfg["pilot"]["seed"] + 7}})],
                                  ignore_index=True)
            for learner, fit in (("t_logistic_core", lambda d: fit_t_learner_logistic(d, CORE, "visit_obs")),
                                 ("t_gbm_full", lambda d: fit_t_learner(d, FEATURES, "visit_obs", PARAMS))):
                uplift = fit(pilot)(deploy)
                score = uplift * value_at_stake
                row = {"pilot_size": size, "pilot_clients": len(pilot), "draw": draw, "learner": learner,
                       "true_effect_correlation": round(float(np.corrcoef(uplift[eligible], effect["tau"][eligible])[0, 1]), 3)}
                for cap in (40, 80):
                    chosen = allocate(score, deploy["advisor_id"], eligible, cap, positive_only=True)
                    row[f"share_of_oracle_{cap}"] = round(float(true_value[chosen].sum() / oracle[cap]), 4)
                rows.append(row)
                print(row, flush=True)
    table = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    table.to_csv(REPORTS / "pilot_size_value.csv", index=False)
    return table


if __name__ == "__main__":
    out = run()
    summary = out.groupby(["learner", "pilot_size"])[
        ["pilot_clients", "true_effect_correlation", "share_of_oracle_40", "share_of_oracle_80"]].mean().round(3)
    print(summary.to_string())
