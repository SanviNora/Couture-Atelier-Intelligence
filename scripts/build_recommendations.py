"""Product picks for clienteling, judged on real purchases.

For each client, rank the current collection at their home store (pieces
launched in the 8 weeks before the cut-off or due in the 4 weeks after) and
check whether they actually bought a top pick in the next 90 days.

Signals, all known at the cut-off:
- chain popularity and home-store popularity (units sold before the cut-off)
- category, colour and fabric affinity (share of the client's past purchases)
- visual similarity: cosine similarity between the piece's image embedding and
  the mean embedding of the client's last 10 purchases (Silhouette Vision's
  Marqo-FashionSigLIP embeddings)

Blend weights are tuned on the 2018 pilot cut-off and tested once on 2019.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml

from couture_atelier.paths import CONFIGS, REPORTS, SILHOUETTE_DIR, WAREHOUSE

SIGNALS = ["chain_popularity", "store_popularity", "category_affinity", "colour_affinity",
           "fabric_affinity", "visual_similarity"]
WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0]
TOP_K = (5, 10)
HISTORY_ITEMS = 10


def load_embeddings() -> tuple[np.ndarray, pd.Index]:
    products = pd.read_parquet(SILHOUETTE_DIR / "data" / "processed" / "visuelle_products.parquet")
    vectors = np.load(SILHOUETTE_DIR / "artifacts" / "embeddings" / "visuelle" / "marqo_fashion_siglip.npy")
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors, pd.Index(products["external_code"].to_numpy())


def candidate_frames(con: duckdb.DuckDBPyConnection, cutoff: str, horizon: int) -> dict[str, pd.DataFrame]:
    c = f"date '{cutoff}'"
    candidates = con.execute(
        f"""
        with assortment as (
            select distinct w.store_id, w.product_id
            from marts.fct_store_product_week w join marts.dim_product p using (product_id)
            where p.release_date > {c} - interval 56 day and p.release_date <= {c} + interval 28 day
        ),
        chain as (
            select product_id, sum(units_sold) as units from marts.fct_store_product_week
            where week_start <= {c} - interval 6 day group by 1
        ),
        store as (
            select store_id, product_id, sum(units_sold) as units from marts.fct_store_product_week
            where week_start <= {c} - interval 6 day group by all
        )
        select a.store_id, a.product_id, p.category, p.color, p.fabric,
               coalesce(chain.units, 0) as chain_units, coalesce(store.units, 0) as store_units
        from assortment a
        join marts.dim_product p using (product_id)
        left join chain using (product_id)
        left join store using (store_id, product_id)
        order by a.store_id, a.product_id
        """
    ).fetchdf()
    history = con.execute(
        f"""
        select f.client_id, f.product_id, f.purchased_at, p.category, p.color, p.fabric
        from marts.fct_purchase f join marts.dim_product p using (product_id)
        where f.purchased_at <= {c} and f.client_id in (select client_id from lux.client)
        order by f.client_id, f.purchased_at, f.product_id
        """
    ).fetchdf()
    future = con.execute(
        f"""
        select distinct f.client_id, f.product_id
        from marts.fct_purchase f
        where f.purchased_at > {c} and f.purchased_at <= {c} + interval {horizon} day
        """
    ).fetchdf()
    homes = con.execute("select client_id, home_store_id from lux.client").fetchdf()
    return {"candidates": candidates, "history": history, "future": future, "homes": homes}


def signal_tensors(data: dict[str, pd.DataFrame], vectors: np.ndarray, vector_index: pd.Index) -> list[dict]:
    """Per store: evaluation clients x candidate pieces x signals, plus the true purchases."""
    candidates, history, future, homes = data["candidates"], data["history"], data["future"], data["homes"]
    candidate_keys = set(zip(candidates["store_id"], candidates["product_id"], strict=True))
    future = future.merge(homes, on="client_id")
    future = future[[k in candidate_keys for k in zip(future["home_store_id"], future["product_id"], strict=True)]]
    eval_clients = set(future["client_id"]) & set(history["client_id"])
    history = history[history["client_id"].isin(eval_clients)]
    bought_before = history.groupby("client_id")["product_id"].agg(set)
    recent = history.groupby("client_id").tail(HISTORY_ITEMS)
    recent = recent[recent["product_id"].isin(vector_index)]
    rows = vector_index.get_indexer(recent["product_id"])
    profile = pd.DataFrame(vectors[rows]).groupby(recent["client_id"].to_numpy()).mean()

    def shares(column: str) -> pd.DataFrame:
        counts = history.groupby(["client_id", column]).size().unstack(fill_value=0)
        return counts.div(counts.sum(axis=1), axis=0)

    affinity = {col: shares(col) for col in ("category", "color", "fabric")}
    truth = future[future["client_id"].isin(eval_clients)].groupby("client_id")["product_id"].agg(set)
    blocks = []
    ordered = pd.Series(sorted(eval_clients))
    store_of = ordered.map(homes.set_index("client_id")["home_store_id"])
    for store, store_clients in ordered.groupby(store_of):
        items = candidates[candidates["store_id"] == store].reset_index(drop=True)
        clients = store_clients.to_numpy()
        n_c, n_i = len(clients), len(items)
        tensor = np.zeros((n_c, n_i, len(SIGNALS)), dtype=np.float32)
        tensor[:, :, 0] = np.log1p(items["chain_units"].to_numpy())[None, :]
        tensor[:, :, 1] = np.log1p(items["store_units"].to_numpy())[None, :]
        for k, (col, key) in enumerate((("category", "category"), ("color", "color"), ("fabric", "fabric")), start=2):
            table = affinity[col].reindex(index=clients, columns=items[key].unique(), fill_value=0)
            tensor[:, :, k] = table[items[key]].to_numpy()
        item_rows = vector_index.get_indexer(items["product_id"])
        has_vec = item_rows >= 0
        client_vecs = profile.reindex(clients).fillna(0).to_numpy()
        visual = np.zeros((n_c, n_i), dtype=np.float32)
        visual[:, has_vec] = client_vecs @ vectors[item_rows[has_vec]].T
        tensor[:, :, 5] = visual
        already = np.array([[p in bought_before.get(cl, set()) for p in items["product_id"]] for cl in clients])
        bought = np.array([[p in truth.get(cl, set()) for p in items["product_id"]] for cl in clients])
        blocks.append({"tensor": tensor, "already": already, "bought": bought,
                       "new_launch": (items["chain_units"].to_numpy() == 0)})
    return blocks


def standardise(blocks: list[dict]) -> list[dict]:
    stacked = np.concatenate([b["tensor"].reshape(-1, len(SIGNALS)) for b in blocks])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-9
    for b in blocks:
        b["z"] = (b["tensor"] - mean) / std
    return blocks


def client_hits(blocks: list[dict], weights: np.ndarray, k: int, new_only: bool = False) -> np.ndarray:
    """1 if any of the client's top-k picks was bought in the horizon, per client.

    With ``new_only`` only pieces with no sales before the cut-off are ranked,
    and only clients who bought one of them are counted.
    """
    out = []
    for b in blocks:
        allowed = ~b["already"] & (b["new_launch"][None, :] if new_only else True)
        relevant = b["bought"] & allowed
        keep = relevant.any(axis=1) if new_only else np.ones(len(relevant), dtype=bool)
        score = np.where(allowed, b["z"] @ weights, -np.inf)
        order = np.argsort(-score, axis=1, kind="stable")
        top = np.take_along_axis(relevant, order[:, :k], axis=1)
        out.append(top.any(axis=1)[keep])
    return np.concatenate(out).astype(float)


def hit_rates(blocks: list[dict], weights: np.ndarray, new_only: bool = False) -> dict[str, float]:
    rates = {f"hit@{k}": round(float(client_hits(blocks, weights, k, new_only).mean()), 4) for k in TOP_K}
    return rates | {"clients": len(client_hits(blocks, weights, TOP_K[0], new_only))}


def paired_difference(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diffs = [(a[i] - b[i]).mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n_boot))]
    low, high = np.percentile(diffs, [2.5, 97.5])
    return {"difference": round(float((a - b).mean()), 4), "ci_low": round(float(low), 4),
            "ci_high": round(float(high), 4), "significant": bool(low > 0 or high < 0)}


def random_hit_rates(blocks: list[dict], new_only: bool = False) -> dict[str, float]:
    """Expected hit rate of random picks (excluding pieces already bought)."""
    out = {}
    for k in TOP_K:
        total, clients = 0.0, 0
        for b in blocks:
            allowed = ~b["already"] & (b["new_launch"][None, :] if new_only else True)
            relevant_mask = b["bought"] & allowed
            keep = relevant_mask.any(axis=1) if new_only else np.ones(len(allowed), dtype=bool)
            available, relevant = allowed.sum(axis=1)[keep], relevant_mask.sum(axis=1)[keep]
            for n, r in zip(available, relevant, strict=True):
                picks = min(k, n)
                miss = np.prod([(n - r - i) / (n - i) for i in range(picks)]) if n > 0 else 1.0
                total += 1 - miss
            clients += len(available)
        out[f"hit@{k}"] = round(total / clients, 4)
    return out


def build_recommendations(warehouse: Path = WAREHOUSE) -> dict:
    cfg = yaml.safe_load((CONFIGS / "clienteling.yml").read_text())
    vectors, vector_index = load_embeddings()
    with duckdb.connect(str(warehouse), read_only=True) as con:
        tune = standardise(signal_tensors(candidate_frames(con, cfg["cutoffs"]["pilot"], cfg["horizon_days"]),
                                          vectors, vector_index))
        test = standardise(signal_tensors(candidate_frames(con, cfg["cutoffs"]["deploy"], cfg["horizon_days"]),
                                          vectors, vector_index))

    def one_hot(name: str) -> np.ndarray:
        return np.array([1.0 if s == name else 0.0 for s in SIGNALS])

    single = {name: one_hot(name) for name in SIGNALS}
    grid = []
    for combo in itertools.product(WEIGHT_GRID, repeat=len(SIGNALS)):
        weights = np.array(combo)
        if weights.sum() == 0:
            continue
        grid.append((weights, hit_rates(tune, weights)["hit@5"]))
    best_weights, best_tune = max(grid, key=lambda g: g[1])
    no_visual = max((g for g in grid if g[0][5] == 0), key=lambda g: g[1])

    report = {
        "cutoffs": cfg["cutoffs"], "horizon_days": cfg["horizon_days"], "signals": SIGNALS,
        "tuning": {"grid_size": len(grid), "best_weights": dict(zip(SIGNALS, best_weights.tolist(), strict=True)),
                   "best_hit@5_on_pilot": best_tune,
                   "best_weights_without_visual": dict(zip(SIGNALS, no_visual[0].tolist(), strict=True))},
        "test": {
            "random": random_hit_rates(test),
            **{f"only_{name}": hit_rates(test, w) for name, w in single.items()},
            "blend_without_visual": hit_rates(test, no_visual[0]),
            "blend_with_visual": hit_rates(test, best_weights),
            "visual_gain_hit@5": paired_difference(client_hits(test, best_weights, 5),
                                                   client_hits(test, no_visual[0], 5)),
        },
        # New launches have no sales yet, so popularity cannot rank them.
        "test_new_launches_only": {
            "random": random_hit_rates(test, new_only=True),
            "affinity_only": hit_rates(test, np.array([0, 0, 1, 1, 1, 0.0]), new_only=True),
            "visual_only": hit_rates(test, one_hot("visual_similarity"), new_only=True),
            "affinity_plus_visual": hit_rates(test, np.array([0, 0, 1, 1, 1, 2.0]), new_only=True),
            "visual_gain_hit@5": paired_difference(
                client_hits(test, np.array([0, 0, 1, 1, 1, 2.0]), 5, new_only=True),
                client_hits(test, np.array([0, 0, 1, 1, 1, 0.0]), 5, new_only=True)),
        },
        "data_provenance": "derived from observed purchases",
    }
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "recommendations.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(build_recommendations(), indent=2))
