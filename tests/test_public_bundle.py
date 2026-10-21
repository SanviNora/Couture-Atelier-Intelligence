import json

import pandas as pd

from couture_atelier.paths import PUBLIC_BUNDLE
from couture_atelier.public_bundle import check_bundle


def write_manifest(bundle, entries) -> None:
    (bundle / "MANIFEST.json").write_text(json.dumps({"files": entries}))


def test_repository_public_bundle_is_clean() -> None:
    assert check_bundle(PUBLIC_BUNDLE) == []


def test_undeclared_file_is_rejected(tmp_path) -> None:
    pd.DataFrame({"store": [1], "n": [50]}).to_parquet(tmp_path / "stores.parquet")
    write_manifest(tmp_path, [])

    assert check_bundle(tmp_path) == ["stores.parquet: not declared in MANIFEST.json"]


def test_raw_client_columns_are_rejected(tmp_path) -> None:
    pd.DataFrame({"customer": [1], "qty": [2]}).to_parquet(tmp_path / "rows.parquet")
    write_manifest(tmp_path, [{"path": "rows.parquet", "provenance": "aggregate"}])

    assert any("forbidden columns" in problem for problem in check_bundle(tmp_path))


def test_small_aggregate_groups_are_rejected(tmp_path) -> None:
    pd.DataFrame({"store": [1, 2], "n": [3, 40]}).to_parquet(tmp_path / "agg.parquet")
    write_manifest(tmp_path, [{"path": "agg.parquet", "provenance": "aggregate"}])

    assert check_bundle(tmp_path) == ["agg.parquet: 1 groups smaller than 10"]


def test_raw_provenance_is_not_publishable(tmp_path) -> None:
    pd.DataFrame({"store": [1], "n": [50]}).to_parquet(tmp_path / "x.parquet")
    write_manifest(tmp_path, [{"path": "x.parquet", "provenance": "observed_public"}])

    assert any("not publishable" in problem for problem in check_bundle(tmp_path))
