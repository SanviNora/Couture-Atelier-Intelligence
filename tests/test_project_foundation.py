from pathlib import Path

import yaml

from couture_atelier.provenance import Provenance

PROJECT_ROOT = Path(__file__).parents[1]


def test_source_registry_has_required_fields() -> None:
    registry = yaml.safe_load((PROJECT_ROOT / "configs/data_sources.yml").read_text())

    assert registry["project"]["name"] == "Couture Atelier Intelligence"
    assert registry["project"]["affiliated_brand"] == "none"
    assert len(registry["sources"]) >= 5
    assert all(source["id"] and source["role"] for source in registry["sources"])


def test_provenance_values_are_explicit() -> None:
    assert {item.value for item in Provenance} == {
        "observed_public",
        "derived",
        "synthetic",
        "scenario",
    }
