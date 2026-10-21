"""Guard rules for the public demo bundle.

The live demo may only ship aggregates, synthetic records, and model outputs.
Raw Visuelle rows and real client identifiers must never leave the machine.
Every file in the bundle must be declared in ``MANIFEST.json`` with an allowed
provenance class.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ALLOWED_PROVENANCE = {"aggregate", "synthetic", "model_output", "public_source", "documentation"}
# Column names that identify real clients or reproduce raw transaction rows.
FORBIDDEN_COLUMNS = {"customer", "source_customer_id", "receipt_id", "transaction_at", "data"}
TABULAR_SUFFIXES = {".parquet", ".csv", ".json"}
MIN_GROUP_SIZE = 10


def _read_table(path: Path) -> pd.DataFrame | None:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path, nrows=1000)
    return None


def check_bundle(bundle: Path) -> list[str]:
    """Return a list of violations; an empty list means the bundle is safe to publish."""
    if not bundle.exists():
        return []
    manifest_path = bundle / "MANIFEST.json"
    files = [p for p in bundle.rglob("*") if p.is_file() and p.name not in {"MANIFEST.json", ".gitkeep"}]
    if not files:
        return []
    if not manifest_path.exists():
        return ["MANIFEST.json is missing"]

    manifest = json.loads(manifest_path.read_text())
    declared = {entry["path"]: entry for entry in manifest.get("files", [])}
    problems: list[str] = []
    for path in files:
        relative = path.relative_to(bundle).as_posix()
        entry = declared.get(relative)
        if entry is None:
            problems.append(f"{relative}: not declared in MANIFEST.json")
            continue
        if entry.get("provenance") not in ALLOWED_PROVENANCE:
            problems.append(f"{relative}: provenance {entry.get('provenance')!r} is not publishable")
        if path.suffix not in TABULAR_SUFFIXES:
            continue
        table = _read_table(path)
        if table is None:
            continue
        leaked = FORBIDDEN_COLUMNS.intersection(map(str, table.columns))
        if leaked:
            problems.append(f"{relative}: contains forbidden columns {sorted(leaked)}")
        if entry.get("provenance") == "aggregate" and "n" in table.columns:
            small = int((table["n"] < MIN_GROUP_SIZE).sum())
            if small:
                problems.append(f"{relative}: {small} groups smaller than {MIN_GROUP_SIZE}")
    return problems
