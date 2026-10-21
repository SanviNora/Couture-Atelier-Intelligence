"""Single source of truth for project paths.

Raw licensed data (Visuelle 2.0) lives outside the repository and is never
committed. Override its location with the ``COUTURE_VISUELLE_DIR`` environment
variable.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
STAGING = DATA / "staging"
PROCESSED = DATA / "processed"
SYNTHETIC = DATA / "synthetic"
REPORTS = ROOT / "reports"
PUBLIC_BUNDLE = ROOT / "public_bundle"
CONFIGS = ROOT / "configs"
SQL = ROOT / "sql"

_DEFAULT_VISUELLE = (
    ROOT.parent / "Silhouette Vision" / "data" / "raw" / "visuelle2" / "visuelle2"
)
VISUELLE_DIR = Path(os.environ.get("COUTURE_VISUELLE_DIR", _DEFAULT_VISUELLE))
WAREHOUSE = DATA / "couture.duckdb"
