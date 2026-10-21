"""Load the Google Trends downloads (Oct 2021 to Oct 2026) and build a market pulse.

Each download is scaled 0-100 on its own. Files that include the anchor term
"Louis Vuitton Speedy" (United States) are rescaled onto the luxury-bags file's
scale, so bags across tiers can be compared. Files without the anchor stay on
their own scale and are only compared within the file.

These series cannot be joined to Visuelle (2016-2021) as model features; they
are a current market signal for the tier dial and the dashboard.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import duckdb
import pandas as pd

from couture_atelier.paths import RAW, REPORTS, WAREHOUSE

SOURCE_DIR = RAW / "google_trends"
ANCHOR = "Louis Vuitton Speedy"
ANCHOR_FILE = "luxury_bags"
FILES = {
    "luxury_bags": "luxury",
    "aspirational_bags": "aspirational",
    "premium_brands": "premium",
    "luxury_brands": "luxury",
    "luxury_brands_worldwide": "luxury",
    "market_themes": "theme",
}
MAX_ANCHOR_BELOW_ONE = 0.10  # an anchor mostly reported as "<1" is too coarse to rescale with
MIN_BASE_FOR_CHANGE = 1.0  # no percent change from a near-zero starting level
HEADER = re.compile(r"^(?P<term>.+): \((?P<region>.+)\)$")


def read_trends_csv(path: Path) -> pd.DataFrame:
    """Parse one 'Interest over time' download into long format."""
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(("Week,", "Month,", "Day,")))
    wide = pd.read_csv(path, skiprows=start, encoding="utf-8-sig")
    period = wide.columns[0]
    long = wide.melt(id_vars=period, var_name="column", value_name="raw")
    parts = long["column"].str.extract(HEADER)
    if parts.isna().any().any():
        raise ValueError(f"Unexpected column header in {path.name}")
    raw = long["raw"].astype(str).str.strip()
    long["interest"] = pd.to_numeric(raw.replace("<1", "0.5"), errors="raise")
    long["below_one"] = raw.eq("<1")
    return pd.DataFrame({
        "week": pd.to_datetime(long[period]),
        "term": parts["term"],
        "region": parts["region"],
        "interest": long["interest"],
        "below_one": long["below_one"],
        "source_file": path.stem,
    })


def rescale_to_anchor(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Put files that share the anchor term on the anchor file's scale."""
    reference = frames[ANCHOR_FILE]
    ref_total = reference.loc[reference["term"] == ANCHOR, "interest"].sum()
    out = []
    for name, frame in frames.items():
        frame = frame.copy()
        anchor = frame[frame["term"] == ANCHOR]
        anchor_total = anchor["interest"].sum()
        has_anchor = anchor_total > 0 and anchor["below_one"].mean() < MAX_ANCHOR_BELOW_ONE
        frame["scale"] = ref_total / anchor_total if has_anchor else 1.0
        frame["common_scale"] = bool(has_anchor)
        frame["interest_common"] = frame["interest"] * frame["scale"]
        frame["tier"] = FILES[name]
        out.append(frame)
    return pd.concat(out, ignore_index=True)


def market_pulse(trends: pd.DataFrame) -> pd.DataFrame:
    """Five-year change per term: mean of the last 52 weeks against the first 52."""
    rows = []
    for (file, term, region), series in trends.groupby(["source_file", "term", "region"]):
        series = series.sort_values("week")
        first, last = series["interest_common"].head(52).mean(), series["interest_common"].tail(52).mean()
        rows.append({
            "source_file": file, "term": term, "region": region, "tier": series["tier"].iloc[0],
            "common_scale": bool(series["common_scale"].iloc[0]),
            "first_year_mean": round(first, 2), "last_year_mean": round(last, 2),
            "change_pct": round((last / first - 1) * 100, 1) if first >= MIN_BASE_FOR_CHANGE else None,
            "weeks_below_one": int(series["below_one"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["source_file", "change_pct"], ascending=[True, False])


def ingest(source_dir: Path = SOURCE_DIR, warehouse: Path = WAREHOUSE) -> dict[str, object]:
    missing = [name for name in FILES if not (source_dir / f"{name}.csv").exists()]
    if missing:
        raise FileNotFoundError(f"Missing Google Trends downloads: {missing}")
    frames = {name: read_trends_csv(source_dir / f"{name}.csv") for name in FILES}
    weeks = {name: (f["week"].min(), f["week"].max(), f["week"].nunique()) for name, f in frames.items()}
    if len({w for w in weeks.values()}) != 1:
        raise ValueError(f"Downloads cover different weeks: {weeks}")
    trends = rescale_to_anchor(frames)
    pulse = market_pulse(trends)

    with duckdb.connect(str(warehouse)) as con:
        con.execute("create schema if not exists trends")
        con.register("t", trends)
        con.execute("create or replace table trends.weekly as select *, 'public_source' as data_provenance from t")
        con.register("p", pulse)
        con.execute("create or replace table trends.market_pulse as select * from p")

    manifest = {
        "source": "Google Trends, Interest over time, downloaded manually",
        "files": {f"{n}.csv": hashlib.sha256((source_dir / f"{n}.csv").read_bytes()).hexdigest() for n in FILES},
        "weeks": {"first": str(trends["week"].min().date()), "last": str(trends["week"].max().date()),
                  "count": int(trends["week"].nunique())},
        "anchor": f"{ANCHOR} (United States), on the {ANCHOR_FILE} scale",
        "scales": {n: round(float(trends.loc[trends.source_file == n, "scale"].iloc[0]), 4) for n in FILES},
        "data_provenance": "public_source",
    }
    (source_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    REPORTS.mkdir(exist_ok=True)
    pulse.to_csv(REPORTS / "market_pulse.csv", index=False)
    return {"weeks": manifest["weeks"], "scales": manifest["scales"]}


if __name__ == "__main__":
    print(json.dumps(ingest(), indent=2))
    print(pd.read_csv(REPORTS / "market_pulse.csv").to_string(index=False))
