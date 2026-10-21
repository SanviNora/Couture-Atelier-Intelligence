"""Download and register the UCI Online Retail II source."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path

SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
SOURCE_ID = "uci_online_retail_ii"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ingest(destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "online_retail_ii.zip"

    if not archive_path.exists():
        subprocess.run(
            ["curl", "--fail", "--location", "--output", str(archive_path), SOURCE_URL],
            check=True,
        )

    extracted_files: list[str] = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            archive.extract(member, destination)
            extracted_files.append(member.filename)

    manifest = {
        "source_id": SOURCE_ID,
        "source_url": SOURCE_URL,
        "downloaded_at_utc": datetime.now(UTC).isoformat(),
        "archive": archive_path.name,
        "archive_sha256": sha256(archive_path),
        "extracted_files": extracted_files,
        "data_provenance": "observed_public",
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/raw/uci_online_retail_ii"),
    )
    args = parser.parse_args()
    manifest = ingest(args.destination)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
