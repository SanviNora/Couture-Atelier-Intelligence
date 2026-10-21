"""Profile the acquired UCI Online Retail II workbook."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

DEFAULT_INPUT = Path("data/raw/uci_online_retail_ii/online_retail_II.xlsx")
DEFAULT_OUTPUT = Path("data/staging/uci_online_retail_ii_profile.json")


def profile_workbook(input_path: Path) -> dict[str, object]:
    workbook = pd.ExcelFile(input_path)
    sheets: list[dict[str, object]] = []
    total_rows = 0

    for sheet_name in workbook.sheet_names:
        frame = pd.read_excel(input_path, sheet_name=sheet_name)
        total_rows += len(frame)
        sheets.append(
            {
                "sheet_name": sheet_name,
                "rows": len(frame),
                "columns": [str(column) for column in frame.columns],
                "missing_values": {
                    str(column): int(value)
                    for column, value in frame.isna().sum().items()
                },
                "date_min": str(frame["InvoiceDate"].min()),
                "date_max": str(frame["InvoiceDate"].max()),
                "cancelled_invoice_rows": int(
                    frame["Invoice"].astype(str).str.startswith("C").sum()
                ),
            }
        )

    return {
        "source_id": "uci_online_retail_ii",
        "data_provenance": "observed_public",
        "workbook_sheets": workbook.sheet_names,
        "total_rows": total_rows,
        "sheets": sheets,
    }


def main() -> None:
    report = profile_workbook(DEFAULT_INPUT)
    DEFAULT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
