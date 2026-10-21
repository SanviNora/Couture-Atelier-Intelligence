# Ingestion

## First source: UCI Online Retail II

The first ingestion target is the [UCI Online Retail II dataset](https://archive.ics.uci.edu/dataset/502/online+retail+ii). It is openly downloadable under CC BY 4.0 and provides a useful transaction backbone for customer-value analysis.

Run from the project root:

```bash
python scripts/ingest_uci_online_retail.py
```

The script:

1. Downloads the source archive only when it is not already present.
2. Extracts the original files without altering them.
3. Computes a SHA-256 checksum for the downloaded archive.
4. Writes `manifest.json` with source URL, timestamp, extracted files, and provenance.

Raw files are intentionally ignored by Git. The manifest is the record of what was acquired and when.

## Validation before transformation

The ingestion profile inspected the workbook schema, missing values, cancellations, date range, customer coverage, and duplicate invoices before staging.

Staging currently applies these rules: exact duplicate rows are removed and counted, cancellation rows are retained as return events, required-field and numeric validity checks fail closed, and missing customer IDs are preserved with an explicit flag.
