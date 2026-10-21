PY ?= ../.venv/bin/python

.PHONY: data test lint

# Rebuild the warehouse from the raw Visuelle files (about 10 seconds).
data:
	$(PY) scripts/ingest_visuelle.py
	$(PY) scripts/build_marts.py
	$(PY) scripts/build_luxury_layer.py

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts dashboard
