PY ?= ../.venv/bin/python

.PHONY: data select models clienteling all test lint

# Rebuild the warehouse from the raw Visuelle and Google Trends files (about 10 seconds).
data:
	$(PY) scripts/ingest_visuelle.py
	$(PY) scripts/build_marts.py
	$(PY) scripts/build_luxury_layer.py
	$(PY) scripts/ingest_google_trends.py

# Model selection on pre-test data only; writes configs/client_models.yml (about 10 minutes).
select:
	$(PY) scripts/experiment_client_models.py

# Refit the selected client models, test once on 2019, score clients (about 1 minute).
models:
	$(PY) scripts/build_client_value.py

# Uplift pilot, policies and power analysis (30 seconds); product picks on real purchases (about 12 minutes).
clienteling:
	$(PY) scripts/build_clienteling.py
	$(PY) scripts/build_recommendations.py

all: data models clienteling

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts dashboard
