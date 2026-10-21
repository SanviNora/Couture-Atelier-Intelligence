PY ?= ../.venv/bin/python

.PHONY: data select models clienteling demand allocation tiers all test lint

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

# Uplift over 3 pilot draws, policies, power analysis (8 minutes); pilot-size study (1 minute); product picks (12 minutes).
clienteling:
	$(PY) scripts/build_clienteling.py
	$(PY) -m scripts.pilot_size_value
	$(PY) scripts/build_recommendations.py

# Lost-sales recovery and launch forecast (2 minutes), then the luxury scarcity scenario (1 minute).
demand:
	$(PY) scripts/build_demand.py
	$(PY) -m scripts.scarcity_scenario

# Store allocation, end of season and buy depth (10 seconds); hero pieces to clients (2 minutes).
allocation:
	$(PY) scripts/build_allocation.py
	$(PY) -m scripts.build_client_allocation

# Markdown lift and discount-acquired clients on real data (1 minute), then the brand tier dial (seconds).
tiers:
	$(PY) scripts/discount_effects.py
	$(PY) scripts/build_tier_dial.py

all: data models clienteling demand allocation tiers

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests scripts dashboard
