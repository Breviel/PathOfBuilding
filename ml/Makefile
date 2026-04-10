.PHONY: help collect-data features train-build-tree train-atlas train-gear train-all serve test clean

PYTHON ?= python
VENV   ?= .venv/bin/python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ── Data Collection ────────────────────────────────────────────────────

collect-data: ## Collect builds from poe.ninja + parse tree data + atlas strategies
	$(PYTHON) -m ml.data.collectors.poe_ninja
	$(PYTHON) -m ml.data.collectors.tree_data
	$(PYTHON) -m ml.data.collectors.atlas_data

collect-ninja: ## Collect builds from poe.ninja only
	$(PYTHON) -m ml.data.collectors.poe_ninja

collect-tree: ## Parse and export PoB passive tree data
	$(PYTHON) -m ml.data.collectors.tree_data

collect-atlas: ## Collect atlas strategy builds from poe-atlas.com + poe.ninja heatmap
	$(PYTHON) -m ml.data.collectors.atlas_data

# ── Feature Engineering ────────────────────────────────────────────────

features: ## Build feature vectors from raw data
	$(PYTHON) -m ml.data.processing.feature_engineering all

# ── Training ───────────────────────────────────────────────────────────

train-build-tree: ## Train the Build Tree Optimizer (GAT)
	$(PYTHON) -m ml.training.train_build_tree

train-atlas: ## Train the Atlas Tree Optimizer (XGBoost)
	$(PYTHON) -m ml.training.train_atlas

train-gear: ## Train the Gear Improvement Suggester
	$(PYTHON) -m ml.training.train_gear

train-all: train-build-tree train-atlas train-gear ## Train all models

# ── Inference ──────────────────────────────────────────────────────────

serve: ## Start FastAPI prediction server on :8000
	uvicorn ml.inference.predict:app --host 0.0.0.0 --port 8000 --reload

# ── Utilities ──────────────────────────────────────────────────────────

test: ## Run basic smoke tests
	$(PYTHON) -c "from ml.data.collectors.pob_parser import ParsedBuild; print('pob_parser OK')"
	$(PYTHON) -c "from ml.data.processing.feature_engineering import stat_text_to_vector; print('features OK')"
	$(PYTHON) -c "from ml.models.build_tree_optimizer import BuildTreeGAT; print('build_tree OK')" 2>/dev/null || echo "PyG not installed (optional)"
	$(PYTHON) -c "from ml.models.atlas_tree_optimizer import AtlasReturnPredictor; print('atlas OK')"
	$(PYTHON) -c "from ml.models.gear_suggester import GearSuggestionPipeline; print('gear OK')"
	@echo "All smoke tests passed."

clean: ## Remove generated data and checkpoints
	rm -rf data/raw/poe_ninja/
	rm -rf data/processed/
	rm -rf checkpoints/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install: ## Install Python dependencies
	pip install -r requirements.txt
