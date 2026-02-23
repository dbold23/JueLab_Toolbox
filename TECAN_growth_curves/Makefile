# =============================================================================
# TECAN Growth Curve Pipeline Makefile
# BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
# =============================================================================

PYTHON = python3
SCRIPTS = scripts
DATA = data/raw
RESULTS = results/tables
SYNTH = synthetic_data

.PHONY: help install test run-pipeline run-haldane validate validate-synthetic clean

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r $(SCRIPTS)/requirements.txt

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick: ## Run fast tests only (no integration)
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not integration"

run-pipeline: ## Run full Gompertz pipeline on all groups
	bash $(SCRIPTS)/run_all_groups.sh

run-haldane: ## Run Haldane feedback inhibition analysis
	$(PYTHON) $(SCRIPTS)/05_haldane_analysis.py

validate: ## Launch interactive curve validation tool
	$(PYTHON) $(SCRIPTS)/validate_results_interactive.py

validate-synthetic: ## Run pipeline on synthetic data and compute metrics
	$(PYTHON) $(SCRIPTS)/01_growth_curve_analysis.py \
		$(SYNTH)/output/comprehensive_test/test_data/DATA \
		-o $(SYNTH)/output/comprehensive_test/validation_latest -q

run-all: run-pipeline run-haldane ## Run full pipeline + Haldane analysis

clean: ## Remove generated output (keeps raw data and results)
	rm -rf $(SYNTH)/output/comprehensive_test/validation_v*/
	rm -rf $(SCRIPTS)/OUTPUT/
	@echo "Cleaned generated output. Raw data and committed results preserved."
