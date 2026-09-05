# =============================================================================
# TECAN Growth Curve Pipeline Makefile
# BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
#
# PIPELINE ORDER (run 'make full-pipeline' for everything):
#   Step 0: Preprocess raw data       (Group 1 only, if needed)
#   Step 1: Train ML classifier       (auto-skips if models/ exists)
#   Step 2: Gompertz curve analysis    (per group, --adaptive, with ML)
#   Step 3: Combine group results      (→ all_groups_results.csv)
#   Step 4: Haldane inhibition         (ODE model + AICc comparison)
#   Step 5: Advanced fitting           (GP, Bootstrap, Ensemble)
#   Step 6: Statistical analysis       (ANOVA, pairwise, pub figures)
#   Step 7: Truncation comparison      (5-method ranking + rescue)
#   Step 8: Export for collaboration    (clean CSV + methodology)
#   Step 9: Synthetic validation       (480 curves → accuracy metrics)
#
# =============================================================================

PYTHON = python3
SCRIPTS = scripts
RESULTS = results/tables
SYNTH = synthetic_data

.PHONY: help install test test-quick \
        full-pipeline core-pipeline dry-run \
        step-0 step-1 step-2 step-3 step-4 step-5 step-6 step-7 step-8 step-9 \
        run-advanced run-advanced-gp run-advanced-bootstrap \
        run-advanced-ensemble run-compare-methods run-compare-bad \
        train-classifier validate validate-synthetic validate-truncation \
        clean

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo ""
	@echo "  TECAN Growth Curve Pipeline"
	@echo "  ==========================="
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─── Full Pipeline ───────────────────────────────────────────────────────────

full-pipeline: ## Run complete end-to-end pipeline (steps 0-9)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py

core-pipeline: ## Run core steps only (skip GP/Ensemble)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --skip-advanced

dry-run: ## Preview all pipeline steps without executing
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --dry-run

# ─── Individual Steps ────────────────────────────────────────────────────────

step-0: ## Step 0: Preprocess raw plate data
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 0

step-1: ## Step 1: Train ML classifier
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 1

step-2: ## Step 2: Gompertz curve analysis (all groups)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 2

step-3: ## Step 3: Combine group results
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 3

step-4: ## Step 4: Haldane inhibition analysis
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 4

step-5: ## Step 5: Advanced fitting (GP + Bootstrap + Ensemble)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 5

step-6: ## Step 6: Statistical analysis + figures
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 6

step-7: ## Step 7: Truncation method comparison + rescue
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 7

step-8: ## Step 8: Export for collaboration
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 8

step-9: ## Step 9: Synthetic validation (accuracy metrics)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 9

# ─── Advanced (granular) ─────────────────────────────────────────────────────

run-advanced: ## Run all advanced fitting (GP + Bootstrap + Ensemble)
	$(PYTHON) $(SCRIPTS)/06_advanced_fitting.py

run-advanced-gp: ## Run GP truncation only
	$(PYTHON) $(SCRIPTS)/06_advanced_fitting.py --gp-only

run-advanced-bootstrap: ## Run bootstrap CIs only
	$(PYTHON) $(SCRIPTS)/06_advanced_fitting.py --bootstrap-only

run-advanced-ensemble: ## Run ensemble truncation only
	$(PYTHON) $(SCRIPTS)/06_advanced_fitting.py --ensemble-only

run-compare-methods: ## Compare truncation methods
	$(PYTHON) $(SCRIPTS)/07_compare_truncation_methods.py

run-compare-bad: ## Compare methods + rescue bad strains
	$(PYTHON) $(SCRIPTS)/07_compare_truncation_methods.py --include-bad

# ─── ML Classifier ───────────────────────────────────────────────────────────

train-classifier: ## Train ML classifier on synthetic + audit data
	$(PYTHON) $(SCRIPTS)/09_train_classifier.py --compare

# ─── Validation & Testing ────────────────────────────────────────────────────

install: ## Install Python dependencies
	$(PYTHON) -m pip install -r $(SCRIPTS)/requirements.txt

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick: ## Run fast tests only (no integration)
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not integration"

validate: ## Launch interactive curve validation tool
	$(PYTHON) $(SCRIPTS)/validate_results_interactive.py

validate-synthetic: ## Run pipeline on synthetic data + compute metrics
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 9

validate-truncation: ## Launch interactive truncation method validator
	$(PYTHON) $(SCRIPTS)/08_validate_truncation.py

# ─── Cleanup ─────────────────────────────────────────────────────────────────

clean: ## Remove generated output (keeps raw data and committed results)
	rm -rf $(SYNTH)/output/comprehensive_test/validation_v*/
	rm -rf $(SCRIPTS)/OUTPUT/
	@echo "Cleaned generated output. Raw data and committed results preserved."
