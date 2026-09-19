# =============================================================================
# TECAN Growth Curve Pipeline Makefile
# BIO380SP25 - Pesticide Bioremediating Bacteria Research Project
#
# PIPELINE ORDER (run 'make full-pipeline' for everything):
#   Step 0: Preprocess raw data       (every group whose *_DATA.csv is missing)
#   Step 1: Train ML classifier       (auto-skips if models/ exists)
#   Step 2: Gompertz curve analysis    (per group; truncation + ML from config.yaml)
#   Step 3: Combine group results      (→ all_groups_results.csv)
#   Step 4: Haldane inhibition         (ODE model + AICc comparison)
#   Step 5: Advanced fitting           (GP, Bootstrap, Ensemble)
#   Step 6: Statistical analysis       (ANOVA, pairwise, pub figures)
#   Step 7: Truncation comparison      (5-method ranking + rescue)
#   Step 8: Export for collaboration    (clean CSV + methodology)
#   Step 9: Synthetic validation       (one run directory per truncation mode)
#
# SYNTHETIC BENCHMARK TARGETS
#   synthetic-data      REGENERATES the committed synthetic curves in place.
#                       This INVALIDATES every published benchmark number and
#                       models/ (the classifier is trained on these curves):
#                       after running it you must re-run step 9, retrain the
#                       classifier, and record the movement in the ledger.
#   synthetic-verify    Read-only. Regenerates into a temp dir and compares
#                       against the committed snapshot on a content key, and
#                       (--self-check) regenerates twice to check determinism.
#   validation-summary  Renders the markdown from an existing run directory.
#
# Step 9 writes $(SYNTH)/output/comprehensive_test/validation/<mode>/, one
# directory per truncation mode, and defaults to first-peak -- the published
# headline. `make clean` empties those run directories of generated files with
# `git clean`, so no git-tracked file is ever deleted. The published benchmark
# snapshot is not in a run directory at all any more: it is tracked under
# $(SYNTH)/benchmark/v1/ (the pre-release baseline) and $(SYNTH)/benchmark/v2/
# (the current release), outside everything `make clean` touches.
#
# =============================================================================

PYTHON = python3
SCRIPTS = scripts
RESULTS = results/tables
SYNTH = synthetic_data

# Where step 9 writes, and which of its run directories the summary renders.
# Override on the command line, e.g. `make validation-summary RUN_DIR=...`.
SYNTH_TEST = $(SYNTH)/output/comprehensive_test
RUN_DIR = $(SYNTH_TEST)/validation/first_peak

# Seed of the committed synthetic suite. synthetic-verify reads it back from
# MANIFEST.json; changing it here regenerates a different benchmark.
SEED = 42

.PHONY: help install app app-check docs figures test test-quick \
        full-pipeline core-pipeline dry-run \
        step-0 step-1 step-2 step-3 step-4 step-5 step-6 step-7 step-8 step-9 \
        step-9-all-modes \
        run-advanced run-advanced-gp run-advanced-bootstrap \
        run-advanced-ensemble run-compare-methods run-compare-bad \
        train-classifier validate validate-synthetic validate-truncation \
        synthetic-data synthetic-verify validation-summary \
        clean

# ─── Help ────────────────────────────────────────────────────────────────────

help: ## Show this help message
	@echo ""
	@echo "  TECAN Growth Curve Pipeline"
	@echo "  ==========================="
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "    \033[36m%-25s\033[0m %s\n", $$1, $$2}'
	@echo ""

# ─── Lab interface ───────────────────────────────────────────────────────────

app: ## Open the lab interface in your browser (streamlit run app/Home.py)
	$(PYTHON) -m streamlit run app/Home.py

app-check: ## Start the interface headless and confirm it serves, then stop it
	$(PYTHON) -m pytest tests/test_app_smoke.py -q

docs: ## Re-render docs/HOW_IT_WORKS.md from docs/how_it_works/*.md with the default thresholds
	$(PYTHON) -m tecan_growth.manual

figures: ## Redraw the manual's diagrams and gallery into docs/images/
	$(PYTHON) $(SCRIPTS)/make_docs_figures.py

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

step-9: ## Step 9: Synthetic validation, first-peak truncation (the published headline)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 9

step-9-all-modes: ## Step 9 in every truncation mode, one run directory each
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 9 --all-modes

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

train-classifier: ## Train ML classifier on synthetic curves (audited real curves held out)
	$(PYTHON) $(SCRIPTS)/09_train_classifier.py --compare

# ─── Validation & Testing ────────────────────────────────────────────────────

install: ## Install Python dependencies (core + interface + tests)
	$(PYTHON) -m pip install -r requirements.txt

test: ## Run test suite
	$(PYTHON) -m pytest tests/ -v --tb=short

test-quick: ## Run fast tests only (no integration)
	$(PYTHON) -m pytest tests/ -v --tb=short -k "not integration"

validate: ## Launch interactive curve validation tool
	$(PYTHON) $(SCRIPTS)/validate_results_interactive.py

validate-synthetic: ## Run pipeline on synthetic data + compute metrics (= step-9)
	$(PYTHON) $(SCRIPTS)/run_full_pipeline.py --steps 9

# ─── Synthetic benchmark data ────────────────────────────────────────────────

synthetic-data: ## REGENERATE the committed synthetic curves (invalidates the published numbers AND models/)
	@echo "This overwrites $(SYNTH_TEST)/test_data/ in place."
	@echo "Every published benchmark number and models/*.joblib are derived from it:"
	@echo "re-run 'make step-9' and 'make train-classifier' afterwards, and record the move."
	cd $(SYNTH) && MPLBACKEND=Agg $(PYTHON) scripts/generate_comprehensive.py \
		-o output/comprehensive_test --seed $(SEED)

synthetic-verify: ## Regenerate into a temp dir and check the committed curves still reproduce
	MPLBACKEND=Agg $(PYTHON) $(SCRIPTS)/verify_synthetic_data.py --self-check

# validation_summary.py is renderer-owned. When it offers --run-dir the first
# branch below uses it; otherwise the target maps RUN_DIR onto the flags that
# renderer does have, so `make validation-summary` renders rather than aborting
# in argparse. There is no second directory to fall back to: the pre-per-mode
# single-directory layout is gone, and the tracked snapshot now lives in
# $(SYNTH)/benchmark/, which is a release snapshot rather than a run directory
# to re-render.

validation-summary: ## Render the markdown summary from an existing run directory (no re-run)
	@if MPLBACKEND=Agg $(PYTHON) $(SYNTH)/scripts/validation_summary.py --help 2>&1 \
			| grep -q -- '--run-dir'; then \
		echo "  rendering $(RUN_DIR)"; \
		MPLBACKEND=Agg $(PYTHON) $(SYNTH)/scripts/validation_summary.py \
			--run-dir $(RUN_DIR); \
	else \
		src=""; \
		for d in $(RUN_DIR); do \
			if [ -f "$$d/validation_report.json" ] && \
			   [ -f "$$d/processing_results.csv" ]; then src="$$d"; break; fi; \
		done; \
		if [ -z "$$src" ]; then \
			echo "No renderable run directory."; \
			echo "  validation_summary.py cannot read the new report.json layout yet,"; \
			echo "  and $(RUN_DIR) holds no validation_report.json +"; \
			echo "  processing_results.csv pair."; \
			echo "  Run 'make step-9' first, or point RUN_DIR at a run directory."; \
			exit 1; \
		fi; \
		echo "  validation_summary.py has no --run-dir yet; rendering $$src file by file"; \
		metrics="$$src/validation_metrics.json"; \
		[ -f "$$metrics" ] || metrics="$(SYNTH_TEST)/validation_metrics.json"; \
		MPLBACKEND=Agg $(PYTHON) $(SYNTH)/scripts/validation_summary.py \
			--metrics-json "$$metrics" \
			--report-json "$$src/validation_report.json" \
			--ground-truth $(SYNTH_TEST)/test_data/ground_truth.csv \
			--pipeline-results "$$src/processing_results.csv" \
			--output "$$src/VALIDATION_SUMMARY.md"; \
	fi

validate-truncation: ## Launch interactive truncation method validator
	$(PYTHON) $(SCRIPTS)/08_validate_truncation.py

# ─── Cleanup ─────────────────────────────────────────────────────────────────

# `git clean -fdx <dir>` is what does the deleting, not `rm -rf`: a step-9 run
# directory is not guaranteed to be purely generated. It holds ~96 MB of
# per-curve plots, and older layouts kept git-tracked files in there too
# (processing_results.csv, validation_report.json, classification_summary.png
# -- the benchmark snapshot every published number is quoted from), which an
# `rm -rf` on that glob would delete. git clean removes untracked and ignored
# files and never touches a tracked one, so the snapshot survives wherever it
# lives -- which since this release is $(SYNTH)/benchmark/v1/ and
# $(SYNTH)/benchmark/v2/, outside the run directories entirely, where it
# belongs.
clean: ## Remove generated output (never deletes a git-tracked file)
	@echo "Removing generated files from the step-9 run directories:"
	@if git rev-parse --git-dir >/dev/null 2>&1; then \
		for d in $(SYNTH_TEST)/validation $(SYNTH_TEST)/validation_*; do \
			[ -d "$$d" ] || continue; \
			n=$$(git clean -ndx -- "$$d" | wc -l); \
			t=$$(git ls-files -- "$$d" | wc -l); \
			git clean -qfdx -- "$$d"; \
			rmdir "$$d" 2>/dev/null || true; \
			if [ -d "$$d" ]; then \
				echo "  $$d: removed $$n generated path(s), kept $$t tracked file(s)"; \
			else \
				echo "  $$d: removed ($$n generated path(s), no tracked file)"; \
			fi; \
		done; \
	else \
		echo "  not a git checkout: refusing to delete $(SYNTH_TEST)/validation*"; \
		echo "  (nothing here can tell a generated plot from the tracked snapshot)"; \
	fi
	rm -rf $(SCRIPTS)/OUTPUT/
	@echo "Cleaned generated output. Raw data, committed results and every"
	@echo "git-tracked file are preserved -- including the benchmark snapshot,"
	@echo "which is tracked under $(SYNTH)/benchmark/v1/ and $(SYNTH)/benchmark/v2/,"
	@echo "outside every directory this target touches."
