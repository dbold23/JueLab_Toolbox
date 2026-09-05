#!/usr/bin/env bash
#
# Run the core growth-curve analysis for every configured group.
#
# This is a thin wrapper around scripts/run_full_pipeline.py, which reads the
# group definitions (raw data and output folders) from scripts/config.yaml.
# It runs step 0 (preprocess raw plate data), step 2 (Gompertz analysis per
# group) and step 3 (combine results into results/tables/all_groups_results.csv).
#
# Usage:
#   scripts/run_all_groups.sh            # steps 0, 2, 3
#   scripts/run_all_groups.sh --dry-run  # show the commands without running
#
# Any extra arguments are passed straight to run_full_pipeline.py
# (for example --no-ml or --config my_config.yaml).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

exec "$PYTHON" "$SCRIPT_DIR/run_full_pipeline.py" --steps 0,2,3 "$@"
