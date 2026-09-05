"""
Make the repository importable when Streamlit runs ``app/Home.py``.

Streamlit puts the script's own folder on ``sys.path``, not the repository
root, so ``import tecan_growth`` would fail without this. Every page imports
this module first.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Headless-safe matplotlib for the CLI plot helpers the export uses.
os.environ.setdefault("MPLBACKEND", "Agg")

EXAMPLE_PLATE = REPO_ROOT / "data" / "raw" / "Group 3" / "GrowthRate_GROUP3.xlsx"
DOCS_DIR = REPO_ROOT / "docs"
RUNS_DIR = REPO_ROOT / "runs"
KEYS_DIR = REPO_ROOT / "keys"
