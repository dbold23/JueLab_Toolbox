"""
Access to the validated numerical core that lives in ``scripts/``.

The pipeline scripts are named ``01_growth_curve_analysis.py`` etc., which are
not importable module names, so they are loaded here by path exactly once.
Nothing in them is modified: every function the interface calls is the same
one the CLI and the validation study used.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
import types
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"

_lock = threading.Lock()
_cache: dict = {}


def _load(module_name: str, filename: str) -> types.ModuleType:
    with _lock:
        if module_name in _cache:
            return _cache[module_name]
        if module_name in sys.modules:
            _cache[module_name] = sys.modules[module_name]
            return sys.modules[module_name]
        path = SCRIPTS_DIR / filename
        if not path.exists():
            raise FileNotFoundError(f"Expected pipeline script at {path}")
        # Headless-safe plotting: the scripts import pyplot at module level.
        os.environ.setdefault("MPLBACKEND", "Agg")
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.insert(0, str(SCRIPTS_DIR))
        spec = importlib.util.spec_from_file_location(module_name, str(path))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        _cache[module_name] = module
        return module


def growth_curve_analysis() -> types.ModuleType:
    """``scripts/01_growth_curve_analysis.py`` (truncation, Gompertz fit, classification)."""
    return _load("growth_curve_analysis", "01_growth_curve_analysis.py")


def ml_classifier() -> types.ModuleType:
    """``scripts/ml_classifier.py`` (optional pre-fit gate and post-fit classifier)."""
    return _load("ml_classifier", "ml_classifier.py")


def haldane_analysis() -> types.ModuleType:
    """``scripts/05_haldane_analysis.py`` (substrate-inhibition ODE model)."""
    return _load("haldane_analysis", "05_haldane_analysis.py")


def model_files_present(models_dir: Optional[Path] = None) -> bool:
    """True when both trained ML classifier files exist."""
    d = Path(models_dir) if models_dir else REPO_ROOT / "models"
    return (d / "prefit_gate.joblib").exists() and (d / "postfit_classifier.joblib").exists()
