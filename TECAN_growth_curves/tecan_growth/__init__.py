"""
tecan_growth — importable core for the TECAN growth-curve analysis pipeline.

The numerical core (truncation, Gompertz fitting, classification) lives in the
validated scripts under ``scripts/`` and is exposed here unchanged through
:mod:`tecan_growth.legacy`. This package adds what the lab interface needs on
top of it:

* :mod:`tecan_growth.io`         — read TECAN Spark exports (.xlsx) and CSV values/keys
* :mod:`tecan_growth.plate`      — plate/well helpers and the plate *key* (well -> media/strain)
* :mod:`tecan_growth.preprocess` — blank subtraction and replicate averaging
* :mod:`tecan_growth.analyze`    — per-curve and per-plate analysis returning tidy results
* :mod:`tecan_growth.plots`      — interactive Plotly figures
* :mod:`tecan_growth.export`     — CSV/Excel/JSON exports and run manifests
"""

__version__ = "1.0.0"

from .plate import PlateKey, well_sort_key, normalize_well, PLATE_96_WELLS  # noqa: F401
from .io import PlateRun, read_tecan_export, read_values_csv, read_key, load_plate  # noqa: F401
from .preprocess import ProcessedPlate, preprocess_plate  # noqa: F401
