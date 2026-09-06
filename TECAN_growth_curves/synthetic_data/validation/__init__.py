"""Validation framework for testing the analysis pipeline.

``benchmark.run_benchmark`` is the entry point; ``pipeline_validator`` holds the
metric library it calls and the legacy CLI that ``scripts/run_full_pipeline.py``
step 9 still invokes; ``matching`` holds the one ground-truth join every
consumer shares.
"""

from .matching import BenchmarkMatchError, extract_curve_id, match_results
from .pipeline_validator import (
    GATE_IDS,
    PipelineValidator,
    classification_metrics,
    failure_analysis,
    gate_attribution,
    parameter_recovery,
    per_scenario_recovery,
    per_scenario_table,
    truncation_metrics,
    uncertainty_calibration,
)
from .comparison_report import ComparisonReport
from .benchmark import BenchmarkProvenanceError, run_benchmark

__all__ = [
    "BenchmarkMatchError",
    "BenchmarkProvenanceError",
    "ComparisonReport",
    "GATE_IDS",
    "PipelineValidator",
    "classification_metrics",
    "extract_curve_id",
    "failure_analysis",
    "gate_attribution",
    "match_results",
    "parameter_recovery",
    "per_scenario_recovery",
    "per_scenario_table",
    "run_benchmark",
    "truncation_metrics",
    "uncertainty_calibration",
]
