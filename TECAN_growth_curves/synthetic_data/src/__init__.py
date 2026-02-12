"""
Synthetic Bacterial Growth Curve Data Generator

This package provides tools to generate synthetic bacterial growth curves
for validating and stress-testing the TECAN growth curve analysis pipeline.
"""

from .growth_models import (
    GompertzModel,
    BaranyiModel,
    LogisticModel,
    RichardsModel,
    DeathPhaseExtension,
)
from .noise_models import (
    GaussianNoise,
    ODDependentNoise,
    InstrumentNoise,
    RMSEBasedNoise,
)
from .parameter_extractor import ParameterExtractor
from .data_generator import SyntheticGrowthCurveGenerator
from .curve_scenarios import GOOD_GROWTH_SCENARIOS, BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS
from .output_formatter import TECANFormatWriter

__version__ = "1.0.0"
__all__ = [
    "GompertzModel",
    "BaranyiModel",
    "LogisticModel",
    "RichardsModel",
    "DeathPhaseExtension",
    "GaussianNoise",
    "ODDependentNoise",
    "InstrumentNoise",
    "RMSEBasedNoise",
    "ParameterExtractor",
    "SyntheticGrowthCurveGenerator",
    "GOOD_GROWTH_SCENARIOS",
    "BAD_CURVE_SCENARIOS",
    "EDGE_CASE_SCENARIOS",
    "TECANFormatWriter",
]
