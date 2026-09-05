"""
Shared test fixtures for the TECAN growth curve pipeline test suite.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

# Add scripts directory to path so we can import pipeline modules
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "synthetic_data" / "src"))


@pytest.fixture
def standard_time():
    """Standard time array: 0 to 50 hours, 0.25h resolution."""
    return np.arange(0, 50, 0.25)


@pytest.fixture
def standard_gompertz_params():
    """Typical Gompertz parameters for a good growth curve."""
    return {'A': 1.2, 'mu': 0.2, 'lambda_': 4.0}


@pytest.fixture
def good_growth_curve(standard_time, standard_gompertz_params):
    """A clean Gompertz growth curve with light noise."""
    from growth_models import GompertzModel
    p = standard_gompertz_params
    clean = GompertzModel.compute(standard_time, p['A'], p['mu'], p['lambda_'])
    rng = np.random.default_rng(42)
    noisy = clean + rng.normal(0, 0.01, len(clean))
    noisy = np.maximum(0, noisy)
    return standard_time, noisy, clean


@pytest.fixture
def flat_curve(standard_time):
    """A flat no-growth curve (H2O control)."""
    rng = np.random.default_rng(42)
    od = 0.01 + rng.normal(0, 0.002, len(standard_time))
    return standard_time, np.maximum(0, od)


@pytest.fixture
def repo_root():
    """Path to the repository root."""
    return REPO_ROOT
