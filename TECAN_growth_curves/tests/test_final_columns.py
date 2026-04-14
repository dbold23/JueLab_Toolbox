"""
Tests for Phase A.1: _select_final_params and the unified *_final / *_source
columns emitted by 01_growth_curve_analysis.py.

Verifies the decision table from the plan:
- identifiable  -> source='refined',              value = refined
- weak           -> source='whole_curve_fallback', value = gompertz (whole-curve)
- unidentifiable -> source='whole_curve_fallback', value = gompertz
- refined=0 or NaN -> source='whole_curve_fallback', value = gompertz
- gompertz also missing -> source='unusable', value = NaN
- pp=None -> source='whole_curve_fallback' for available gompertz_*, 'unusable' otherwise
"""
from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).parent.parent


def _load_step01():
    spec = importlib.util.spec_from_file_location(
        "gca", REPO_ROOT / "scripts" / "01_growth_curve_analysis.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@dataclass
class PPStub:
    mu_value: float = 0.18
    mu_rel_err: float = 0.05
    A_value: float = 1.25
    A_rel_err: float = 0.03
    lam_value: float = 3.5
    lam_rel_err: float = 0.08
    # Placeholders to satisfy PerParamTruncation shape
    mu_best_idx: int = 0
    mu_best_time: float = 0.0
    A_best_idx: int = 0
    A_best_time: float = 0.0
    lam_best_idx: int = 0
    lam_best_time: float = 0.0
    disagreement_hours: float = 0.0
    r2_at_mu_trunc: float = 0.99
    r2_at_lam_trunc: float = 0.99
    r2_at_A_trunc: float = 0.99


@pytest.fixture(scope="module")
def gca():
    return _load_step01()


def test_all_identifiable_uses_refined(gca):
    pp = PPStub()
    r = gca._select_final_params(
        gompertz_a=1.3, gompertz_a_err=0.02,
        gompertz_mu=0.15, gompertz_mu_err=0.01,
        gompertz_lam=3.0, gompertz_lam_err=0.2,
        pp=pp,
        mu_ident='identifiable', A_ident='identifiable', lam_ident='identifiable',
    )
    assert r['mu_source'] == 'refined'
    assert r['A_source'] == 'refined'
    assert r['lam_source'] == 'refined'
    assert r['mu_final'] == pp.mu_value
    assert r['A_final'] == pp.A_value
    assert r['lam_final'] == pp.lam_value
    # Errors should be refined_rel_err * refined_value
    assert r['mu_final_err'] == pytest.approx(pp.mu_rel_err * pp.mu_value)


def test_weak_falls_back_to_whole_curve(gca):
    # Only λ is weak; should fall back to gompertz_lambda
    pp = PPStub()
    r = gca._select_final_params(
        gompertz_a=1.3, gompertz_a_err=0.02,
        gompertz_mu=0.15, gompertz_mu_err=0.01,
        gompertz_lam=3.0, gompertz_lam_err=0.2,
        pp=pp,
        mu_ident='identifiable', A_ident='identifiable', lam_ident='weak',
    )
    assert r['lam_source'] == 'whole_curve_fallback'
    assert r['lam_final'] == 3.0
    assert r['lam_final_err'] == 0.2
    # μ, A still refined
    assert r['mu_source'] == 'refined'
    assert r['A_source'] == 'refined'


def test_unidentifiable_falls_back(gca):
    pp = PPStub()
    r = gca._select_final_params(
        gompertz_a=1.3, gompertz_a_err=0.02,
        gompertz_mu=0.15, gompertz_mu_err=0.01,
        gompertz_lam=3.0, gompertz_lam_err=0.2,
        pp=pp,
        mu_ident='unidentifiable', A_ident='identifiable', lam_ident='identifiable',
    )
    assert r['mu_source'] == 'whole_curve_fallback'
    assert r['mu_final'] == 0.15


def test_zero_refined_falls_back(gca):
    """When the fallback path in find_per_param_truncation emits value=0,
    _select_final_params must detect and fall back to whole-curve."""
    pp = PPStub(mu_value=0.0, mu_rel_err=999.0,
                A_value=0.0, A_rel_err=999.0,
                lam_value=0.0, lam_rel_err=999.0)
    r = gca._select_final_params(
        gompertz_a=1.3, gompertz_a_err=0.02,
        gompertz_mu=0.15, gompertz_mu_err=0.01,
        gompertz_lam=3.0, gompertz_lam_err=0.2,
        pp=pp,
        mu_ident='identifiable', A_ident='identifiable', lam_ident='identifiable',
    )
    # Even though ident == 'identifiable', value=0 should trigger fallback
    assert r['mu_source'] == 'whole_curve_fallback'
    assert r['mu_final'] == 0.15


def test_both_missing_is_unusable(gca):
    pp = PPStub()
    r = gca._select_final_params(
        gompertz_a=None, gompertz_a_err=None,
        gompertz_mu=None, gompertz_mu_err=None,
        gompertz_lam=None, gompertz_lam_err=None,
        pp=pp,
        mu_ident='unidentifiable', A_ident='unidentifiable', lam_ident='unidentifiable',
    )
    assert r['mu_source'] == 'unusable'
    assert r['A_source'] == 'unusable'
    assert r['lam_source'] == 'unusable'
    assert np.isnan(r['mu_final'])


def test_pp_none_uses_whole_curve(gca):
    r = gca._select_final_params(
        gompertz_a=1.3, gompertz_a_err=0.02,
        gompertz_mu=0.15, gompertz_mu_err=0.01,
        gompertz_lam=3.0, gompertz_lam_err=0.2,
        pp=None,
        mu_ident=None, A_ident=None, lam_ident=None,
    )
    assert r['mu_source'] == 'whole_curve_fallback'
    assert r['A_source'] == 'whole_curve_fallback'
    assert r['lam_source'] == 'whole_curve_fallback'
    assert r['mu_final'] == 0.15
