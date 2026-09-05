"""
Tests for tecan_growth.analyze: the new in-memory path must give the same
numbers as the CLI script, and its explanations must be consistent with the
verdicts.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tecan_growth import load_plate, preprocess_plate
from tecan_growth.analyze import (analyze_curve, analyze_plate, doubling_time_h,
                                  legacy_results_frame, RESULT_COLUMNS)
from tecan_growth.settings import AnalysisSettings

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def group3_fast():
    pp = preprocess_plate(load_plate(RAW / "Group 3" / "GROUP3_VALUES.csv", RAW / "Group 3" / "GROUP3_KEY.csv"))
    return analyze_plate(pp, AnalysisSettings.from_config(use_adaptive=False))


class TestGoldenRegression:
    def test_group1_adaptive_matches_cli_reference(self):
        """
        Golden file produced by:
          python scripts/01_growth_curve_analysis.py data/raw/Group1/Group_1_DATA --adaptive --no-plots -q
        before analyze_curve() was factored out. The in-memory path (xlsx/csv reader ->
        preprocess -> analyze_plate) must reproduce it exactly.
        """
        ref = pd.read_csv(FIXTURES / "group1_adaptive_reference.csv").sort_values("strain").reset_index(drop=True)
        pp = preprocess_plate(load_plate(RAW / "Group1" / "GrowthRate_ Group1_Values.csv", RAW / "Group1" / "GROUP1_KEY_V2.csv"))
        res = analyze_plate(pp, AnalysisSettings.from_config(use_adaptive=True))
        got = legacy_results_frame(res).sort_values("strain").reset_index(drop=True)
        assert list(got.strain) == list(ref.strain)
        assert list(got.is_good) == list(ref.is_good)
        assert list(got.classification_reason) == list(ref.classification_reason)
        for col in ["gompertz_a", "gompertz_mu", "gompertz_lambda", "gompertz_a_err", "gompertz_mu_err",
                    "fit_r_squared", "fit_rmse", "truncation_time", "points_used", "delta_od", "max_od", "snr",
                    "baseline_std", "monotone_fraction", "residual_autocorr"]:
            np.testing.assert_allclose(got[col].astype(float).values, ref[col].astype(float).values,
                                       rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)


class TestPlateResults:
    def test_table_shape_and_columns(self, group3_fast):
        t = group3_fast.table
        assert list(t.columns) == RESULT_COLUMNS
        assert len(t) == 24
        assert set(t.stage) <= {"classified", "prefit_rejected", "truncation_invalid", "too_few_points"}
        assert t.loc[t.stage == "classified", "fit_r_squared"].notna().all()
        assert t.loc[t.stage != "classified", "gompertz_a"].isna().all()

    def test_lb_controls_grow_and_pass(self, group3_fast):
        t = group3_fast.table.set_index("curve_id")
        for cid in ["LB-IMID2", "LB-IMID5", "LB-LCY01"]:
            assert t.loc[cid, "is_good"] and t.loc[cid, "gompertz_a"] > 1.0

    def test_explanations_match_verdicts(self, group3_fast):
        for cid, cr in group3_fast.curves.items():
            text = cr.explanation()
            assert text
            fails = cr.failed_gates()
            if cr.is_good:
                assert not fails, f"{cid} is GOOD but has failed gates {[g.name for g in fails]}"
                assert "All quality gates passed" in text
            elif cr.legacy.stage == "classified" and cr.fit_succeeded:
                assert fails, f"{cid} is BAD after a fit but no gate is marked failed: {cr.legacy.classification.reason}"

    def test_failed_gate_names_agree_with_reason_text(self, group3_fast):
        keyword = {"r_squared": "R²", "a_err_pct": "A error", "mu_err_pct": "mu error", "snr": "SNR",
                   "delta_od_ci": "not significant", "delta_od": "Insufficient growth",
                   "amplitude_vs_noise": "below noise", "monotone": "Non-monotonic", "residual_autocorr": "autocorrelation"}
        for cid, cr in group3_fast.curves.items():
            if cr.legacy.stage != "classified":
                continue
            reason = cr.legacy.classification.reason
            for g in cr.failed_gates():
                if g.name in keyword:
                    assert keyword[g.name] in reason, f"{cid}: gate {g.name} failed but reason says: {reason}"

    def test_curve_result_fitted_curve(self, group3_fast):
        cr = group3_fast.curve("LB-IMID2")
        t, y = cr.fitted_curve(50)
        assert len(t) == 50 and y.max() <= cr.fit.a_opt * 1.0001
        assert cr.wells == ["A4", "A5", "A6"] and cr.n_replicates == 3

    def test_subset_and_progress(self):
        pp = preprocess_plate(load_plate(RAW / "Group 3" / "GROUP3_VALUES.csv", RAW / "Group 3" / "GROUP3_KEY.csv"))
        seen = []
        res = analyze_plate(pp, AnalysisSettings.from_config(use_adaptive=False), curve_ids=["LB-IMID2", "H2O-IMID2"],
                            progress=lambda d, n, c: seen.append((d, n, c)))
        assert list(res.table.curve_id) == ["LB-IMID2", "H2O-IMID2"]
        assert seen[-1][:2] == (2, 2)

    def test_parallel_matches_sequential(self):
        pp = preprocess_plate(load_plate(RAW / "Group 3" / "GROUP3_VALUES.csv", RAW / "Group 3" / "GROUP3_KEY.csv"))
        s = AnalysisSettings.from_config(use_adaptive=False)
        a = analyze_plate(pp, s, n_jobs=1).table
        b = analyze_plate(pp, s, n_jobs=2).table
        pd.testing.assert_frame_equal(a, b)


class TestAnalyzeCurve:
    def test_recovers_parameters_from_clean_gompertz(self):
        from tecan_growth.legacy import growth_curve_analysis
        gca = growth_curve_analysis()
        t = np.arange(0, 40, 0.25)
        rng = np.random.default_rng(1)
        y = gca.gompertz_model(t, 1.2, 0.2, 4.0) + rng.normal(0, 0.005, len(t))
        cr = analyze_curve(t, np.maximum(y, 0), "synthetic", AnalysisSettings.from_config(use_adaptive=False))
        assert cr.is_good
        assert abs(cr.fit.a_opt - 1.2) < 0.05 and abs(cr.fit.mu_opt - 0.2) < 0.03 and abs(cr.fit.lambda_opt - 4.0) < 0.5

    def test_flat_curve_is_rejected_before_fitting(self):
        t = np.arange(0, 40, 0.25)
        y = 0.02 + np.random.default_rng(0).normal(0, 0.002, len(t))
        cr = analyze_curve(t, np.maximum(y, 0), "flat", AnalysisSettings.from_config(use_adaptive=False))
        assert not cr.is_good and cr.legacy.stage == "prefit_rejected"
        assert "Rejected before fitting" in cr.explanation()

    def test_too_few_points(self):
        cr = analyze_curve(np.arange(5.0), np.linspace(0, 1, 5), "tiny", AnalysisSettings.from_config())
        assert not cr.is_good and cr.legacy.stage == "too_few_points" and cr.row["stage"] == "too_few_points"

    def test_nan_values_are_dropped(self):
        from tecan_growth.legacy import growth_curve_analysis
        gca = growth_curve_analysis()
        t = np.arange(0, 40, 0.25)
        y = gca.gompertz_model(t, 1.0, 0.2, 3.0)
        y[10] = np.nan
        cr = analyze_curve(t, y, "nan", AnalysisSettings.from_config(use_adaptive=False))
        assert len(cr.od) == len(t) - 1 and cr.is_good


class TestDoublingTime:
    def test_formula(self):
        # specific rate at inflection = mu*e/A -> doubling = ln2 * A / (mu e)
        assert abs(doubling_time_h(1.0, 0.25) - np.log(2) / (0.25 * np.e)) < 1e-12
        assert doubling_time_h(1.0, 0.0) is None and doubling_time_h(None, 0.2) is None

    def test_consistent_with_gompertz_specific_rate(self):
        from tecan_growth.legacy import growth_curve_analysis
        gca = growth_curve_analysis()
        A, mu, lam = 1.3, 0.3, 2.0
        t = np.linspace(0, 30, 30001)
        y = gca.gompertz_model(t, A, mu, lam)
        spec = np.gradient(y, t) / y  # specific growth rate d ln(y)/dt
        i = np.argmax(np.gradient(y, t))  # inflection: fastest absolute growth
        assert abs(spec[i] - mu * np.e / A) / (mu * np.e / A) < 0.01
        assert abs(np.log(2) / spec[i] - doubling_time_h(A, mu)) / doubling_time_h(A, mu) < 0.01


class TestDeterminism:
    def test_adaptive_truncation_is_deterministic_across_calls(self):
        """
        Regression for a thread-timing bug: find_optimal_truncation_v2 consumed
        MCCV results in completion order and overwrote its running best score
        unconditionally, so the chosen start index (and hence A, mu, lambda and
        their standard errors) could change between two calls on the same data.
        """
        pp = preprocess_plate(load_plate(RAW / "Group1" / "GrowthRate_ Group1_Values.csv", RAW / "Group1" / "GROUP1_KEY_V2.csv"))
        t, y = pp.curve("FlupyradifuroneANDLB-FLUPC8")
        s = AnalysisSettings.from_config(use_adaptive=True)
        runs = [analyze_curve(t, y, "FLUPC8", s) for _ in range(3)]
        keys = [(r.truncation.start_index, len(r.truncation.od_truncated), r.fit.a_opt, r.fit.a_err, r.fit.mu_opt) for r in runs]
        assert keys[0] == keys[1] == keys[2], keys
