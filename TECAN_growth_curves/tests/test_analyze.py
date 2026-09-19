"""
Tests for tecan_growth.analyze: the new in-memory path must give the same
numbers as the CLI script, and its explanations must be consistent with the
verdicts.
"""
from pathlib import Path

import re

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


_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?|inf|nan")


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
        # The gate names and their order must match exactly. The numbers inside
        # the text are compared numerically: a degenerate (flat) curve gives a
        # relative error of ~1e11 % whose last digits are floating-point noise.
        for got_reason, ref_reason in zip(got.classification_reason, ref.classification_reason):
            assert _NUM.sub("#", got_reason) == _NUM.sub("#", ref_reason)
            got_nums = [float(x) for x in _NUM.findall(got_reason)]
            ref_nums = [float(x) for x in _NUM.findall(ref_reason)]
            np.testing.assert_allclose(got_nums, ref_nums, rtol=1e-4, err_msg=ref_reason)
        for col in ["gompertz_a", "gompertz_mu", "gompertz_lambda", "gompertz_a_err", "gompertz_mu_err",
                    "fit_r_squared", "fit_rmse", "truncation_time", "points_used", "delta_od", "max_od", "snr",
                    "baseline_std", "monotone_fraction", "residual_autocorr"]:
            got_v = got[col].astype(float).values
            ref_v = ref[col].astype(float).values
            if col.endswith("_err"):
                # A standard error thousands of times larger than the estimate
                # belongs to a degenerate (flat) fit; its digits are noise.
                param = ref[col[:-4]].astype(float).values
                degenerate = ref_v > 10 * np.abs(param)
                assert np.array_equal(got_v > 10 * np.abs(param), degenerate), col
                got_v, ref_v = got_v[~degenerate], ref_v[~degenerate]
            np.testing.assert_allclose(got_v, ref_v, rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)


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
        # LB-IMID3 and LB-IMID8 grow fast and then drift upward for 80 h; the
        # first-peak rule must cut them after the primary phase (the audited
        # cross-validated cuts are 14 h and 16 h), not fit the drift.
        for cid in ["LB-IMID2", "LB-IMID3", "LB-IMID5", "LB-IMID8", "LB-LCY01"]:
            assert t.loc[cid, "is_good"], f"{cid}: {t.loc[cid, 'classification_reason']}"
            assert t.loc[cid, "gompertz_a"] > 0.6
            assert t.loc[cid, "truncation_time"] < 40
        # LB-LCY2 is still rising when the run ends and its whole-run fit is poor.
        assert not t.loc["LB-LCY2", "is_good"]


class TestFirstPeakRule:
    def _curve(self, drift_od: float):
        t = np.arange(0, 60, 0.25)
        a, mu, lam = 1.0, 0.25, 3.0
        y = a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1))
        # A slow secondary rise after 14 h that levels off, as the LB-IMID3/8
        # wells of Group 3 do: it starts at ~8 % of the peak rate and fades.
        y = y + drift_od * (1 - np.exp(-np.clip(t - 14, 0, None) / 15))
        return t, y + np.random.default_rng(1).normal(0, 0.004, len(t))

    def test_primary_growth_is_cut_before_a_slow_drift(self):
        t, y = self._curve(0.3)
        cr = analyze_curve(t, y, "drift", AnalysisSettings.from_config(use_adaptive=False))
        assert cr.is_good, cr.explanation()
        assert 8 < cr.truncation.truncation_time < 20
        assert abs(cr.fit.mu_opt - 0.25) / 0.25 < 0.1
        assert abs(cr.fit.a_opt - 1.0) < 0.1

    def test_ideal_sigmoid_is_cut_near_its_ceiling(self):
        t, y = self._curve(0.0)
        cr = analyze_curve(t, y, "ideal", AnalysisSettings.from_config(use_adaptive=False))
        assert cr.is_good, cr.explanation()
        od_at_cut = float(y[cr.truncation.truncation_index])
        assert od_at_cut > 0.85                                  # ~90 % of A plus the buffer
        assert abs(cr.fit.a_opt - 1.0) < 0.05

    def test_fast_grower_in_a_long_run_is_fitted(self):
        # mu = 0.6 OD/h with a 0.8 h lag peaks at ~4 h; in a 99 h run that is 4 %
        # of the run, which the old "first 5 % of the run" rule rejected.
        t = np.arange(0, 99, 0.25)
        a, mu, lam = 1.3, 0.6, 0.8
        y = a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1)) + np.random.default_rng(3).normal(0, 0.004, len(t))
        cr = analyze_curve(t, y, "fast", AnalysisSettings.interactive_defaults())
        assert cr.legacy.stage == "classified", cr.explanation()
        assert cr.is_good, cr.explanation()
        assert abs(cr.fit.mu_opt - mu) / mu < 0.1

    def test_spike_at_the_start_is_not_a_growth_window(self):
        t = np.arange(0, 24, 0.25)
        y = np.full(len(t), 0.02) + np.random.default_rng(4).normal(0, 0.003, len(t))
        y[1:4] = [0.6, 0.9, 0.5]                                # a bubble in the first hour
        cr = analyze_curve(t, y, "spike", AnalysisSettings.interactive_defaults())
        assert not cr.is_good

    def test_slow_grower_is_not_cut_early(self):
        t = np.arange(0, 60, 0.25)
        a, mu, lam = 0.6, 0.03, 5.0                             # 0.03 OD/h: five times slower
        y = a * np.exp(-np.exp((mu * np.e / a) * (lam - t) + 1)) + np.random.default_rng(2).normal(0, 0.003, len(t))
        cr = analyze_curve(t, y, "slow", AnalysisSettings.from_config(use_adaptive=False))
        assert cr.is_good, cr.explanation()
        assert float(y[cr.truncation.truncation_index]) > 0.85 * a
        assert abs(cr.fit.mu_opt - mu) / mu < 0.1

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
