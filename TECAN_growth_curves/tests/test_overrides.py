"""
Manual corrections: moving the truncation cut, dropping a bad replicate well,
re-applying changed thresholds without refitting, and recording who decided what.

The rules these tests protect:

* passing the automatic cut back in as an override must change nothing at all,
* an override must change only the window, never the fitting or the gates,
* :func:`reclassify` must give exactly what a full re-analysis would give.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tecan_growth import load_plate, preprocess_plate
from tecan_growth.analyze import analyze_curve, analyze_plate, reclassify, reclassify_plate
from tecan_growth.export import (overrides_frame, record_review, set_excluded_wells,
                                 set_truncation_override, set_verdict_override)
from tecan_growth.legacy import growth_curve_analysis
from tecan_growth.preprocess import exclude_wells
from tecan_growth.settings import AnalysisSettings

ROOT = Path(__file__).resolve().parent.parent
GROUP3 = ROOT / "data" / "raw" / "Group 3" / "GrowthRate_GROUP3.xlsx"

#: Threshold changes that must be reproducible by reclassify() alone.
PERTURBATIONS = [
    {},
    {"min_r_squared": 0.9},
    {"max_param_error_pct": 10.0},
    {"min_absolute_delta_od": 0.3},
]


@pytest.fixture(scope="module")
def processed():
    return preprocess_plate(load_plate(GROUP3))


@pytest.fixture(scope="module")
def settings():
    return AnalysisSettings.from_config(use_adaptive=False)


@pytest.fixture(scope="module")
def results(processed, settings):
    return analyze_plate(processed, settings)


def _fit_signature(cr):
    """The numbers a fit is judged by, as a comparable tuple."""
    f = cr.fit
    if f is None or not f.success:
        return (cr.legacy.stage, bool(cr.is_good), cr.legacy.classification.reason)
    return (cr.legacy.stage, bool(cr.is_good), f.a_opt, f.mu_opt, f.lambda_opt,
            f.a_err, f.mu_err, f.lambda_err, f.r_squared, f.rmse, f.mae)


class TestTruncationOverrideIsInvisibleWhenItMatchesTheAutomaticCut:
    def test_every_group3_curve_is_bit_identical(self, processed, settings):
        """Handing back the cut the software chose must reproduce it exactly."""
        checked = 0
        for cid in processed.conditions.curve_id:
            t, y = processed.curve(cid)
            auto = analyze_curve(t, y, cid, settings)
            if auto.truncation is None:
                continue
            window = (int(auto.truncation.start_index), int(auto.truncation.truncation_index))
            forced = analyze_curve(t, y, cid, settings, truncation_override=window)
            assert _fit_signature(forced) == _fit_signature(auto), cid
            np.testing.assert_array_equal(forced.truncation.time_truncated, auto.truncation.time_truncated)
            np.testing.assert_array_equal(forced.truncation.od_truncated, auto.truncation.od_truncated)
            np.testing.assert_array_equal(forced.truncation.smoothed_od, auto.truncation.smoothed_od)
            assert forced.row["classification_reason"] == auto.row["classification_reason"]
            assert forced.truncation_override == window and auto.truncation_override is None
            checked += 1
        assert checked == 24

    def test_core_helper_matches_truncate_at_max(self, processed, settings):
        gca = growth_curve_analysis()
        t, y = processed.curve("LB-IMID2")
        auto = gca.truncate_at_max(t, y, **{k: v for k, v in settings.truncation_params().items()
                                            if k in ("smoothing_window", "buffer_points", "use_first_peak", "use_adaptive")})
        manual = gca.truncation_from_indices(t, y, auto.start_index, auto.truncation_index,
                                             smoothing_window=settings.smoothing_window)
        assert manual.truncation_index == auto.truncation_index
        assert manual.truncation_time == auto.truncation_time
        assert manual.start_index == auto.start_index
        np.testing.assert_array_equal(manual.time_truncated, auto.time_truncated)
        np.testing.assert_array_equal(manual.od_truncated, auto.od_truncated)
        np.testing.assert_array_equal(manual.smoothed_od, auto.smoothed_od)
        # max_od is the only field that may differ: the manual window reports the
        # highest measured OD it contains, the automatic cut the peak it detected
        # before adding its buffer points. Nothing numerical depends on it.
        assert manual.max_od == np.max(auto.od_truncated) >= auto.max_od


class TestTruncationOverrideChangesTheWindow:
    def test_earlier_cut_uses_fewer_points(self, processed, settings):
        t, y = processed.curve("LB-IMID2")
        auto = analyze_curve(t, y, "LB-IMID2", settings)
        end = int(auto.truncation.truncation_index)
        earlier = end - 15
        assert earlier + 1 >= settings.min_points_after_truncation, "the earlier cut must stay fittable"
        cut = analyze_curve(t, y, "LB-IMID2", settings, truncation_override=(0, earlier))
        assert cut.row["points_used"] == earlier + 1 < auto.row["points_used"]
        assert cut.truncation.truncation_time < auto.truncation.truncation_time
        assert cut.fit_succeeded and cut.fit.a_opt != auto.fit.a_opt

    def test_too_short_a_window_is_refused_with_a_reason(self, processed, settings):
        t, y = processed.curve("LB-IMID2")
        cr = analyze_curve(t, y, "LB-IMID2", settings, truncation_override=(0, 5))
        assert cr.legacy.stage == "truncation_invalid" and not cr.is_good
        assert "Too few points" in cr.legacy.classification.reason

    def test_indices_are_clipped_not_crashed(self, processed, settings):
        t, y = processed.curve("LB-IMID2")
        cr = analyze_curve(t, y, "LB-IMID2", settings, truncation_override=(-10, 10_000))
        assert cr.truncation.start_index == 0
        assert cr.truncation.truncation_index == len(t) - 1


class TestReclassify:
    @pytest.mark.parametrize("overrides", PERTURBATIONS)
    def test_matches_a_full_reanalysis_on_every_group3_curve(self, processed, results, overrides):
        new = AnalysisSettings.from_config(use_adaptive=False, **overrides)
        for cid in processed.conditions.curve_id:
            t, y = processed.curve(cid)
            fresh = analyze_curve(t, y, cid, new,
                                  media=results.curve(cid).media, strain=results.curve(cid).strain,
                                  wells=results.curve(cid).wells)
            quick = reclassify(results.curve(cid), new)
            assert quick.is_good == fresh.is_good, cid
            assert quick.legacy.classification.reason == fresh.legacy.classification.reason, cid
            assert [(g.name, g.status, g.value, g.threshold) for g in quick.gates] == \
                   [(g.name, g.status, g.value, g.threshold) for g in fresh.gates], cid
            for k, v in fresh.row.items():
                got = quick.row[k]
                if isinstance(v, float) and np.isnan(v):
                    assert got is None or (isinstance(got, float) and np.isnan(got)), f"{cid}.{k}"
                else:
                    assert got == v or (isinstance(v, float) and abs(got - v) < 1e-12), f"{cid}.{k}"

    def test_does_not_touch_the_original(self, results, settings):
        cr = results.curve("LB-IMID2")
        before = (cr.is_good, cr.settings.min_r_squared, cr.legacy.classification.reason)
        reclassify(cr, AnalysisSettings.from_config(min_r_squared=0.999, use_adaptive=False))
        assert (cr.is_good, cr.settings.min_r_squared, cr.legacy.classification.reason) == before

    def test_a_stricter_threshold_flips_curves_to_bad(self, results):
        strict = AnalysisSettings.from_config(use_adaptive=False, min_r_squared=0.999)
        flipped = [cid for cid, cr in results.curves.items()
                   if cr.is_good and not reclassify(cr, strict).is_good]
        assert flipped, "no curve reacted to an R2 threshold of 0.999"

    def test_plate_level_helper_matches_a_full_rerun(self, processed, results):
        new = AnalysisSettings.from_config(use_adaptive=False, min_r_squared=0.9)
        quick = reclassify_plate(results, new)
        fresh = analyze_plate(processed, new)
        assert list(quick.table.curve_id) == list(fresh.table.curve_id)
        assert list(quick.table.is_good) == list(fresh.table.is_good)
        assert list(quick.table.classification_reason) == list(fresh.table.classification_reason)
        assert quick.settings.min_r_squared == 0.9


class TestExcludeWells:
    def test_average_is_recomputed_without_the_dropped_well(self, processed):
        trimmed = exclude_wells(processed, "LB-IMID2", ["A6"])
        cond = trimmed.conditions.set_index("curve_id")
        assert cond.loc["LB-IMID2", "wells"] == ["A4", "A5"]
        assert cond.loc["LB-IMID2", "n_replicates"] == 2

        reps = processed.replicates
        raw = np.column_stack([reps[(reps.curve_id == "LB-IMID2") & (reps.well == w)].od_raw.to_numpy()
                               for w in ("A4", "A5")])
        blank = processed.curves[processed.curves.curve_id == "LB-IMID2"].blank_mean.to_numpy()
        expected = np.maximum(raw.mean(axis=1) - blank, 0.0)
        _, od = trimmed.curve("LB-IMID2")
        np.testing.assert_allclose(od, expected, rtol=0, atol=1e-12)

    def test_other_curves_and_the_original_are_untouched(self, processed):
        before = processed.curve("H2O-IMID2")[1].copy()
        trimmed = exclude_wells(processed, "LB-IMID2", ["A6"])
        np.testing.assert_array_equal(trimmed.curve("H2O-IMID2")[1], before)
        assert processed.conditions.set_index("curve_id").loc["LB-IMID2", "wells"] == ["A4", "A5", "A6"]
        assert len(trimmed.replicates) == len(processed.replicates) - len(processed.time_h)
        assert "excluded_wells" not in processed.settings

    def test_exclusion_is_recorded_for_the_manifest(self, processed):
        trimmed = exclude_wells(processed, "LB-IMID2", ["A6"])
        assert trimmed.settings["excluded_wells"] == {"LB-IMID2": ["A6"]}
        assert any("A6" in note and "LB-IMID2" in note for note in trimmed.qc)

    def test_unknown_wells_are_ignored_and_the_last_one_is_protected(self, processed):
        assert exclude_wells(processed, "LB-IMID2", ["Z9"]) is processed
        with pytest.raises(ValueError, match="at least one replicate"):
            exclude_wells(processed, "LB-IMID2", ["A4", "A5", "A6"])
        with pytest.raises(KeyError):
            exclude_wells(processed, "NO-SUCH-CURVE", ["A1"])

    def test_dropping_an_outlier_changes_the_analysis(self, processed, settings):
        trimmed = exclude_wells(processed, "LB-IMID2", ["A6"])
        before = analyze_plate(processed, settings, curve_ids=["LB-IMID2"]).curve("LB-IMID2")
        after = analyze_plate(trimmed, settings, curve_ids=["LB-IMID2"]).curve("LB-IMID2")
        assert after.n_replicates == 2 and before.n_replicates == 3
        assert after.fit.a_opt != before.fit.a_opt


class TestOverrideRecords:
    def test_a_verdict_change_needs_a_note_and_is_timestamped(self):
        ov = {}
        with pytest.raises(ValueError, match="note is required"):
            set_verdict_override(ov, "LB-IMID2", "BAD")
        with pytest.raises(ValueError, match="GOOD"):
            set_verdict_override(ov, "LB-IMID2", "MAYBE", note="hmm")
        set_verdict_override(ov, "LB-IMID2", "bad", note="second growth phase")
        assert ov["LB-IMID2"]["verdict"] == "BAD"
        assert ov["LB-IMID2"]["note"] == "second growth phase"
        assert ov["LB-IMID2"]["verdict_timestamp"].endswith("Z")
        set_verdict_override(ov, "LB-IMID2", None)
        assert "verdict" not in ov["LB-IMID2"]

    def test_cuts_exclusions_and_reviews_are_recorded(self):
        ov = {}
        set_truncation_override(ov, "LB-IMID2", 0, 180, previous=(0, 264), note="cut the second rise")
        set_excluded_wells(ov, "LB-IMID2", ["A6"], note="bubble")
        record_review(ov, "LB-IMID2", "wrong", note="looks fine to me")
        with pytest.raises(ValueError, match="audit_result"):
            record_review(ov, "LB-IMID2", "maybe")
        entry = ov["LB-IMID2"]
        assert entry["truncation"]["start_index"] == 0 and entry["truncation"]["end_index"] == 180
        assert entry["truncation"]["previous"] == [0, 264]
        assert entry["excluded_wells"]["wells"] == ["A6"]
        assert entry["review"]["audit_result"] == "wrong"
        for part in ("truncation", "excluded_wells", "review"):
            assert entry[part]["timestamp"].endswith("Z")

    def test_frame_has_one_row_per_change(self):
        ov = {}
        set_verdict_override(ov, "A-1", "GOOD", note="obviously grew")
        set_truncation_override(ov, "B-2", 0, 100)
        frame = overrides_frame(ov)
        assert list(frame.columns) == ["curve_id", "change", "value", "previous", "note", "timestamp"]
        assert set(frame.change) == {"verdict", "truncation"}
        assert overrides_frame(None).empty
        assert isinstance(frame, pd.DataFrame)
