"""
Exports: the tables a run produces, the folder it leaves behind, and the
promise that the folder is enough to redo the analysis without this software.

The two claims worth protecting here are

* ``processed/*_DATA.csv`` + ``key.csv`` rerun through the command-line
  pipeline reproduce ``processing_results.csv`` to the last decimal, and
* reopening a run folder restores every verdict without recomputing anything.
"""
import io
import json
import os
import subprocess
import sys
import types
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
GROUP3 = ROOT / "data" / "raw" / "Group 3" / "GrowthRate_GROUP3.xlsx"


def _install_media_stub() -> None:
    """
    ``tecan_growth.media`` belongs to the plotting side of the package. When it
    is not present yet, stand in for it with the classification rules the
    exports rely on, so these tests exercise this module and not that one.
    """
    import tecan_growth

    try:
        import tecan_growth.media  # noqa: F401
        return
    except ImportError:
        pass

    module = types.ModuleType("tecan_growth.media")
    module.TREATMENT_TYPES = ["LB", "H2O", "Pesticide", "Pesticide+LB"]

    def classify_media(media: str) -> dict:
        name = str(media).strip()
        upper = name.upper()
        if upper == "LB":
            return {"treatment_type": "LB", "pesticide": None, "has_nutrients": True}
        if upper in ("H2O", "WATER"):
            return {"treatment_type": "H2O", "pesticide": None, "has_nutrients": False}
        if upper.endswith("ANDLB"):
            return {"treatment_type": "Pesticide+LB", "pesticide": name[:-5].title(), "has_nutrients": True}
        return {"treatment_type": "Pesticide", "pesticide": name.title(), "has_nutrients": False}

    module.classify_media = classify_media
    sys.modules["tecan_growth.media"] = module
    tecan_growth.media = module


_install_media_stub()

from tecan_growth import load_plate, preprocess_plate  # noqa: E402
from tecan_growth.analyze import analyze_plate, legacy_results_frame  # noqa: E402
from tecan_growth.export import (CLEAN_COLUMNS, EXPORT_COLUMNS, RUN_FOLDER_FILES,  # noqa: E402
                                 bundle_zip, clean_summary, curves_long, file_sha256,
                                 git_commit, load_run_folder, methods_paragraph,
                                 results_table, run_manifest, set_excluded_wells,
                                 set_truncation_override, set_verdict_override, write_run_folder)
from tecan_growth.analyze import RESULT_COLUMNS  # noqa: E402
from tecan_growth.settings import AnalysisSettings  # noqa: E402


@pytest.fixture(scope="module")
def plate():
    """The example plate, all the way through: raw -> processed -> analysed."""
    run = load_plate(GROUP3)
    processed = preprocess_plate(run)
    results = analyze_plate(processed, AnalysisSettings.from_config(use_adaptive=False))
    return run, run.key, processed, results


@pytest.fixture(scope="module")
def run_folder(plate, tmp_path_factory):
    run, key, processed, results = plate
    overrides = {}
    set_verdict_override(overrides, "LB-IMID2", "BAD", note="second growth phase, do not trust A")
    set_truncation_override(overrides, "LB-IMID5", 0, 120, previous=(0, 200), note="cut before the dip")
    set_excluded_wells(overrides, "H2O-IMID2", ["B6"], note="bubble")
    out = tmp_path_factory.mktemp("run") / "Group3_example"
    written = write_run_folder(out, run, key, processed, results, overrides=overrides,
                               input_sha256=file_sha256(GROUP3))
    return out, written, overrides


class TestResultsTable:
    def test_columns_and_labels(self, plate):
        _, _, processed, results = plate
        table = results_table(results, processed=processed)
        assert list(table.columns) == RESULT_COLUMNS + EXPORT_COLUMNS
        assert len(table) == 24
        assert set(table.treatment_type) == {"LB", "H2O", "Pesticide", "Pesticide+LB"}
        assert table.loc[table.media == "LB", "pesticide"].isna().all()
        assert set(table.loc[table.media == "IMIDACLOPRIDANDLB", "pesticide"]) == {"Imidacloprid"}
        assert set(table.blank_source) == {"per_media"}

    def test_final_call_defaults_to_the_software_verdict(self, plate):
        _, _, _, results = plate
        table = results_table(results).set_index("curve_id")
        for cid, row in table.iterrows():
            assert row.final_call == ("GOOD" if row.is_good else "BAD"), cid
            assert row.override_note == ""

    def test_final_call_follows_a_hand_set_verdict(self, plate):
        _, _, _, results = plate
        overrides = {}
        set_verdict_override(overrides, "LB-IMID2", "BAD", note="second growth phase")
        table = results_table(results, overrides).set_index("curve_id")
        assert table.loc["LB-IMID2", "is_good"], "the fixture curve should be GOOD before the override"
        assert table.loc["LB-IMID2", "final_call"] == "BAD"
        assert "second growth phase" in table.loc["LB-IMID2", "override_note"]
        assert table.loc["LB-IMID5", "final_call"] == "GOOD"

    def test_manual_cut_and_excluded_wells_are_visible(self, plate):
        _, _, _, results = plate
        overrides = {}
        set_truncation_override(overrides, "LB-IMID5", 0, 120)
        set_excluded_wells(overrides, "H2O-IMID2", ["B6"], note="bubble")
        table = results_table(results, overrides).set_index("curve_id")
        assert table.loc["LB-IMID5", "truncation_override"] == "0-120"
        assert table.loc["H2O-IMID2", "excluded_wells"] == "B6"
        assert "bubble" in table.loc["H2O-IMID2", "override_note"]


class TestCleanSummary:
    def test_columns_are_exactly_the_shared_schema(self, plate):
        _, _, _, results = plate
        clean = clean_summary(results)
        assert list(clean.columns) == CLEAN_COLUMNS

    def test_only_good_curves_unless_asked(self, plate):
        _, _, _, results = plate
        clean = clean_summary(results)
        assert len(clean) == results.n_good
        assert set(clean.Classification) == {"Good"}
        assert clean.Max_OD600_A.notna().all() and (clean.R_squared >= 0.9).all()
        both = clean_summary(results, include_bad=True)
        assert len(both) == results.n_total

    def test_treatment_and_strain_are_split_on_the_last_hyphen(self, plate):
        _, _, _, results = plate
        clean = clean_summary(results).set_index("Strain")
        row = clean.loc["LB-IMID2"]
        assert row.Treatment == "LB" and row.Strain_ID == "IMID2" and row.Treatment_Type == "LB"

    def test_doubling_time_matches_the_gompertz_definition(self, plate):
        _, _, _, results = plate
        clean = clean_summary(results)
        full = results_table(results).set_index("curve_id")
        for _, row in clean.iterrows():
            a = float(full.loc[row.Strain, "gompertz_a"])
            mu = float(full.loc[row.Strain, "gompertz_mu"])
            assert abs(row.Doubling_Time_hrs - np.log(2) * a / (mu * np.e)) < 1e-6, row.Strain

    def test_a_hand_set_bad_verdict_removes_the_row(self, plate):
        _, _, _, results = plate
        overrides = {}
        set_verdict_override(overrides, "LB-IMID2", "BAD", note="not trustworthy")
        clean = clean_summary(results, overrides)
        assert "LB-IMID2" not in set(clean.Strain)
        assert len(clean) == results.n_good - 1


@pytest.fixture(scope="module")
def long(plate):
    _, _, processed, results = plate
    return curves_long(processed, results)


class TestCurvesLong:

    def test_shape_and_columns(self, long, plate):
        _, _, processed, _ = plate
        assert list(long.columns) == ["time_h", "media", "strain", "curve_id", "well", "od_raw",
                                      "blank_mean", "od_blanked", "in_fit", "od_predicted", "residual"]
        assert len(long) == len(processed.replicates) + len(processed.curves)
        assert set(long.curve_id) == set(processed.conditions.curve_id)

    def test_mean_rows_are_the_series_that_was_fitted(self, long, plate):
        _, _, processed, results = plate
        mean = long[(long.curve_id == "LB-IMID2") & (long.well == "MEAN")]
        _, od = processed.curve("LB-IMID2")
        np.testing.assert_allclose(mean.od_blanked.to_numpy(), od)
        cr = results.curve("LB-IMID2")
        used = mean[mean.in_fit]
        assert len(used) == len(cr.truncation.od_truncated)
        np.testing.assert_allclose(used.od_blanked.to_numpy(), cr.truncation.od_truncated)
        np.testing.assert_allclose(used.residual.to_numpy(), cr.fit.residuals, atol=1e-9)

    def test_blank_subtraction_is_reproducible_from_the_file(self, long):
        wells = long[(long.curve_id == "LB-IMID2") & (long.well == "A4")]
        np.testing.assert_allclose(wells.od_blanked.to_numpy(),
                                   np.maximum(wells.od_raw - wells.blank_mean, 0.0).to_numpy())

    def test_unfitted_curves_have_no_prediction(self, long, plate):
        _, _, _, results = plate
        for cid, cr in results.curves.items():
            if not cr.fit_succeeded:
                assert long[long.curve_id == cid].od_predicted.isna().all(), cid
                assert not long[long.curve_id == cid].in_fit.any(), cid


class TestManifest:
    def test_provenance_fields(self, plate):
        run, key, processed, results = plate
        digest = file_sha256(GROUP3)
        manifest = run_manifest(run, key, processed, results, input_sha256=digest)
        assert manifest["input"]["sha256"] == digest and len(digest) == 64
        assert manifest["input"]["file"] == "GrowthRate_GROUP3.xlsx"
        assert manifest["input"]["sheets_used"] == ["Result sheet", "Key"]
        assert manifest["instrument"]["wavelength_nm"] == 600
        assert manifest["run_summary"]["n_cycles"] == 400
        assert manifest["settings"]["min_r_squared"] == 0.95
        assert manifest["classifier"]["ml_used"] is False
        assert manifest["counts"]["n_curves"] == 24
        assert manifest["counts"]["n_good"] == results.n_good
        assert manifest["environment"]["numpy"] == np.__version__

    def test_git_commit_is_recorded(self, plate):
        run, key, processed, results = plate
        manifest = run_manifest(run, key, processed, results)
        commit = manifest["software"]["git_commit"]
        assert commit == git_commit()
        if commit is not None:
            assert len(commit) == 40 and all(c in "0123456789abcdef" for c in commit)

    def test_manifest_is_json_serialisable(self, plate):
        run, key, processed, results = plate
        overrides = {}
        set_verdict_override(overrides, "LB-IMID2", "BAD", note="second growth phase")
        manifest = run_manifest(run, key, processed, results, overrides=overrides)
        text = json.dumps(manifest)
        again = json.loads(text)
        assert again["overrides"]["LB-IMID2"]["verdict"] == "BAD"
        assert again["overrides"]["LB-IMID2"]["verdict_timestamp"].endswith("Z")
        assert again["counts"]["n_overridden_verdicts"] == 1

    def test_changed_settings_are_called_out(self, plate):
        run, key, processed, _ = plate
        results = analyze_plate(processed, AnalysisSettings.from_config(use_adaptive=False, min_r_squared=0.9),
                                curve_ids=["LB-IMID2"])
        manifest = run_manifest(run, key, processed, results)
        assert manifest["settings_changed_from_defaults"] == {"min_r_squared": 0.9}


class TestMethodsParagraph:
    def test_contains_the_thresholds_actually_used(self, plate):
        _, _, _, results = plate
        text = methods_paragraph(results.settings, results)
        assert "Zwietering" in text and "1990" in text
        assert "0.95" in text and "20 %" in text and "0.15" in text
        assert "primary growth" in text and "25 %" in text
        assert f"{results.n_good}" in text

    def test_reflects_changed_thresholds(self):
        text = methods_paragraph(AnalysisSettings.interactive_defaults(min_r_squared=0.8, use_first_peak=False))
        assert "0.8" in text and "global maximum" in text


class TestRunFolder:
    def test_every_expected_file_is_written(self, run_folder):
        out, written, _ = run_folder
        names = {p.name for p in written}
        for expected in RUN_FOLDER_FILES:
            assert expected in names, expected
            assert (out / expected).exists()
        processed_dir = out / "processed"
        assert processed_dir.is_dir()
        assert {p.name for p in processed_dir.glob("*_DATA.csv")} == {
            "LB_DATA.csv", "H2O_DATA.csv", "IMIDACLOPRID_DATA.csv", "IMIDACLOPRIDANDLB_DATA.csv",
            "LAMBDACYHALOTHRIN_DATA.csv", "LAMBDACYHALOTHRINANDLB_DATA.csv"}

    def test_processing_results_matches_the_in_memory_results(self, run_folder, plate):
        out, _, _ = run_folder
        _, _, _, results = plate
        written = pd.read_csv(out / "processing_results.csv")
        expected = legacy_results_frame(results)
        assert list(written.columns) == list(expected.columns)
        np.testing.assert_allclose(written.gompertz_a.astype(float), expected.gompertz_a.astype(float),
                                   equal_nan=True, rtol=0, atol=1e-12)

    def test_key_csv_is_the_cli_format(self, run_folder):
        out, _, _ = run_folder
        key = pd.read_csv(out / "key.csv")
        assert list(key.columns) == ["Cell", "Media", "Strain"]
        assert len(key) == 96

    def test_workbook_has_the_expected_sheets(self, run_folder):
        out, _, _ = run_folder
        book = pd.read_excel(out / "report.xlsx", sheet_name=None)
        assert list(book) == ["Results", "Clean", "Curves", "Key", "Settings", "Overrides"]
        assert len(book["Results"]) == 24
        assert set(book["Overrides"].change) == {"verdict", "truncation", "excluded_wells"}
        assert "min_r_squared" in set(book["Settings"].setting)

    def test_figures_are_saved_offline(self, plate, tmp_path):
        go = pytest.importorskip("plotly.graph_objects")
        run, key, processed, results = plate
        fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
        out = tmp_path / "with_figures"
        write_run_folder(out, run, key, processed, results, figures={"overview": fig})
        html = (out / "figures" / "overview.html").read_text()
        # the plotting library is embedded, so the file opens with no internet
        assert "Plotly.newPlot" in html and len(html) > 1_000_000

    def test_reopening_restores_verdicts_and_overrides(self, run_folder):
        out, _, overrides = run_folder
        loaded = load_run_folder(out)
        assert set(loaded) == {"results_table", "manifest", "overrides"}
        table = loaded["results_table"].set_index("curve_id")
        assert table.loc["LB-IMID2", "final_call"] == "BAD"
        assert table.loc["LB-IMID2", "is_good"] in (True, "True", np.True_)
        assert loaded["overrides"]["LB-IMID2"]["note"] == overrides["LB-IMID2"]["note"]
        assert loaded["manifest"]["input"]["sha256"] == file_sha256(GROUP3)

    def test_missing_folder_is_reported_clearly(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a run folder"):
            load_run_folder(tmp_path)

    def test_bundle_zip_contains_the_whole_folder(self, run_folder):
        out, _, _ = run_folder
        blob = bundle_zip(out)
        with zipfile.ZipFile(io.BytesIO(blob)) as zf:
            names = set(zf.namelist())
        for expected in RUN_FOLDER_FILES:
            assert expected in names
        assert "processed/LB_DATA.csv" in names
        assert len(blob) > 1000


class TestCliReproduction:
    def test_processed_csvs_rerun_through_the_cli_reproduce_the_results(self, run_folder, tmp_path):
        """
        Acceptance check 10: the exported inputs must reproduce the exported
        results through the untouched command-line pipeline.
        """
        out, _, _ = run_folder
        cli_out = tmp_path / "cli"
        env = dict(os.environ, MPLBACKEND="Agg")
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "01_growth_curve_analysis.py"),
             str(out / "processed"), "--config", str(out / "config.yaml"), "-o", str(cli_out), "--no-plots", "-q"],
            cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600, check=False)
        assert proc.returncode == 0, proc.stderr[-2000:]

        cli = pd.read_csv(cli_out / "processing_results.csv").sort_values("strain").reset_index(drop=True)
        ours = pd.read_csv(out / "processing_results.csv").sort_values("strain").reset_index(drop=True)
        assert list(cli.strain) == list(ours.strain)
        assert list(cli.is_good) == list(ours.is_good)
        assert list(cli.classification_reason) == list(ours.classification_reason)
        for col in ["gompertz_a", "gompertz_a_err", "gompertz_mu", "gompertz_mu_err",
                    "gompertz_lambda", "gompertz_lambda_err", "fit_r_squared", "fit_rmse", "fit_mae",
                    "truncation_time", "points_used", "delta_od", "max_od", "snr", "baseline_std",
                    "monotone_fraction", "residual_autocorr", "delta_od_ci_lower"]:
            # CSV round-trips are exact to ~1e-16, which the curve fitter can amplify
            # to ~1e-9 relative on ill-conditioned curves (same tolerance as the golden test).
            np.testing.assert_allclose(cli[col].astype(float), ours[col].astype(float),
                                       rtol=1e-6, atol=1e-9, equal_nan=True, err_msg=col)
