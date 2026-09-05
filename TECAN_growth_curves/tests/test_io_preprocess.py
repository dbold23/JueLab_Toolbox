"""
Tests for tecan_growth.io / plate / preprocess against the bundled raw exports.

The committed ``data/raw/Group*/..._DATA/*_DATA.csv`` files were produced by the
lab's original notebook (Groups 2-4) and by scripts/02_preprocess_raw_plate_data.py
(Group 1). The new reader + preprocessing must reproduce them exactly.
"""
import glob
import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tecan_growth import (PlateKey, load_plate, preprocess_plate, read_key,
                          read_tecan_export, read_values_csv)
from tecan_growth.plate import normalize_well, well_sort_key, PLATE_96_WELLS

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

GROUPS = {
    # name: (xlsx export, values csv, key csv, committed processed dir)
    "Group1": (RAW / "Group1" / "GrowthRate_JueLab (Modified)_20250304_134357.xlsx",
               RAW / "Group1" / "GrowthRate_ Group1_Values.csv",
               RAW / "Group1" / "GROUP1_KEY_V2.csv",
               RAW / "Group1" / "Group_1_DATA"),
    "Group2": (RAW / "Group 2" / "GrowthRate_JueLab (Modified)_20250310_170415.xlsx",
               RAW / "Group 2" / "GrowthRate_GROUP2_DATA.csv",
               RAW / "Group 2" / "GrowthRate_GROUP2_KEY.csv",
               RAW / "Group 2" / "Group_2_DATA"),
    "Group3": (RAW / "Group 3" / "GrowthRate_GROUP3.xlsx",
               RAW / "Group 3" / "GROUP3_VALUES.csv",
               RAW / "Group 3" / "GROUP3_KEY.csv",
               RAW / "Group 3" / "Group_3_DATA"),
    "Group4": (RAW / "Group 4" / "GrowthRate_JueLab (Modified)_20250324_132631.xlsx",
               RAW / "Group 4" / "G4_VALUES.csv",
               RAW / "Group 4" / "G4_KEY.csv",
               RAW / "Group 4" / "Group4_DATA"),
}


def _assert_reproduces_legacy(processed, legacy_dir):
    n_cols = 0
    for f in sorted(glob.glob(str(legacy_dir / "*_DATA.csv"))):
        old = pd.read_csv(f)
        media = Path(f).name.replace("_DATA.csv", "")
        new = processed.wide(media)
        for c in old.columns:
            # Groups 3-4 were saved by the notebook without the '_blanked' suffix
            cn = c if c in new.columns else f"{c}_blanked"
            assert cn in new.columns, f"{media}: column {c} missing from new output ({list(new.columns)})"
            np.testing.assert_allclose(new[cn].values, old[c].values, atol=1e-9, rtol=0,
                                       err_msg=f"{media}/{c} differs from committed data")
            n_cols += 1
    assert n_cols > 0


# --------------------------------------------------------------------------- wells

class TestWells:
    def test_normalize_well(self):
        assert normalize_well("A1") == "A1"
        assert normalize_well(" a01 ") == "A1"
        assert normalize_well("H12") == "H12"
        assert normalize_well("P24") == "P24"
        assert normalize_well("Cycle Nr.") is None
        assert normalize_well("Time [s]") is None
        assert normalize_well(None) is None
        assert normalize_well("A13") == "A13"  # 384-well plates go to 24
        assert normalize_well("A25") is None

    def test_sort_order_row_major(self):
        assert sorted(["B1", "A2", "A1", "A10"], key=well_sort_key) == ["A1", "A2", "A10", "B1"]

    def test_plate_96_wells_column_major_like_tecan(self):
        assert PLATE_96_WELLS[:3] == ["A1", "B1", "C1"]
        assert len(PLATE_96_WELLS) == 96


# ----------------------------------------------------------------------------- key

class TestPlateKey:
    def test_blank_labels_are_case_insensitive(self):
        df = pd.DataFrame({"Cell": ["A1", "A2", "A3"], "Media": ["LB", "LB", "LB"], "Strain": ["BLANK", "Blank", "MAL8"]})
        key = PlateKey.from_dataframe(df)
        assert key.blank_wells("LB") == ["A1", "A2"]
        assert key.wells_for("LB", "MAL8") == ["A3"]

    def test_unused_wells_are_kept_but_not_used(self):
        df = pd.DataFrame({"Cell": ["A1", "A2", "A3"], "Media": ["LB", "NA", None], "Strain": ["MAL8", "NA", None]})
        key = PlateKey.from_dataframe(df)
        assert list(key.table.in_use) == [True, False, False]
        assert key.media_names == ["LB"]

    def test_group2_csv_key_has_no_nan_media(self):
        key = read_key(GROUPS["Group2"][2])
        assert "NA" not in key.media_names and "nan" not in key.media_names
        assert len(key.blank_wells()) == 18
        assert set(key.media_names) == {"LB", "MALATHIONANDLB", "H2O", "MALATHION", "LAMBDACYHALOTHRIN", "LAMBDACYHALOTHRINANDLB"}

    def test_names_differing_by_case_are_merged_with_warning(self):
        df = pd.DataFrame({"Cell": ["A1", "A2", "A3"], "Media": ["Lambda-cyhalothrin", "Lambda-Cyhalothrin", "Bifenthrin "],
                           "Strain": ["LCY4", "LCY4", "BIF2"]})
        key = PlateKey.from_dataframe(df)
        assert key.wells_for("Lambda-cyhalothrin", "LCY4") == ["A1", "A2"]
        assert "Bifenthrin" in key.media_names
        assert any("merged" in w for w in key.warnings)

    def test_column_aliases(self):
        df = pd.DataFrame({"Well": ["A1"], "Medium": ["LB"], "Sample": ["X"]})
        key = PlateKey.from_dataframe(df)
        assert key.wells_for("LB", "X") == ["A1"]

    def test_bad_well_id_is_reported(self):
        df = pd.DataFrame({"Cell": ["A1", "Z9"], "Media": ["LB", "LB"], "Strain": ["X", "Y"]})
        key = PlateKey.from_dataframe(df)
        assert key.wells == ["A1"]
        assert any("Z9" in w for w in key.warnings)

    def test_validate_reports_missing_data_wells(self):
        df = pd.DataFrame({"Cell": ["A1", "A2"], "Media": ["LB", "LB"], "Strain": ["Blank", "X"]})
        key = PlateKey.from_dataframe(df)
        v = key.validate(["A1"])  # A2 has no data column
        assert not v.ok and "A2" in v.errors[0]

    def test_validate_warns_when_media_has_no_blank(self):
        df = pd.DataFrame({"Cell": ["A1", "A2"], "Media": ["LB", "H2O"], "Strain": ["Blank", "X"]})
        key = PlateKey.from_dataframe(df)
        v = key.validate(["A1", "A2"])
        assert v.ok
        assert any("H2O" in w and "no blank" in w for w in v.warnings)

    def test_plate_grid_shape(self):
        key = read_key(GROUPS["Group3"][2])
        grid = key.plate_grid("strain")
        assert grid.shape == (8, 12)
        assert grid.loc["A", 4] == "IMID2"


# --------------------------------------------------------------------------- reader

class TestReadTecanExport:
    @pytest.mark.parametrize("group", list(GROUPS))
    def test_reads_every_bundled_export(self, group):
        xlsx, csv, _, _ = GROUPS[group]
        run = read_tecan_export(xlsx)
        assert len(run.well_ids) == 96
        assert run.n_cycles > 90
        assert run.source["kinetic_sheet"] == "Result sheet"
        assert run.metadata["device"] == "Spark"
        assert run.metadata["wavelength_nm"] == 600
        assert run.metadata["interval_time"] == "00:15:00"
        assert run.metadata["kinetic_cycles"] in (96, 400)
        assert run.metadata["date"] and str(run.metadata["date"]).startswith("2025-03")
        assert run.metadata["start_time"] < run.metadata["end_time"]
        assert run.layout["A1"] == "BL1_1" and len(run.layout) == 96
        assert abs(run.interval_h * 60 - 15.0) < 0.05
        assert np.all(np.diff(run.time_h) > 0)
        assert run.time_h[0] == 0.0

    @pytest.mark.parametrize("group", list(GROUPS))
    def test_csv_values_equal_xlsx_values(self, group):
        xlsx, csv, _, _ = GROUPS[group]
        rx, rc = read_tecan_export(xlsx), read_values_csv(csv)
        assert rc.well_ids == rx.well_ids
        np.testing.assert_allclose(rc.wells.values, rx.wells.values, atol=0)
        np.testing.assert_allclose(rc.time_h, rx.time_h, atol=0)
        assert rc.temperature_c is not None and rx.temperature_c is not None

    def test_key_sheet_detected(self):
        run = read_tecan_export(GROUPS["Group3"][0])
        assert run.key is not None and run.source["key_sheet"] == "Key"
        assert len(run.key.blank_wells()) == 18

    def test_multiple_key_sheets_are_exposed(self):
        run = read_tecan_export(GROUPS["Group1"][0])
        assert set(run.key_candidates) == {"Bif_Flup_LCY_Schematic", "BIF_KEY"}

    def test_to_values_frame_round_trips(self):
        run = read_tecan_export(GROUPS["Group4"][0])
        frame = run.to_values_frame()
        assert list(frame.columns[:3]) == ["Cycle Nr.", "Time [s]", "Temp. [°C]"]
        again = read_values_csv(io.BytesIO(frame.to_csv(index=False).encode("utf-8")))
        np.testing.assert_allclose(again.wells.values, run.wells.values)
        np.testing.assert_allclose(again.time_h, run.time_h, atol=1e-12)

    def test_time_in_hours_header_is_converted(self):
        txt = "Time [h],A1,A2\n0,0.1,0.2\n0.25,0.11,0.21\n0.5,0.12,0.22\n"
        run = read_values_csv(io.BytesIO(txt.encode()))
        np.testing.assert_allclose(run.time_h, [0, 0.25, 0.5])
        assert any("interpreted as h" in w for w in run.warnings)

    def test_seconds_header_is_converted(self):
        txt = "Cycle Nr.,Time [s],Temp. [°C],A1\n1,0,37,0.1\n2,900,37,0.2\n"
        run = read_values_csv(io.BytesIO(txt.encode()))
        np.testing.assert_allclose(run.time_h, [0, 0.25])

    def test_unsorted_times_are_sorted(self):
        txt = "Cycle Nr.,Time [s],A1\n1,900,0.2\n2,0,0.1\n"
        run = read_values_csv(io.BytesIO(txt.encode()))
        np.testing.assert_allclose(run.time_h, [0, 0.25])
        np.testing.assert_allclose(run.wells["A1"].values, [0.1, 0.2])

    def test_missing_table_is_a_clear_error(self, tmp_path):
        p = tmp_path / "junk.csv"
        p.write_text("hello,world\n1,2\n")
        with pytest.raises(ValueError, match="kinetic table"):
            read_values_csv(p)

    def test_workbook_without_kinetic_table_is_a_clear_error(self):
        with pytest.raises(ValueError, match="no kinetic table"):
            read_tecan_export(RAW / "TECAN Growth Assay.xlsx")


# ----------------------------------------------------------------------- preprocess

class TestPreprocess:
    @pytest.mark.parametrize("group", ["Group2", "Group3", "Group4"])
    def test_reproduces_committed_data_from_csv_values_and_key(self, group):
        xlsx, csv, key, legacy = GROUPS[group]
        pp = preprocess_plate(load_plate(csv, key))
        _assert_reproduces_legacy(pp, legacy)

    @pytest.mark.parametrize("group", ["Group3", "Group4"])
    def test_reproduces_committed_data_from_xlsx_with_embedded_key(self, group):
        xlsx, csv, key, legacy = GROUPS[group]
        pp = preprocess_plate(read_tecan_export(xlsx))
        _assert_reproduces_legacy(pp, legacy)

    def test_group1_reproduces_02_script_output(self):
        xlsx, csv, key, legacy = GROUPS["Group1"]
        pp = preprocess_plate(load_plate(csv, key))
        _assert_reproduces_legacy(pp, legacy)
        assert len(pp.conditions) == 24

    def test_tidy_tables_are_consistent(self):
        xlsx, csv, key, legacy = GROUPS["Group2"]
        pp = preprocess_plate(load_plate(csv, key))
        n = len(pp.time_h)
        assert len(pp.curves) == n * len(pp.conditions)
        assert set(pp.curves.columns) >= {"curve_id", "media", "strain", "time_h", "od", "od_raw_mean", "od_sd", "n_replicates", "blank_mean"}
        assert (pp.conditions.n_replicates == 3).all()
        assert set(pp.conditions.blank_source) == {"per_media"}
        # od == clip(raw_mean - blank_mean, 0)
        expect = np.maximum(pp.curves.od_raw_mean - pp.curves.blank_mean, 0)
        np.testing.assert_allclose(pp.curves.od.values, expect.values, atol=1e-12)
        t, od = pp.curve("LB-MAL8")
        assert len(t) == n and od.max() > 0.5

    def test_clip_negative_false_keeps_noise_below_zero(self):
        xlsx, csv, key, legacy = GROUPS["Group2"]
        run = load_plate(csv, key)
        pp = preprocess_plate(run, clip_negative=False)
        assert (pp.curves.od < 0).any()
        pp2 = preprocess_plate(run, clip_negative=True)
        assert not (pp2.curves.od < 0).any()

    def test_missing_blank_falls_back_to_global_blanks_with_qc_message(self):
        run = load_plate(GROUPS["Group2"][1])
        df = pd.DataFrame({"Cell": ["A1", "A2", "A4", "B4"], "Media": ["LB", "LB", "LB", "H2O"],
                           "Strain": ["BLANK", "BLANK", "MAL8", "MAL8"]})
        pp = preprocess_plate(run, PlateKey.from_dataframe(df))
        row = pp.conditions.set_index("curve_id").loc["H2O-MAL8"]
        assert row.blank_source == "global" and row.n_blank_wells == 2
        assert any("H2O" in m and "whole plate" in m for m in pp.qc)

    def test_blank_mode_none(self):
        run = load_plate(GROUPS["Group2"][1], GROUPS["Group2"][2])
        pp = preprocess_plate(run, blank_mode="none")
        np.testing.assert_allclose(pp.curves.od.values, pp.curves.od_raw_mean.values)

    def test_key_mismatch_is_an_error(self):
        run = load_plate(GROUPS["Group2"][1])
        df = pd.DataFrame({"Cell": ["A1", "A2"], "Media": ["LB", "LB"], "Strain": ["BLANK", "MAL8"]})
        run.wells = run.wells.drop(columns=["A2"])
        with pytest.raises(ValueError, match="A2"):
            preprocess_plate(run, PlateKey.from_dataframe(df))

    def test_write_legacy_csvs(self, tmp_path):
        xlsx, csv, key, legacy = GROUPS["Group3"]
        pp = preprocess_plate(load_plate(csv, key))
        paths = pp.write_legacy_csvs(tmp_path)
        names = sorted(p.name for p in paths)
        assert names == sorted(Path(f).name for f in glob.glob(str(legacy / "*_DATA.csv")))
        back = pd.read_csv(tmp_path / "LB_DATA.csv")
        assert back.columns[0] == "TIME[H]"
