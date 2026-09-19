"""
Regression tests for the orchestration findings of the adversarial audit.

Covered here (one class per finding group):

* data-io-01/03/05/08, orch-04 -- ``scripts/02_preprocess_raw_plate_data.py``
  rewritten on top of :mod:`tecan_growth`: case-insensitive blanks, no
  ``nan_DATA.csv``, header-driven time column, readable errors, NaN-tolerant
  averages.
* orch-01 -- ``run_full_pipeline.py`` exits non-zero when *any* requested step
  failed, and step 2 fails instead of "succeeding" with no data.
* orch-02, trunc-06 -- ``truncation.use_adaptive`` and ``ml_classifier.enabled``
  in config.yaml actually drive the orchestrator.
* orch-03 -- ``01_growth_curve_analysis.py`` takes its thresholds from
  config.yaml (and a ``--config`` file) instead of argparse defaults.
* orch-04, data-io-07 -- step 0 knows every group's raw files.
* orch-05 -- 05/06 read the group directories from config.yaml.
* mpl-01 -- ``ax.boxplot(..., tick_labels=)`` on matplotlib >= 3.9.
"""
import glob
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
RAW = ROOT / "data" / "raw"

ENV = dict(os.environ, MPLBACKEND="Agg")

#: group -> (raw values csv, raw key csv, committed processed dir)
GROUP_RAW = {
    "Group1": (RAW / "Group1" / "GrowthRate_ Group1_Values.csv",
               RAW / "Group1" / "GROUP1_KEY_V2.csv",
               RAW / "Group1" / "Group_1_DATA"),
    "Group2": (RAW / "Group 2" / "GrowthRate_GROUP2_DATA.csv",
               RAW / "Group 2" / "GrowthRate_GROUP2_KEY.csv",
               RAW / "Group 2" / "Group_2_DATA"),
    "Group3": (RAW / "Group 3" / "GROUP3_VALUES.csv",
               RAW / "Group 3" / "GROUP3_KEY.csv",
               RAW / "Group 3" / "Group_3_DATA"),
    "Group4": (RAW / "Group 4" / "G4_VALUES.csv",
               RAW / "Group 4" / "G4_KEY.csv",
               RAW / "Group 4" / "Group4_DATA"),
}


def _load(path, name):
    """Import a numerically-named pipeline script as a module."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pre():
    return _load(SCRIPTS / "02_preprocess_raw_plate_data.py", "preprocess_raw_plate_data")


@pytest.fixture(scope="module")
def pipeline():
    return _load(SCRIPTS / "run_full_pipeline.py", "run_full_pipeline_mod")


# =============================================================================
# data-io-01/03/05/08 -- scripts/02 is now a wrapper over the package
# =============================================================================

class TestPreprocessReproducesCommittedData:
    """Every group must round-trip from its raw export to the committed CSVs."""

    @pytest.mark.parametrize("group", sorted(GROUP_RAW))
    def test_regenerates_committed_data_csvs(self, group, pre, tmp_path):
        values, key, committed = GROUP_RAW[group]
        out = tmp_path / group
        written = pre.preprocess_group1(str(values), str(key), str(out), verbose=False)
        assert written, "no files written"

        names = {p.name for p in out.glob("*_DATA.csv")}
        expected = {Path(f).name for f in glob.glob(str(committed / "*_DATA.csv"))}
        assert names == expected, f"{group}: media files differ from the committed set"

        n_cols = 0
        for f in sorted(glob.glob(str(committed / "*_DATA.csv"))):
            old = pd.read_csv(f)
            new = pd.read_csv(out / Path(f).name)
            # Groups 3-4 were saved by the lab notebook without the '_blanked' suffix
            by_stem = {c.replace("_blanked", ""): c for c in new.columns}
            for c in old.columns:
                cn = by_stem.get(c.replace("_blanked", ""))
                assert cn is not None, f"{group}/{Path(f).name}: no column for {c}"
                np.testing.assert_allclose(new[cn].to_numpy(float), old[c].to_numpy(float),
                                           atol=1e-9, rtol=0,
                                           err_msg=f"{group}/{Path(f).name}/{c}")
                n_cols += 1
        assert n_cols > 0

    @pytest.mark.parametrize("group", sorted(GROUP_RAW))
    def test_no_nan_data_file_is_written(self, group, pre, tmp_path):
        """Key rows with an empty strain used to become a 'nan' media."""
        values, key, _ = GROUP_RAW[group]
        out = tmp_path / group
        pre.preprocess_group1(str(values), str(key), str(out), verbose=False)
        written = {p.name.lower() for p in out.glob("*.csv")}
        assert "nan_data.csv" not in written
        assert not [n for n in written if n.startswith("nan")]

    def test_uppercase_blank_label_is_subtracted(self, pre, tmp_path):
        """Groups 2-4 spell the blank 'BLANK'; the old script only matched 'Blank'."""
        values, key, _ = GROUP_RAW["Group3"]
        out = tmp_path / "g3"
        pre.preprocess_group1(str(values), str(key), str(out), verbose=False)
        raw = pd.read_csv(values)
        blanks = pd.read_csv(key)
        blank_wells = list(blanks.loc[
            (blanks.Media == "LB") & (blanks.Strain.astype(str).str.upper() == "BLANK"), "Cell"])
        assert blank_wells, "fixture no longer has uppercase BLANK wells"
        expected_blank = raw[blank_wells].to_numpy(float).mean(axis=1)
        # the first raw sample well of LB/IMID2 minus the blank must match the output
        wells = list(blanks.loc[(blanks.Media == "LB") & (blanks.Strain == "IMID2"), "Cell"])
        expected = np.maximum(raw[wells].to_numpy(float).mean(axis=1) - expected_blank, 0.0)
        got = pd.read_csv(out / "LB_DATA.csv")["LB_IMID2_blanked"].to_numpy(float)
        np.testing.assert_allclose(got, expected, atol=1e-12)
        assert expected_blank.max() > 0.05, "blank was not actually subtracted"


class TestPreprocessRobustness:
    """The failure modes the audit found on hand-edited exports."""

    @staticmethod
    def _write(tmp_path, values_text, key_text):
        v = tmp_path / "values.csv"
        k = tmp_path / "key.csv"
        v.write_text(textwrap.dedent(values_text).lstrip())
        k.write_text(textwrap.dedent(key_text).lstrip())
        return v, k

    def test_time_column_in_minutes_is_honoured(self, pre, tmp_path):
        v, k = self._write(
            tmp_path,
            """
            Cycle Nr.,Time [min],A1,A2,A3,B1,B2,B3
            1,0,0.10,0.10,0.10,0.20,0.20,0.20
            2,30,0.10,0.10,0.10,0.30,0.30,0.30
            3,60,0.10,0.10,0.10,0.50,0.50,0.50
            """,
            """
            Cell,Media,Strain
            A1,LB,BLANK
            A2,LB,BLANK
            A3,LB,BLANK
            B1,LB,S1
            B2,LB,S1
            B3,LB,S1
            """)
        pre.preprocess_group1(str(v), str(k), str(tmp_path / "out"), verbose=False)
        df = pd.read_csv(tmp_path / "out" / "LB_DATA.csv")
        np.testing.assert_allclose(df["TIME[H]"], [0.0, 0.5, 1.0])
        np.testing.assert_allclose(df["LB_S1_blanked"], [0.1, 0.2, 0.4])

    def test_missing_and_non_numeric_cells_do_not_poison_the_average(self, pre, tmp_path):
        v, k = self._write(
            tmp_path,
            """
            Cycle Nr.,Time [s],A1,A2,B1,B2
            1,0,0.10,0.10,0.20,0.30
            2,3600,0.10,0.10,,0.30
            3,7200,0.10,0.10,OVER,0.50
            """,
            """
            Cell,Media,Strain
            A1,LB,BLANK
            A2,LB,BLANK
            B1,LB,S1
            B2,LB,S1
            """)
        pre.preprocess_group1(str(v), str(k), str(tmp_path / "out"), verbose=False)
        od = pd.read_csv(tmp_path / "out" / "LB_DATA.csv")["LB_S1_blanked"].to_numpy(float)
        assert not np.isnan(od).any(), "a single missing cell wiped out the whole curve"
        np.testing.assert_allclose(od, [0.15, 0.20, 0.40])

    def test_key_well_missing_from_the_data_is_a_readable_error(self, pre, tmp_path):
        v, k = self._write(
            tmp_path,
            """
            Cycle Nr.,Time [s],A1,A2
            1,0,0.10,0.20
            2,3600,0.10,0.40
            """,
            """
            Cell,Media,Strain
            A1,LB,BLANK
            A2,LB,S1
            H12,LB,S2
            """)
        with pytest.raises(pre.PreprocessError) as exc:
            pre.preprocess_group1(str(v), str(k), str(tmp_path / "out"), verbose=False)
        assert "H12" in str(exc.value)
        assert "no data column" in str(exc.value)

    def test_cli_exits_non_zero_with_a_message_not_a_traceback(self, tmp_path):
        v, k = self._write(
            tmp_path,
            """
            Cycle Nr.,Time [s],A1,A2
            1,0,0.10,0.20
            2,3600,0.10,0.40
            """,
            """
            Cell,Media,Strain
            A1,LB,BLANK
            A2,LB,S1
            H12,LB,S2
            """)
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "02_preprocess_raw_plate_data.py"),
             str(v), str(k), "-o", str(tmp_path / "out")],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
        assert proc.returncode != 0
        assert "Traceback" not in proc.stderr
        assert "ERROR" in proc.stderr and "H12" in proc.stderr

    def test_cli_keeps_its_signature_and_quiet_flag(self, tmp_path):
        values, key, _ = GROUP_RAW["Group1"]
        loud = subprocess.run(
            [sys.executable, str(SCRIPTS / "02_preprocess_raw_plate_data.py"),
             str(values), str(key), "-o", str(tmp_path / "loud")],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=600)
        assert loud.returncode == 0, loud.stderr[-2000:]
        assert "Processing media" in loud.stdout

        quiet = subprocess.run(
            [sys.executable, str(SCRIPTS / "02_preprocess_raw_plate_data.py"),
             str(values), str(key), "--output", str(tmp_path / "quiet"), "-q"],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=600)
        assert quiet.returncode == 0, quiet.stderr[-2000:]
        assert quiet.stdout.strip() == ""
        assert (tmp_path / "quiet" / "LB_DATA.csv").exists()


# =============================================================================
# orch-01 -- exit code and step 2's "no data" case
# =============================================================================

def _fake_steps(ok_by_step, critical=()):
    """A STEPS registry of trivial step functions with scripted outcomes."""
    calls = []

    def make(num):
        def fn(config, dry_run=False, **kwargs):
            calls.append(num)
            return ok_by_step[num]
        return fn

    steps = [(n, make(n), f"step {n}", False, n in critical) for n in sorted(ok_by_step)]
    return steps, calls


class TestPipelineExitCode:
    def test_failing_non_critical_step_exits_one(self, pipeline, monkeypatch, capsys):
        steps, calls = _fake_steps({2: True, 4: False, 6: True}, critical={2})
        monkeypatch.setattr(pipeline, "STEPS", steps)
        monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--steps", "2,4,6", "--dry-run"])
        with pytest.raises(SystemExit) as exc:
            pipeline.main()
        assert exc.value.code == 1
        assert calls == [2, 4, 6], "a non-critical failure must not abort the rest"
        assert "FAILED steps: 4" in capsys.readouterr().out

    def test_all_steps_ok_exits_zero(self, pipeline, monkeypatch):
        steps, calls = _fake_steps({2: True, 4: True}, critical={2})
        monkeypatch.setattr(pipeline, "STEPS", steps)
        monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--steps", "2,4", "--dry-run"])
        pipeline.main()  # no SystemExit
        assert calls == [2, 4]

    def test_failing_critical_step_aborts_and_exits_one(self, pipeline, monkeypatch):
        steps, calls = _fake_steps({2: False, 4: True}, critical={2})
        monkeypatch.setattr(pipeline, "STEPS", steps)
        monkeypatch.setattr(sys, "argv", ["run_full_pipeline.py", "--steps", "2,4", "--dry-run"])
        with pytest.raises(SystemExit) as exc:
            pipeline.main()
        assert exc.value.code == 1
        assert calls == [2], "a critical failure must abort the remaining steps"


class TestStep2WithoutData:
    def test_missing_data_dir_is_a_failure(self, pipeline, tmp_path):
        config = {"groups": {"GroupX": {"data_dir": str(tmp_path / "nope"),
                                        "output_dir": str(tmp_path / "out")}}}
        assert pipeline.step2_gompertz(config, dry_run=True) is False

    def test_no_groups_at_all_is_a_failure(self, pipeline):
        assert pipeline.step2_gompertz({}, dry_run=True) is False
        assert pipeline.step2_gompertz({"groups": {}}, dry_run=True) is False


# =============================================================================
# orch-02 / trunc-06 -- config.yaml drives adaptive truncation and the ML gate
# =============================================================================

class TestAdaptiveAndMlResolution:
    def test_adaptive_comes_from_config(self, pipeline):
        assert pipeline.resolve_use_adaptive({"truncation": {"use_adaptive": True}}) is True
        assert pipeline.resolve_use_adaptive({"truncation": {"use_adaptive": False}}) is False
        assert pipeline.resolve_use_adaptive({}) is False

    def test_cli_overrides_config(self, pipeline):
        cfg = {"truncation": {"use_adaptive": True}}
        assert pipeline.resolve_use_adaptive(cfg, adaptive=False) is False
        assert pipeline.resolve_use_adaptive({}, adaptive=True) is True
        assert pipeline.resolve_use_adaptive(cfg, adaptive=None) is True

    def test_ml_needs_config_and_models(self, pipeline, monkeypatch):
        monkeypatch.setattr(pipeline, "ml_models_exist", lambda: True)
        assert pipeline.resolve_use_ml({"ml_classifier": {"enabled": False}}) is False
        assert pipeline.resolve_use_ml({"ml_classifier": {"enabled": True}}) is True
        assert pipeline.resolve_use_ml({"ml_classifier": {"enabled": True}}, no_ml=True) is False
        assert pipeline.resolve_use_ml({"ml_classifier": {"enabled": False}}, force_ml=True) is True

        monkeypatch.setattr(pipeline, "ml_models_exist", lambda: False)
        assert pipeline.resolve_use_ml({"ml_classifier": {"enabled": True}}) is False
        assert pipeline.resolve_use_ml({}, force_ml=True) is False

    def test_repo_config_asks_for_adaptive_truncation(self):
        """The published analysis used adaptive truncation; config.yaml must say so."""
        cfg = yaml.safe_load((SCRIPTS / "config.yaml").read_text())
        assert cfg["truncation"]["use_adaptive"] is True

    @pytest.mark.parametrize("use_adaptive,expected,unexpected",
                             [(True, "--adaptive", "--no-adaptive"),
                              (False, "--no-adaptive", " --adaptive")])
    def test_dry_run_passes_the_configured_flag(self, tmp_path, use_adaptive, expected, unexpected):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "truncation": {"use_adaptive": use_adaptive},
            "ml_classifier": {"enabled": False},
            "groups": {"Group1": {"data_dir": "data/raw/Group1/Group_1_DATA",
                                  "output_dir": str(tmp_path / "res")}},
        }))
        out = _dry_run(["--steps", "2", "--config", str(cfg)])
        cmd = [l for l in out.splitlines() if "01_growth_curve_analysis.py" in l]
        assert cmd, out
        assert expected in cmd[0]
        assert unexpected not in cmd[0]
        assert "--ml-classify" not in cmd[0]


def _dry_run(extra_args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_full_pipeline.py"), "--dry-run"] + extra_args,
        cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    return proc.stdout


# =============================================================================
# orch-03 -- 01_growth_curve_analysis.py honours config.yaml and --config
# =============================================================================

@pytest.fixture(scope="module")
def g3_lb(tmp_path_factory):
    """A one-file data directory holding Group 3's LB plate (LB-LCY2 has R^2 ~ 0.947)."""
    d = tmp_path_factory.mktemp("g3lb")
    (d / "LB_DATA.csv").write_bytes((RAW / "Group 3" / "Group_3_DATA" / "LB_DATA.csv").read_bytes())
    return d


class TestScript01ConfigResolution:
    """
    Group 3's LB-LCY2 fits with R^2 ~ 0.947 under first-peak truncation, so
    lowering ``classification.min_r_squared`` from 0.95 to 0.90 must flip its
    verdict. Before the fix, argparse's default silently won and the config
    file made no difference at all.
    """

    @staticmethod
    def _config(path, min_r2):
        path.write_text(yaml.safe_dump({
            "classification": {"min_r_squared": min_r2, "max_param_error_pct": 20.0,
                               "min_points_for_fit": 15},
            "truncation": {"use_adaptive": False, "use_first_peak": True},
        }))
        return path

    @staticmethod
    def _run(data_dir, out_dir, *extra):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "01_growth_curve_analysis.py"), str(data_dir),
             "-o", str(out_dir), "--no-plots", "-q", *extra],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
        return pd.read_csv(Path(out_dir) / "processing_results.csv").set_index("strain")

    def test_min_r_squared_from_config_changes_a_verdict(self, g3_lb, tmp_path):
        strict = self._run(g3_lb, tmp_path / "strict",
                           "--config", str(self._config(tmp_path / "c95.yaml", 0.95)))
        loose = self._run(g3_lb, tmp_path / "loose",
                          "--config", str(self._config(tmp_path / "c90.yaml", 0.90)))

        r2 = float(strict.loc["LB-LCY2", "fit_r_squared"])
        assert 0.90 < r2 < 0.95, f"fixture curve moved (R^2={r2})"
        assert bool(strict.loc["LB-LCY2", "is_good"]) is False
        assert bool(loose.loc["LB-LCY2", "is_good"]) is True
        # nothing else moved
        others = [s for s in strict.index if s != "LB-LCY2"]
        assert list(strict.loc[others, "is_good"]) == list(loose.loc[others, "is_good"])

    def test_cli_still_beats_the_config_file(self, g3_lb, tmp_path):
        res = self._run(g3_lb, tmp_path / "cli",
                        "--config", str(self._config(tmp_path / "c90.yaml", 0.90)),
                        "--min-r2", "0.99")
        assert bool(res.loc["LB-LCY2", "is_good"]) is False

    def test_empty_config_reproduces_the_module_defaults(self, g3_lb, tmp_path):
        """With no usable config file the behaviour must be what it was before."""
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        defaults = self._run(g3_lb, tmp_path / "defaults", "--config", str(empty))
        strict = self._run(g3_lb, tmp_path / "strict2",
                           "--config", str(self._config(tmp_path / "c95.yaml", 0.95)))
        assert list(defaults["is_good"]) == list(strict["is_good"])
        np.testing.assert_allclose(defaults["fit_r_squared"].to_numpy(float),
                                   strict["fit_r_squared"].to_numpy(float), rtol=0, atol=1e-12)

    def test_missing_config_file_is_reported(self, g3_lb, tmp_path):
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "01_growth_curve_analysis.py"), str(g3_lb),
             "-o", str(tmp_path / "x"), "--config", str(tmp_path / "absent.yaml")],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 2
        assert "config file not found" in proc.stderr

    def test_truncation_flags_resolve_from_config(self, g3_lb, tmp_path):
        cfg = tmp_path / "adaptive.yaml"
        cfg.write_text(yaml.safe_dump({"truncation": {"use_adaptive": True}}))
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "01_growth_curve_analysis.py"), str(g3_lb),
             "-o", str(tmp_path / "a"), "--no-plots", "-q", "--config", str(cfg), "--no-adaptive"],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stderr[-2000:]
        assert "FIRST PEAK" in proc.stdout, proc.stdout


class TestConfigForwarding:
    def test_config_path_reaches_the_sub_scripts(self, tmp_path):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "truncation": {"use_adaptive": False},
            "ml_classifier": {"enabled": False},
            "groups": {"Group1": {"data_dir": "data/raw/Group1/Group_1_DATA",
                                  "output_dir": str(tmp_path / "res")}},
            "paths": {"results": str(tmp_path / "res")},
        }))
        out = _dry_run(["--steps", "2,5,7", "--config", str(cfg)])
        for script in ("01_growth_curve_analysis.py", "06_advanced_fitting.py",
                       "07_compare_truncation_methods.py"):
            line = [l for l in out.splitlines() if script in l]
            assert line, f"{script} was not run\n{out}"
            assert f"--config {cfg}" in line[0], line[0]


# =============================================================================
# orch-04 / data-io-07 -- step 0 covers every group
# =============================================================================

class TestStep0AllGroups:
    @staticmethod
    def _config(tmp_path, group_dirs):
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({
            "groups": {
                name: {"data_dir": str(d),
                       "raw_values": str(GROUP_RAW[name][0].relative_to(ROOT)),
                       "raw_key": str(GROUP_RAW[name][1].relative_to(ROOT)),
                       "output_dir": str(tmp_path / f"{name}_res")}
                for name, d in group_dirs.items()},
        }))
        return cfg

    def test_every_missing_group_is_rebuilt(self, tmp_path):
        cfg = self._config(tmp_path, {g: tmp_path / g for g in GROUP_RAW})
        out = _dry_run(["--steps", "0", "--config", str(cfg)])
        for name, (values, key, _) in GROUP_RAW.items():
            line = [l for l in out.splitlines()
                    if "02_preprocess_raw_plate_data.py" in l and str(tmp_path / name) in l]
            assert line, f"{name} was not preprocessed\n{out}"
            assert str(values) in line[0] and str(key) in line[0]

    def test_existing_group_is_skipped_unless_forced(self, tmp_path):
        existing = tmp_path / "Group2_done"
        existing.mkdir()
        cfg = self._config(tmp_path, {"Group2": existing, "Group3": tmp_path / "Group3"})

        out = _dry_run(["--steps", "0", "--config", str(cfg)])
        assert "Group2 already preprocessed" in out
        assert len([l for l in out.splitlines() if "02_preprocess_raw_plate_data.py" in l]) == 1

        forced = _dry_run(["--steps", "0", "--config", str(cfg), "--force-preprocess"])
        assert len([l for l in forced.splitlines() if "02_preprocess_raw_plate_data.py" in l]) == 2

    def test_repo_config_lists_raw_files_for_every_group(self):
        cfg = yaml.safe_load((SCRIPTS / "config.yaml").read_text())
        for name, entry in cfg["groups"].items():
            for field in ("raw_values", "raw_key"):
                assert field in entry, f"{name} has no {field}"
                assert (ROOT / entry[field]).exists(), f"{name}.{field} -> {entry[field]} missing"

    def test_step0_round_trips_the_real_groups(self, pipeline, tmp_path):
        """Rebuilding into a fresh directory reproduces the committed CSVs."""
        cfg = yaml.safe_load((SCRIPTS / "config.yaml").read_text())
        groups = {name: {**entry, "data_dir": str(tmp_path / name)}
                  for name, entry in cfg["groups"].items()}
        assert pipeline.step0_preprocess({"groups": groups}) is True
        for name, (_, _, committed) in GROUP_RAW.items():
            built = {p.name for p in (tmp_path / name).glob("*_DATA.csv")}
            assert built == {Path(f).name for f in glob.glob(str(committed / "*_DATA.csv"))}


# =============================================================================
# orch-05 -- group directories come from config.yaml in 05 and 06
# =============================================================================

@pytest.fixture(scope="module", params=["05_haldane_analysis.py", "06_advanced_fitting.py"])
def mod(request):
    """05 and 06 both resolve group directories; both must read config.yaml."""
    return _load(SCRIPTS / request.param, "audit_" + request.param.split("_")[0])


class TestGroupDirsFromConfig:
    def test_directories_follow_the_config(self, mod):
        cfg = {"paths": {"raw_data": "data/raw"},
               "groups": {"Group2": {"data_dir": "data/raw/Renamed/G2_DATA"}}}
        assert mod.resolve_group_dir(RAW, "Group2", config=cfg) == RAW / "Renamed" / "G2_DATA"

    def test_literal_map_is_only_a_fallback(self, mod):
        assert mod.resolve_group_dir(RAW, "Group2", config={}) == RAW / "Group 2" / "Group_2_DATA"
        assert mod.resolve_group_dir(RAW, "Group4", config={}) == RAW / "Group 4" / "Group4_DATA"

    def test_repo_config_resolves_to_the_real_directories(self, mod):
        for group in GROUP_RAW:
            p = mod.resolve_group_dir(RAW, group)
            assert p.exists(), f"{group} -> {p}"
            assert p == GROUP_RAW[group][2]

    def test_load_raw_data_still_finds_a_curve(self, mod):
        time, od = mod.load_raw_data(RAW, "LB-IMID2", "Group3")
        assert time is not None and len(time) == len(od) > 100


# =============================================================================
# mpl-01 -- boxplot tick labels on matplotlib >= 3.9
# =============================================================================

class TestBoxplotTickLabels:
    def test_no_script_still_passes_labels_to_boxplot(self):
        import re
        for script in ("03_statistical_analysis.py", "07_compare_truncation_methods.py",
                       "03_mle_model_fitting.py", "05_haldane_analysis.py"):
            text = (SCRIPTS / script).read_text()
            for call in re.findall(r"boxplot\((?:[^()]|\([^()]*\))*\)", text, re.S):
                assert "labels=" not in call.replace("tick_labels=", ""), f"{script}: {call}"

    def test_statistical_analysis_runs_end_to_end(self, tmp_path):
        results = ROOT / "results" / "tables" / "all_groups_results.csv"
        if not results.exists():
            pytest.skip("results/tables/all_groups_results.csv not present")
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "03_statistical_analysis.py"),
             "--input", str(results), "--output-dir", str(tmp_path)],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
        assert (tmp_path / "results_for_R_import.csv").exists()
        pngs = list(tmp_path.rglob("*.png"))
        assert pngs, "no figure was written"
        assert any("boxplot" in p.name for p in pngs)
