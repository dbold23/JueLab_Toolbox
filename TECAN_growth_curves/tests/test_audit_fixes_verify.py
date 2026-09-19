"""
Regression tests for the findings the end-to-end verification pass turned up.

* verify-02 -- ``07_compare_truncation_methods.py --config FILE`` crashed with
  ``AttributeError: 'str' object has no attribute 'exists'`` before it read
  anything: it passes argparse's string straight into 06's ``load_config()``,
  which only handled ``Path``. run_full_pipeline.py forwards ``--config`` to
  step 7, so step 7 died immediately on every custom-config run.
* verify-01 -- ``run_full_pipeline.py`` hands the configured ``paths.results``
  (and ``advanced.output_dir``) to the sub-scripts that cannot read config.yaml
  for themselves. Before this, steps 4, 5 and 7 wrote into the repository's
  ``results/tables`` no matter which config file the run was given, so a run
  pointed at a scratch tree still overwrote the published Haldane, ensemble and
  truncation-comparison outputs.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

ENV = dict(os.environ, MPLBACKEND="Agg")


def _dry_run(extra_args):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_full_pipeline.py"), "--dry-run"] + extra_args,
        cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    return proc.stdout


def _command_for(out, script):
    lines = [l for l in out.splitlines() if script in l and "Command:" in l]
    assert lines, f"{script} was not run\n{out}"
    return lines[0]


def _scratch_config(tmp_path):
    results = tmp_path / "results"
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.safe_dump({
        "truncation": {"use_adaptive": False},
        "ml_classifier": {"enabled": False},
        "paths": {"results": str(results)},
        "advanced": {"output_dir": str(results / "Advanced_Analysis")},
        "groups": {"Group1": {"data_dir": "data/raw/Group1/Group_1_DATA",
                              "output_dir": str(results / "Group1_Results")}},
    }))
    return cfg, results


class TestResultsDirIsForwarded:
    """verify-01: a run pointed at another results tree must stay inside it."""

    def test_haldane_step_gets_the_configured_results_dir(self, tmp_path):
        cfg, results = _scratch_config(tmp_path)
        cmd = _command_for(_dry_run(["--steps", "4", "--config", str(cfg)]),
                           "05_haldane_analysis.py")
        assert f"--results-dir {results}" in cmd, cmd

    def test_advanced_step_gets_the_configured_results_dir(self, tmp_path):
        cfg, results = _scratch_config(tmp_path)
        cmd = _command_for(_dry_run(["--steps", "5", "--config", str(cfg)]),
                           "06_advanced_fitting.py")
        assert f"--results-dir {results}" in cmd, cmd

    def test_comparison_step_gets_the_configured_paths(self, tmp_path):
        cfg, results = _scratch_config(tmp_path)
        cmd = _command_for(_dry_run(["--steps", "7", "--config", str(cfg)]),
                           "07_compare_truncation_methods.py")
        assert f"--results-csv {results / 'all_groups_results.csv'}" in cmd, cmd
        assert ("--ensemble-csv "
                f"{results / 'Advanced_Analysis' / 'ensemble_truncation' / 'ensemble_truncation_results.csv'}"
                in cmd), cmd
        assert f"--output-dir {results / 'Advanced_Analysis' / 'truncation_comparison'}" in cmd, cmd

    @pytest.mark.parametrize("step,script,flag", [
        ("4", "05_haldane_analysis.py", "--results-dir"),
        ("5", "06_advanced_fitting.py", "--results-dir"),
        ("7", "07_compare_truncation_methods.py", "--results-csv"),
    ])
    def test_no_step_writes_outside_the_configured_tree(self, tmp_path, step, script, flag):
        """Nothing in the forwarded command may still point at results/tables."""
        cfg, results = _scratch_config(tmp_path)
        cmd = _command_for(_dry_run(["--steps", step, "--config", str(cfg)]), script)
        assert flag in cmd, cmd
        assert str(ROOT / "results" / "tables") not in cmd, cmd

    def test_repo_config_still_resolves_to_results_tables(self):
        """The shipped config must keep writing exactly where it always did."""
        out = _dry_run(["--steps", "4,5,7", "--no-ml"])
        tables = ROOT / "results" / "tables"
        assert f"--results-dir {tables}" in _command_for(out, "05_haldane_analysis.py")
        assert f"--results-dir {tables}" in _command_for(out, "06_advanced_fitting.py")
        cmp_cmd = _command_for(out, "07_compare_truncation_methods.py")
        assert f"--results-csv {tables / 'all_groups_results.csv'}" in cmp_cmd
        assert (f"--output-dir {tables / 'Advanced_Analysis' / 'truncation_comparison'}"
                in cmp_cmd)


def _load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class TestConfigPathIsAccepted:
    """verify-02: --config must survive being a plain string."""

    def test_load_config_accepts_a_string_path(self, tmp_path):
        adv = _load_module(SCRIPTS / "06_advanced_fitting.py", "adv06_verify")
        cfg = tmp_path / "config.yaml"
        cfg.write_text(yaml.safe_dump({"classification": {"min_r_squared": 0.91}}))
        assert adv.load_config(str(cfg))["classification"]["min_r_squared"] == 0.91
        assert adv.load_config(cfg)["classification"]["min_r_squared"] == 0.91

    def test_load_config_of_a_missing_or_empty_file_is_a_dict(self, tmp_path):
        adv = _load_module(SCRIPTS / "06_advanced_fitting.py", "adv06_verify")
        assert adv.load_config(str(tmp_path / "nope.yaml")) == {}
        empty = tmp_path / "empty.yaml"
        empty.write_text("")
        assert adv.load_config(str(empty)) == {}

    def test_comparison_script_reads_its_config_file(self, tmp_path):
        """07 --config must get as far as looking for its inputs, not crash on argparse's str."""
        cfg, results = _scratch_config(tmp_path)
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "07_compare_truncation_methods.py"),
             "--config", str(cfg),
             "--ensemble-csv", str(tmp_path / "missing_ensemble.csv"),
             "--results-csv", str(tmp_path / "missing_results.csv"),
             "--output-dir", str(tmp_path / "out")],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
        assert "AttributeError" not in proc.stderr, proc.stderr[-2000:]
        assert "Traceback" not in proc.stderr, proc.stderr[-2000:]
        assert proc.returncode == 1
        assert "Ensemble results not found" in proc.stdout, proc.stdout[-2000:]
