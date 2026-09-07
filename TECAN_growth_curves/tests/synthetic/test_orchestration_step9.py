"""
Orchestration of the synthetic benchmark (WP10).

The two ways this repository could publish a number nobody chose:

* step 9 ran whatever ``truncation.use_adaptive`` said and overwrote a single
  ``validation_latest/`` in place, so an adaptive run silently replaced the
  first-peak headline the docs quote (M14). Step 9 now writes one directory per
  truncation mode, defaults to the headline mode, and names the mode on every
  line it prints.
* step 9 appended ``--ml-classify`` whenever the config flag happened to be on.
  The 480 synthetic curves are the classifier's entire training set, so that
  turned an out-of-sample rule benchmark into an in-sample self-evaluation
  worth about ten accuracy points, with nothing recording it (M26). The ML pass
  is now opt-in, separately located and announced.

Also pinned here: the ``make clean`` glob really matches the directory step 9
writes, the run directories are git-ignored, the suite size is read from
MANIFEST.json rather than typed, and the dead ``validate_pipeline.py`` entry
point is gone.
"""
import glob
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SYNTH = ROOT / "synthetic_data"
ENV = dict(os.environ, MPLBACKEND="Agg")


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pipeline():
    return _load(SCRIPTS / "run_full_pipeline.py", "run_full_pipeline_wp10")


def _dry_run(extra_args, expect_ok=True):
    proc = subprocess.run(
        [sys.executable, str(SCRIPTS / "run_full_pipeline.py"), "--dry-run"] + extra_args,
        cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
    if expect_ok:
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]
    return proc.stdout


def _fake_synth_root(tmp_path, n_curves=6, n_scenarios=3):
    """A synthetic suite root shaped like the committed one, with no real data.

    Step 9 skips itself when ``test_data/DATA`` and ``ground_truth.csv`` are
    missing, so the dry-run tests need the shape even though nothing is read.
    """
    root = tmp_path / "synth"
    (root / "test_data" / "DATA").mkdir(parents=True)
    rows = ["curve_id,scenario,expected_class"]
    for i in range(n_curves):
        rows.append(f"CURVE{i + 1:04d},scen{i % n_scenarios},GOOD")
    (root / "test_data" / "ground_truth.csv").write_text("\n".join(rows) + "\n")
    (root / "generation_summary.json").write_text(json.dumps({
        "total_curves": n_curves,
        "scenarios": {f"scen{i}": n_curves // n_scenarios for i in range(n_scenarios)},
        "seed": 42,
    }))
    return root


def _scratch_config(tmp_path, **overrides):
    """A config.yaml pointed entirely at ``tmp_path``."""
    cfg = {
        "truncation": {"use_adaptive": True, "use_first_peak": True},
        "ml_classifier": {"enabled": False},
        "paths": {"results": str(tmp_path / "results")},
        "groups": {"Group1": {"data_dir": "data/raw/Group1/Group_1_DATA",
                              "output_dir": str(tmp_path / "results" / "Group1_Results")}},
    }
    for key, value in overrides.items():
        cfg[key] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


# =============================================================================
# M14 -- one directory per truncation mode, headline mode by default
# =============================================================================

class TestTruncationModeSelection:
    def test_config_adaptive_does_not_reach_step_9(self, pipeline):
        """`make step-9` must not inherit the batch default for the real plates."""
        cfg = {"truncation": {"use_adaptive": True}}
        assert pipeline.resolve_step9_modes(cfg) == ["first_peak"]
        assert pipeline.resolve_step9_modes(cfg, adaptive=None) == ["first_peak"]
        # ...while step 2 still follows it.
        assert pipeline.resolve_use_adaptive(cfg) is True

    def test_explicit_cli_flags_pick_the_mode(self, pipeline):
        cfg = {"truncation": {"use_adaptive": True}}
        assert pipeline.resolve_step9_modes(cfg, adaptive=True) == ["adaptive"]
        assert pipeline.resolve_step9_modes(cfg, adaptive=False) == ["first_peak"]

    def test_all_modes_runs_both(self, pipeline):
        modes = pipeline.resolve_step9_modes({}, all_modes=True)
        assert modes == list(pipeline.STEP9_TRUNCATION_MODES)
        assert modes[0] == pipeline.HEADLINE_TRUNCATION_MODE

    def test_each_mode_has_its_own_directory(self, pipeline, tmp_path):
        dirs = {m: pipeline.step9_run_dir(m, synthetic_out_dir=str(tmp_path))
                for m in pipeline.STEP9_TRUNCATION_MODES}
        assert len(set(dirs.values())) == len(dirs), "modes share a directory"
        for mode, path in dirs.items():
            assert path.name == mode
            assert path.parent.name == "validation"

    def test_ml_pass_directories_never_collide(self, pipeline, tmp_path):
        single = pipeline.step9_ml_run_dir("first_peak", 1, synthetic_out_dir=str(tmp_path))
        assert single.name == "validation_latest_ml"
        both = {m: pipeline.step9_ml_run_dir(m, 2, synthetic_out_dir=str(tmp_path))
                for m in pipeline.STEP9_TRUNCATION_MODES}
        assert len(set(both.values())) == 2


class TestStep9DryRunLayout:
    def test_default_run_writes_only_the_first_peak_directory(self, tmp_path):
        """The acceptance case: `make step-9` with use_adaptive true in config."""
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path)
        out = _dry_run(["--steps", "9", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert str(tmp_path / "synth" / "validation" / "first_peak") in out
        assert str(tmp_path / "synth" / "validation" / "adaptive") not in out
        assert "--no-adaptive" in out and "--first-peak" in out
        assert not (tmp_path / "synth" / "validation" / "adaptive").exists()

    def test_all_modes_names_both_directories(self, tmp_path):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path)
        out = _dry_run(["--steps", "9", "--all-modes", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        for mode in ("first_peak", "adaptive"):
            assert str(tmp_path / "synth" / "validation" / mode) in out

    def test_every_result_line_names_its_mode(self, tmp_path):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path)
        out = _dry_run(["--steps", "9", "--all-modes", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert "STEP 9a [first_peak]" in out
        assert "STEP 9a [adaptive]" in out
        assert "Score the first_peak run" in out
        assert "Score the adaptive run" in out

    def test_the_benchmark_is_called_not_the_dead_validator(self, tmp_path):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path)
        out = _dry_run(["--steps", "9", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert "run_benchmark(" in out
        assert "pipeline_validator" not in out
        assert "--skip-run" not in out


# =============================================================================
# M26 -- the in-sample ML pass is opt-in, separate and announced
# =============================================================================

class TestInSampleMlPass:
    def test_ml_enabled_without_the_flag_stays_rule_based(self, tmp_path, monkeypatch):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path, ml_classifier={"enabled": True})
        out = _dry_run(["--steps", "9", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert "--ml-classify" not in out
        assert "in-sample ML pass skipped" in out
        assert "validation_latest_ml" not in out

    def test_the_skip_line_is_printed_exactly_once(self, tmp_path):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path)
        out = _dry_run(["--steps", "9", "--all-modes", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert out.count("in-sample ML pass skipped") == 1

    @pytest.mark.skipif(not (ROOT / "models" / "postfit_classifier.joblib").exists(),
                        reason="trained models are not present")
    def test_the_flag_adds_a_labelled_in_sample_pass(self, tmp_path):
        _fake_synth_root(tmp_path)
        cfg = _scratch_config(tmp_path, ml_classifier={"enabled": True})
        out = _dry_run(["--steps", "9", "--allow-in-sample-ml", "--config", str(cfg),
                        "--synthetic-out-dir", str(tmp_path / "synth")])
        assert ("step 9b: ML pass is IN-SAMPLE (models were trained on these curves)"
                in out)
        assert str(tmp_path / "synth" / "validation_latest_ml") in out
        ml_cmds = [line for line in out.splitlines()
                   if "01_growth_curve_analysis.py" in line and "--ml-classify" in line]
        assert len(ml_cmds) == 1, out
        assert "validation_latest_ml" in ml_cmds[0]
        # The rule-based headline directory is still written, untouched by ML.
        rule_cmds = [line for line in out.splitlines()
                     if "01_growth_curve_analysis.py" in line and "--ml-classify" not in line]
        assert any(str(tmp_path / "synth" / "validation" / "first_peak") in line
                   for line in rule_cmds)

    def test_the_note_says_why_when_ml_is_simply_off(self, pipeline):
        note = pipeline._in_sample_ml_note({"ml_classifier": {"enabled": False}})
        assert "step 9b:" in note and "\n" not in note


# =============================================================================
# The suite size is read, not typed
# =============================================================================

class TestSuiteSizeComesFromTheManifest:
    def test_manifest_wins(self, pipeline, tmp_path):
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 42,
            "n_per_scenario": {f"s{i}": 20 for i in range(7)},
        }))
        (tmp_path / "generation_summary.json").write_text(json.dumps({
            "total_curves": 480, "scenarios": {"old": 480}}))
        n_curves, n_scenarios, source = pipeline.synthetic_suite_counts(tmp_path)
        assert (n_curves, n_scenarios, source) == (140, 7, "MANIFEST.json")
        assert "140 synthetic curves (7 scenarios)" == pipeline.describe_synthetic_suite(tmp_path)

    def test_generation_summary_is_the_fallback(self, pipeline, tmp_path):
        (tmp_path / "generation_summary.json").write_text(json.dumps({
            "total_curves": 480, "scenarios": {f"s{i}": 15 for i in range(32)}}))
        assert pipeline.synthetic_suite_counts(tmp_path) == (480, 32, "generation_summary.json")

    def test_ground_truth_is_the_last_resort(self, pipeline, tmp_path):
        data = tmp_path / "test_data"
        data.mkdir()
        (data / "ground_truth.csv").write_text(
            "curve_id,scenario\nCURVE0001,a\nCURVE0002,a\nCURVE0003,b\n")
        assert pipeline.synthetic_suite_counts(tmp_path) == (3, 2, "ground_truth.csv")

    def test_no_hand_typed_curve_count_survives(self):
        source = (SCRIPTS / "run_full_pipeline.py").read_text()
        assert "480 synthetic curves" not in source
        assert "480 curves" not in source

    def test_the_committed_suite_is_described_from_disk(self, pipeline):
        text = pipeline.describe_synthetic_suite()
        assert re.match(r"^\d+ synthetic curves \(\d+ scenarios\)$", text), text


# =============================================================================
# Makefile and .gitignore
# =============================================================================

class TestMakefile:
    @pytest.fixture(scope="class")
    def makefile(self):
        return (ROOT / "Makefile").read_text()

    def test_new_targets_exist(self, makefile):
        targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", makefile, flags=re.M))
        for target in ("synthetic-data", "synthetic-verify", "validation-summary",
                       "step-9", "validate-synthetic", "clean"):
            assert target in targets, f"missing target {target}"

    def test_synthetic_data_warns_that_it_invalidates_published_numbers(self, makefile):
        head = makefile.split("PYTHON =")[0]
        assert "INVALIDATES" in head
        assert "models/" in head

    @pytest.fixture(scope="class")
    def clean_recipe(self):
        proc = subprocess.run(["make", "-n", "clean"], cwd=str(ROOT),
                              capture_output=True, text=True, timeout=120)
        assert proc.returncode == 0, proc.stderr
        return proc.stdout

    def test_clean_glob_matches_the_directory_step_9_writes(self, clean_recipe):
        """M25: the old glob was validation_v*/ and matched nothing."""
        patterns = re.findall(r"(\S*comprehensive_test/validation\S*)", clean_recipe)
        assert patterns, clean_recipe
        assert not any("validation_v*" in g for g in patterns)
        expected = ROOT / "synthetic_data" / "output" / "comprehensive_test"
        existing = [d for d in expected.glob("validation*") if d.is_dir()]
        if existing:
            matched = set()
            for pattern in patterns:
                matched.update(glob.glob(str(ROOT / pattern)))
            assert matched, f"none of {patterns} matched any of {existing}"

    def test_clean_deletes_with_git_clean_not_rm_rf(self, clean_recipe):
        """
        The run directories are not purely generated: validation_latest/ holds
        the tracked benchmark snapshot as well as ~96 MB of plots, so `rm -rf`
        on the glob deletes tracked files while the target prints that it keeps
        them. git clean removes untracked and ignored files only.
        """
        assert "git clean" in clean_recipe
        removed = re.findall(r"rm -rf (\S+)", clean_recipe)
        assert all("validation" not in path for path in removed), removed

    def test_clean_never_lists_a_tracked_file_for_deletion(self):
        """The protection itself, measured on the real tree."""
        base = "synthetic_data/output/comprehensive_test"
        tracked = subprocess.run(
            ["git", "ls-files", "--", f"{base}/validation", f"{base}/validation_latest"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120).stdout.split()
        if not tracked:
            pytest.skip("no tracked file inside a step-9 run directory")
        doomed = subprocess.run(
            ["git", "clean", "-ndx", "--", f"{base}/validation", f"{base}/validation_latest"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=120).stdout
        for path in tracked:
            assert path not in doomed, f"make clean would delete tracked {path}"

    def test_clean_removes_generated_files(self, tmp_path):
        """
        Run the real target against a probe tree inside the repo: git clean
        needs a path inside the work tree, so tmp_path cannot be used.
        """
        probe = ROOT / "synthetic_data" / "output" / "_wp10_clean_probe"
        try:
            (probe / "validation" / "first_peak").mkdir(parents=True)
            (probe / "validation_latest" / "plots").mkdir(parents=True)
            (probe / "validation" / "first_peak" / "report.json").write_text("{}")
            (probe / "validation_latest" / "plots" / "c0.png").write_text("x")
            proc = subprocess.run(
                ["make", "clean", f"SYNTH_TEST=synthetic_data/output/_wp10_clean_probe"],
                cwd=str(ROOT), capture_output=True, text=True, timeout=300)
            assert proc.returncode == 0, proc.stdout + proc.stderr
            assert not (probe / "validation").exists(), proc.stdout
            assert not (probe / "validation_latest").exists(), proc.stdout
        finally:
            shutil.rmtree(probe, ignore_errors=True)

    def test_the_clean_message_names_a_path_that_exists(self, makefile):
        """
        The old target closed with 'the tracked benchmark snapshot under
        synthetic_data/benchmark/ are preserved' -- a directory that is not in
        the tree. A message naming a place the reader cannot find is not a
        promise they can check.
        """
        recipe = makefile.split("\nclean:")[1]
        messages = "\n".join(line for line in recipe.splitlines()
                             if line.lstrip().startswith(("@echo", "echo")))
        for path in re.findall(r"\$\(SYNTH[A-Z_]*\)/[\w./*-]+", messages):
            resolved = (path.replace("$(SYNTH_TEST)",
                                     "synthetic_data/output/comprehensive_test")
                            .replace("$(SYNTH)", "synthetic_data"))
            if "*" in resolved:
                continue
            assert glob.glob(str(ROOT / resolved)), f"clean names a missing path: {resolved}"


class TestValidationSummaryTarget:
    """
    `make validation-summary` invoked validation_summary.py --run-dir, an
    argument that script does not have yet, so the target aborted in argparse.
    It now uses --run-dir when the renderer offers it and maps RUN_DIR onto the
    flags it does have otherwise.
    """

    @pytest.fixture(scope="class")
    def recipe(self):
        makefile = (ROOT / "Makefile").read_text()
        return makefile.split("\nvalidation-summary:")[1].split("\n\n")[0]

    def test_it_detects_whether_run_dir_is_supported(self, recipe):
        assert "--help" in recipe and "--run-dir" in recipe

    def test_it_has_a_fallback_that_uses_the_flags_that_exist(self, recipe):
        supported = subprocess.run(
            [sys.executable, str(SYNTH / "scripts" / "validation_summary.py"), "--help"],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=120).stdout
        for flag in re.findall(r"--[a-z-]+", recipe):
            if flag in ("--help", "--run-dir"):
                continue
            assert flag in supported, f"{flag} is not a validation_summary.py flag"

    def test_the_target_runs(self):
        proc = subprocess.run(["make", "validation-summary"], cwd=str(ROOT), env=ENV,
                              capture_output=True, text=True, timeout=600)
        if "No renderable run directory" in proc.stdout:
            pytest.skip("no run directory in this tree to render")
        assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-2000:]


class TestGitignore:
    @pytest.fixture(scope="class")
    def gitignore(self):
        return (ROOT / ".gitignore").read_text()

    def test_run_directories_are_ignored(self, gitignore):
        assert "synthetic_data/output/comprehensive_test/validation/" in gitignore
        assert "synthetic_data/output/comprehensive_test/validation_*/" in gitignore

    def test_the_validation_source_package_is_not_ignored(self):
        proc = subprocess.run(
            ["git", "check-ignore", "-q", "synthetic_data/validation/benchmark.py"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 1, "the validator source must stay tracked"

    def test_a_step_9_run_directory_is_ignored(self):
        proc = subprocess.run(
            ["git", "check-ignore", "-q",
             "synthetic_data/output/comprehensive_test/validation/first_peak/report.json"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        assert proc.returncode == 0


# =============================================================================
# The dead entry point is gone
# =============================================================================

class TestValidatePipelineIsDeleted:
    def test_the_script_no_longer_exists(self):
        assert not (SYNTH / "scripts" / "validate_pipeline.py").exists()

    def test_nothing_wp10_owns_references_it(self):
        for path in (SCRIPTS / "run_full_pipeline.py", ROOT / "Makefile",
                     SCRIPTS / "config.yaml", ROOT / ".gitignore",
                     SCRIPTS / "verify_synthetic_data.py"):
            assert "validate_pipeline" not in path.read_text(), path

    def test_no_document_still_tells_a_reader_to_run_it(self):
        """
        Acceptance item 5: every surviving mention is historical prose.

        A mention is a defect when a reader can act on it -- a shell line that
        invokes the deleted script, or a project-tree listing that still shows
        the file as if it were there. Sentences that say it is gone are the
        point of the change and stay.
        """
        tracked = subprocess.run(
            ["git", "ls-files", "*.md", "*.rst", "*.txt", "*.sh", "Makefile"],
            cwd=str(ROOT), capture_output=True, text=True, timeout=60)
        assert tracked.returncode == 0, tracked.stderr
        runnable = re.compile(r"(?:^|[`$\s(])(?:python[\w.]*|bash|sh|make)\s[^\n]*"
                              r"validate_pipeline")
        tree_entry = re.compile(r"[\u2502\u251c\u2514\u2500]\s*\S*validate_pipeline")
        offenders = []
        for rel in tracked.stdout.split():
            path = ROOT / rel
            if not path.exists():
                continue
            for lineno, line in enumerate(path.read_text().splitlines(), 1):
                if "validate_pipeline" not in line:
                    continue
                if runnable.search(line) or tree_entry.search(line):
                    offenders.append(f"{rel}:{lineno}: {line.strip()}")
        assert not offenders, (
            "live references to the deleted script:\n" + "\n".join(offenders))


# =============================================================================
# make synthetic-verify
# =============================================================================

class TestVerifyScript:
    @pytest.fixture(scope="class")
    def verify(self):
        return _load(SCRIPTS / "verify_synthetic_data.py", "verify_synthetic_data_wp10")

    # ── The generation command is replayed, never guessed ──────────────────
    #
    # The first version of this check collapsed MANIFEST.json's per-scenario
    # map to its SMALLEST count and passed "--n-per-scenario <min>", which
    # flattened every scenario to that count. `make synthetic-data` passes no
    # --n-per-scenario at all, so a snapshot made by the shipped command
    # regenerated 350 rows against 1139 committed ones and the check reported
    # a composition change of its own making: it could not pass on any suite
    # with per-scenario counts, whatever WP12 regenerated.

    def test_the_recorded_generation_command_is_replayed(self, verify, tmp_path):
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 42,
            "n_per_scenario": {"a": 5, "b": 50},
            "generation_command":
                "generate_comprehensive.py -o output/comprehensive_test --seed 42",
        }))
        plan = verify.generator_settings(tmp_path)
        assert plan.seed == 42
        assert plan.args == []
        assert "generation_command" in plan.source
        command = plan.command("/tmp/out")
        assert "--n-per-scenario" not in command
        assert command[-4:] == ["--seed", "42", "-o", "/tmp/out"]

    def test_replay_keeps_the_flags_it_cannot_reconstruct(self, verify, tmp_path):
        """--weighting/--total-curves/--lab-prior are not derivable from counts."""
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 3, "weighting": "lab",
            "n_per_scenario": {"a": 7, "b": 9},
            "generation_command": ("generate_comprehensive.py --weighting lab "
                                   "--total-curves 900 --lab-prior config/lab_prior.json "
                                   "-o output/lab --seed 3 -v"),
        }))
        plan = verify.generator_settings(tmp_path)
        assert plan.args == ["--weighting", "lab", "--total-curves", "900",
                             "--lab-prior", "config/lab_prior.json"]
        assert "output/lab" not in " ".join(plan.command("/tmp/out"))

    def test_per_scenario_counts_never_collapse_to_one_number(self, verify, tmp_path):
        """A recorded map with 5..50 curves must not become --n-per-scenario 5."""
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 42, "n_per_scenario": {"a": 5, "b": 12, "c": 50}}))
        plan = verify.generator_settings(tmp_path)
        assert "--n-per-scenario" not in plan.args, plan.args
        assert any("non-uniform" in note for note in plan.notes), plan.notes
        assert "5..50" in plan.composition

    def test_the_catalogue_counts_replay_as_no_override(self, verify, tmp_path):
        catalog = verify.catalog_counts()
        assert catalog, "this tree's scenario catalogue must be importable"
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 42, "n_per_scenario": catalog}))
        plan = verify.generator_settings(tmp_path)
        assert plan.args == []
        assert any("catalogue defaults" in note for note in plan.notes), plan.notes

    def test_a_uniform_override_replays_as_one_number(self, verify, tmp_path):
        (tmp_path / "MANIFEST.json").write_text(json.dumps({
            "seed": 7, "n_per_scenario": {"a": 20, "b": 20}}))
        plan = verify.generator_settings(tmp_path)
        assert (plan.seed, plan.args, plan.source) == (
            7, ["--n-per-scenario", "20"], "MANIFEST.json")

    def test_settings_fall_back_to_the_generation_summary(self, verify, tmp_path):
        (tmp_path / "generation_summary.json").write_text(json.dumps({
            "seed": 42, "scenarios": {"a": 15, "b": 15}}))
        plan = verify.generator_settings(tmp_path)
        assert (plan.seed, plan.args, plan.source) == (
            42, ["--n-per-scenario", "15"], "generation_summary.json")

    def test_the_default_matches_what_make_synthetic_data_runs(self, verify, tmp_path):
        """No manifest at all: the documented defaults, not an invented count."""
        plan = verify.generator_settings(tmp_path)
        assert (plan.seed, plan.args) == (42, [])
        recipe = re.search(r"^synthetic-data:.*?(?=\n\S|\Z)",
                           (ROOT / "Makefile").read_text(), re.S | re.M).group(0)
        assert "generate_comprehensive.py" in recipe
        assert "--n-per-scenario" not in recipe, (
            "the verifier replays `make synthetic-data`; if that target grows a "
            "uniform count the replay has to grow it too")

    def test_a_generated_snapshot_verifies_end_to_end(self, tmp_path):
        """
        Generate a suite the way the Makefile does, then check it verifies.

        Small (--n-per-scenario 2) only to keep the file inside its time
        budget; the composition path this pins is the same one the shipped
        1139-curve suite takes, since both are read back out of MANIFEST.json's
        generation_command.
        """
        snapshot = tmp_path / "snapshot"
        gen = subprocess.run(
            [sys.executable, str(SYNTH / "scripts" / "generate_comprehensive.py"),
             "-o", str(snapshot), "--seed", "11", "--n-per-scenario", "2"],
            cwd=str(SYNTH), env=ENV, capture_output=True, text=True, timeout=600)
        assert gen.returncode == 0, gen.stdout[-3000:] + gen.stderr[-2000:]

        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "verify_synthetic_data.py"),
             "--committed", str(snapshot), "--tmp-dir", str(tmp_path / "regen")],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=900)
        assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-2000:]
        assert "PASSED" in proc.stdout
        assert "COMPOSITION CHANGED" not in proc.stdout

    def _write_gt(self, path, rows):
        header = "curve_id,scenario,expected_class,true_A,true_mu,true_lambda\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + "".join(",".join(str(v) for v in r) + "\n" for r in rows))

    def test_identical_tables_pass(self, verify, tmp_path):
        rows = [("CURVE0001", "standard", "GOOD", 1.2, 0.3, 2.0),
                ("CURVE0002", "high_A", "GOOD", 2.4, 0.4, 1.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", rows)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", rows)
        results = verify.compare_ground_truth(
            tmp_path / "a" / "test_data" / "ground_truth.csv",
            tmp_path / "b" / "test_data" / "ground_truth.csv")
        assert all(r.ok for r in results), [r.render() for r in results if not r.ok]

    def test_a_drifted_parameter_fails_and_names_the_scenario(self, verify, tmp_path):
        old = [("CURVE0001", "standard", "GOOD", 1.2, 0.3, 2.0),
               ("CURVE0002", "borderline_delta_od", "BAD", 0.32, 0.4, 1.0)]
        new = [("CURVE0001", "standard", "GOOD", 1.2, 0.3, 2.0),
               ("CURVE0002", "borderline_delta_od", "BAD", 0.17, 0.4, 1.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new)
        results = verify.compare_ground_truth(
            tmp_path / "a" / "test_data" / "ground_truth.csv",
            tmp_path / "b" / "test_data" / "ground_truth.csv")
        failed = [r for r in results if not r.ok]
        assert failed, "a 0.15 OD shift in true_A must not pass"
        text = "\n".join(r.render() for r in failed)
        assert "borderline_delta_od" in text
        assert "true_A" in text

    def test_a_new_scenario_fails_and_names_it(self, verify, tmp_path):
        old = [("CURVE0001", "standard", "GOOD", 1.2, 0.3, 2.0)]
        new = old + [("CURVE0002", "haldane_strong_inhibition", "BAD", 0.4, 0.1, 5.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new)
        results = verify.compare_ground_truth(
            tmp_path / "a" / "test_data" / "ground_truth.csv",
            tmp_path / "b" / "test_data" / "ground_truth.csv")
        text = "\n".join(r.render() for r in results if not r.ok)
        assert "haldane_strong_inhibition" in text

    def test_a_flipped_label_fails_and_names_the_scenario(self, verify, tmp_path):
        old = [("CURVE0001", "borderline_noise", "BAD", 1.2, 0.3, 2.0)]
        new = [("CURVE0001", "borderline_noise", "GOOD", 1.2, 0.3, 2.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new)
        results = verify.compare_ground_truth(
            tmp_path / "a" / "test_data" / "ground_truth.csv",
            tmp_path / "b" / "test_data" / "ground_truth.csv")
        text = "\n".join(r.render() for r in results if not r.ok)
        assert "borderline_noise" in text and "expected_class" in text

    def test_growing_the_catalogue_does_not_invent_drift(self, verify, tmp_path):
        """
        curve_id is a positional index. Inserting a scenario shifts every later
        id onto different data, and a comparison keyed on curve_id then reports
        drift in scenarios that never moved -- the committed suite produced
        "true_A: flat_no_growth -- max |delta| 1.49" for a scenario whose A is
        identically 0. Keyed on content, the unchanged scenarios stay silent.
        """
        old = [(0, "flat_no_growth", "BAD", 0.0, 0.0, 0.0),
               (1, "flat_no_growth", "BAD", 0.0, 0.0, 0.0),
               (2, "standard", "GOOD", 1.2, 0.3, 2.0)]
        # A new scenario lands first and pushes both of the others along.
        new_rows = [(0, "all_zero", "BAD", 0.0, 0.0, 0.0),
                    (1, "flat_no_growth", "BAD", 0.0, 0.0, 0.0),
                    (2, "flat_no_growth", "BAD", 0.0, 0.0, 0.0),
                    (3, "standard", "GOOD", 1.2, 0.3, 2.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new_rows)
        results = verify.compare_ground_truth(
            tmp_path / "a" / "test_data" / "ground_truth.csv",
            tmp_path / "b" / "test_data" / "ground_truth.csv")
        by_name = {r.name: r for r in results}

        composition = [name for name in verify.COMPOSITION_CHECKS
                       if name in by_name and not by_name[name].ok]
        assert composition, "adding a scenario is a composition change"
        assert "all_zero" in by_name["scenario set"].render()

        values = [r for name, r in by_name.items()
                  if name not in verify.COMPOSITION_CHECKS]
        text = "\n".join(r.render() for r in values)
        assert all(r.ok for r in values), text
        assert "flat_no_growth" not in text, text

    def test_the_content_key_prefers_the_per_curve_seed(self, verify):
        pd = pytest.importorskip("pandas")
        old = pd.DataFrame({"scenario": ["s", "s", "t"], "seed": [11, 12, 13]})
        # Same curves, emitted in a different order.
        new = pd.DataFrame({"scenario": ["t", "s", "s"], "seed": [13, 12, 11]})
        keys_old, keys_new, description = verify.content_keys(old, new)
        assert "seed" in description
        assert set(keys_old) == set(keys_new)

    def test_the_content_key_falls_back_to_position(self, verify):
        pd = pytest.importorskip("pandas")
        frame = pd.DataFrame({"scenario": ["s", "s", "t"]})
        _keys_old, _keys_new, description = verify.content_keys(frame, frame)
        assert "position" in description

    def test_a_composition_change_is_not_called_drift(self, verify, tmp_path):
        """The closing verdict has to say which of the two happened."""
        old = [(0, "standard", "GOOD", 1.2, 0.3, 2.0)]
        new_rows = old + [(1, "sudden_crash", "GOOD", 1.5, 0.4, 3.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new_rows)
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "verify_synthetic_data.py"),
             "--committed", str(tmp_path / "a"),
             "--tmp-dir", str(tmp_path / "b"), "--skip-regenerate"],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 1, proc.stdout
        assert "COMPOSITION CHANGED" in proc.stdout
        assert "DRIFT:" not in proc.stdout

    def test_a_value_drift_is_called_drift(self, verify, tmp_path):
        old = [(0, "standard", "GOOD", 1.2, 0.3, 2.0)]
        new_rows = [(0, "standard", "GOOD", 1.9, 0.3, 2.0)]
        self._write_gt(tmp_path / "a" / "test_data" / "ground_truth.csv", old)
        self._write_gt(tmp_path / "b" / "test_data" / "ground_truth.csv", new_rows)
        proc = subprocess.run(
            [sys.executable, str(SCRIPTS / "verify_synthetic_data.py"),
             "--committed", str(tmp_path / "a"),
             "--tmp-dir", str(tmp_path / "b"), "--skip-regenerate"],
            cwd=str(ROOT), env=ENV, capture_output=True, text=True, timeout=300)
        assert proc.returncode == 1, proc.stdout
        assert "DRIFT:" in proc.stdout
        assert "standard" in proc.stdout

    def test_a_digest_difference_is_reported_but_not_fatal(self, verify, tmp_path):
        for name, payload in (("a", "x"), ("b", "y")):
            data = tmp_path / name / "test_data"
            (data / "DATA").mkdir(parents=True)
            (data / "ground_truth.csv").write_text(payload)
            (data / "DATA" / "P_DATA.csv").write_text(payload)
        result = verify.compare_digests(tmp_path / "a", tmp_path / "b")
        assert result.fatal is False
        assert result.ok is False
        assert "numpy/BLAS" in "\n".join(result.lines)
