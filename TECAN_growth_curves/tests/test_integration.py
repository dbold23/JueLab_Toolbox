"""
Integration tests: verify that pipeline outputs exist and are consistent.
"""
import pytest
from pathlib import Path
import pandas as pd


@pytest.fixture
def results_dir(repo_root):
    return repo_root / "results" / "tables"


class TestPipelineOutputs:
    """Verify that pipeline results files exist and are well-formed."""

    def test_all_groups_results_exists(self, results_dir):
        """all_groups_results.csv should exist."""
        path = results_dir / "all_groups_results.csv"
        assert path.exists(), f"Missing: {path}"

    def test_all_groups_results_columns(self, results_dir):
        """Results CSV should have required columns."""
        df = pd.read_csv(results_dir / "all_groups_results.csv")
        required_cols = ['strain', 'is_good', 'group', 'gompertz_a', 'gompertz_mu', 'gompertz_lambda']
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_groups_results_has_data(self, results_dir):
        """Results should have >= 90 rows (we know there are 92 strains)."""
        df = pd.read_csv(results_dir / "all_groups_results.csv")
        assert len(df) >= 90

    def test_all_four_groups_present(self, results_dir):
        """All four groups should be represented."""
        df = pd.read_csv(results_dir / "all_groups_results.csv")
        groups = set(df['group'].unique())
        for g in ['Group1', 'Group2', 'Group3', 'Group4']:
            assert g in groups, f"Missing group: {g}"

    def test_group_result_dirs_exist(self, results_dir):
        """Each group should have a results directory."""
        for g in ['Group1', 'Group2', 'Group3', 'Group4']:
            gdir = results_dir / f"{g}_Results"
            assert gdir.exists(), f"Missing: {gdir}"

    def test_diagnostic_plots_are_generated(self, repo_root, tmp_path):
        """
        Per-strain diagnostic PNGs are regenerable output (gitignored), so run
        the core pipeline on the smallest bundled group into a temp dir and
        check that it writes one plot per strain plus the summary figure.
        """
        import importlib.util
        import os
        os.environ.setdefault("MPLBACKEND", "Agg")
        spec = importlib.util.spec_from_file_location(
            "gca_integration", repo_root / "scripts" / "01_growth_curve_analysis.py")
        gca = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gca)
        results = gca.process_growth_curves(
            str(repo_root / "data" / "raw" / "Group1" / "Group_1_DATA"), str(tmp_path), verbose=False)
        assert len(results) == 24
        pngs = list((tmp_path / "plots").glob("*.png"))
        # Curves rejected because no valid truncation window exists get no per-strain plot.
        expected = sum(1 for r in results.values() if r.stage != "truncation_invalid")
        assert len(pngs) == expected, f"expected {expected} diagnostic plots, found {len(pngs)}"
        assert expected >= 20
        assert (tmp_path / "classification_summary.png").exists()
        assert (tmp_path / "processing_results.csv").exists()


class TestHaldaneOutputs:
    """Verify Haldane analysis outputs."""

    def test_haldane_comparison_exists(self, results_dir):
        """haldane_comparison.csv should exist after running Haldane analysis."""
        path = results_dir / "Haldane_Analysis" / "haldane_comparison.csv"
        if not path.exists():
            pytest.skip("Haldane analysis not yet run")
        df = pd.read_csv(path)
        assert len(df) > 0
        assert 'haldane_Ki' in df.columns
        assert 'preferred_model' in df.columns

    def test_haldane_summary_exists(self, results_dir):
        """haldane_summary.csv should exist."""
        path = results_dir / "Haldane_Analysis" / "haldane_summary.csv"
        if not path.exists():
            pytest.skip("Haldane analysis not yet run")
        df = pd.read_csv(path)
        assert len(df) > 0
        assert 'pesticide' in df.columns

    def test_haldane_overview_plot_exists(self, results_dir):
        """haldane_overview.png should exist."""
        path = results_dir / "Haldane_Analysis" / "haldane_overview.png"
        if not path.exists():
            pytest.skip("Haldane analysis not yet run")
        assert path.stat().st_size > 10000  # non-trivial file


class TestValidationOutputs:
    """Verify validation audit outputs."""

    def test_validation_audit_exists(self, results_dir):
        """validation_audit.csv should exist after running the validator."""
        path = results_dir / "validation_audit.csv"
        if not path.exists():
            pytest.skip("Validation audit not yet run")
        df = pd.read_csv(path)
        assert len(df) > 0
        assert 'audit_result' in df.columns

    def test_validation_audit_coverage(self, results_dir):
        """Audit should cover all (or nearly all) strains."""
        audit_path = results_dir / "validation_audit.csv"
        results_path = results_dir / "all_groups_results.csv"
        if not audit_path.exists():
            pytest.skip("Validation audit not yet run")

        audit = pd.read_csv(audit_path)
        results = pd.read_csv(results_path)
        coverage = len(audit) / len(results)
        assert coverage >= 0.9, f"Only {coverage*100:.0f}% of strains audited"
