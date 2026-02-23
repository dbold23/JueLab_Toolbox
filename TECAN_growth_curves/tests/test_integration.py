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

    def test_plots_exist(self, results_dir):
        """At least some diagnostic plots should exist."""
        plot_count = 0
        for g in ['Group1', 'Group2', 'Group3', 'Group4']:
            plots_dir = results_dir / f"{g}_Results" / "plots"
            if plots_dir.exists():
                plot_count += len(list(plots_dir.glob("*.png")))
        assert plot_count > 0, "No diagnostic plots found"


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
