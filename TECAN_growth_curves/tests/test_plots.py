"""
Smoke and encoding tests for tecan_growth.plots and tecan_growth.media.

Every figure must build from the bundled Group 3 plate, carry hover text on
its marks, use the documented palette, and never draw dashed gridlines or a
second y-axis scale.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import plotly.graph_objects as go

from tecan_growth import load_plate, preprocess_plate, media, plots
from tecan_growth.analyze import analyze_plate
from tecan_growth.settings import AnalysisSettings

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}


@pytest.fixture(scope="module")
def plate():
    run = load_plate(RAW / "Group 3" / "GrowthRate_GROUP3.xlsx")
    processed = preprocess_plate(run)
    results = analyze_plate(processed, AnalysisSettings.from_config(use_adaptive=False))
    return run, processed, results


def _all_figures(run, processed, results):
    table = media.annotate(results.table)
    bad = next(cid for cid, c in results.curves.items() if not c.is_good and c.legacy.stage == "classified")
    return {
        "plate_layout": plots.plate_grid(run.key, None, run, mode="layout"),
        "plate_verdict": plots.plate_grid(run.key, results, run, mode="verdict"),
        "plate_max_od": plots.plate_grid(run.key, results, run, mode="max_od"),
        "delta_od_heatmap": plots.delta_od_heatmap(run, run.key),
        "replicate_facets": plots.replicate_facets(processed, "LB"),
        "temperature": plots.temperature_trace(run),
        "strip_good": plots.pipeline_strip(processed, results.curve("LB-IMID2")),
        "strip_bad": plots.pipeline_strip(processed, results.curve(bad)),
        "residuals": plots.residual_plot(results.curve("LB-IMID2")),
        "sparkline": plots.curve_sparkline(results.curve("LB-IMID2")),
        "dotplot": plots.parameter_dotplot(table, "gompertz_mu"),
        "dotplot_bad": plots.parameter_dotplot(table, "gompertz_a", include_bad=True),
        "ratio": plots.ratio_strip(table),
        "overlay": plots.strain_overlay(processed, results, "IMID2"),
    }


class TestFiguresBuild:
    def test_every_figure_builds_and_has_hover(self, plate):
        figs = _all_figures(*plate)
        for name, fig in figs.items():
            assert isinstance(fig, go.Figure), name
            assert len(fig.data) > 0, name
            hover = [t for t in fig.data if getattr(t, "hovertemplate", None) or getattr(t, "hoverinfo", "") not in ("", "skip", "none")]
            assert hover, f"{name}: no trace carries hover text"

    def test_no_dashed_gridlines_or_dual_axes(self, plate):
        figs = _all_figures(*plate)
        for name, fig in figs.items():
            layout = fig.layout
            for key in layout:
                if key.startswith(("xaxis", "yaxis")):
                    ax = layout[key]
                    assert getattr(ax, "griddash", None) in (None, "solid"), f"{name}: dashed grid on {key}"
                    assert getattr(ax, "overlaying", None) is None, f"{name}: overlaid (dual) axis {key}"

    def test_plate_grid_has_96_wells_with_customdata(self, plate):
        run, processed, results = plate
        fig = plots.plate_grid(run.key, results, run, mode="verdict")
        wells = set()
        for trace in fig.data:
            if trace.customdata is not None:
                for cd in trace.customdata:
                    wells.add(cd[0] if isinstance(cd, (list, tuple, np.ndarray)) else cd)
        assert len(wells) == 96
        assert {"A1", "H12"} <= wells

    def test_verdict_colours_use_status_palette(self, plate):
        run, processed, results = plate
        fig = plots.plate_grid(run.key, results, run, mode="verdict")
        used = set()
        for trace in fig.data:
            colour = trace.marker.color if trace.marker is not None else None
            if colour is None:
                continue
            for c in (colour if isinstance(colour, (list, tuple, np.ndarray)) else [colour]):
                if isinstance(c, str) and c.startswith("#"):
                    used.add(c.lower())
        assert STATUS["good"] in used and STATUS["critical"] in used

    def test_gate_table_matches_curve_gates(self, plate):
        run, processed, results = plate
        cr = results.curve("LB-IMID2")
        table = plots.gate_table(cr)
        assert {"gate", "description", "value", "threshold", "comparison", "status"} <= set(table.columns)
        assert len(table) == len(cr.gates)
        assert (table.status.isin(["PASS", "FAIL", "BYPASSED", "NOT APPLIED"])).all()

    def test_pipeline_strip_marks_incomplete_curves(self, plate):
        run, processed, results = plate
        inc = [c for c in results.curves.values() if c.legacy.is_incomplete and c.legacy.stage == "classified"]
        if not inc:
            pytest.skip("no incomplete curve on this plate")
        fig = plots.pipeline_strip(processed, inc[0])
        texts = " ".join(str(a.text) for a in fig.layout.annotations)
        assert "INCOMPLETE" in texts.upper()

    def test_figures_build_fast(self, plate):
        import time
        t0 = time.perf_counter()
        _all_figures(*plate)
        assert time.perf_counter() - t0 < 10.0


class TestMedia:
    @pytest.mark.parametrize("name,ttype,pest", [
        ("LB", "LB", None), ("H2O", "H2O", None),
        ("MALATHION", "Pesticide", "Malathion"), ("MALATHIONANDLB", "Pesticide+LB", "Malathion"),
        ("LambdaCyhalothrinANDLB", "Pesticide+LB", "Lambda-Cyhalothrin"),
        ("LAMBDACYHALOTHRIN", "Pesticide", "Lambda-Cyhalothrin"),
        ("FlupyradifuroneANDLB", "Pesticide+LB", "Flupyradifurone"),
        ("BifenthrinANDLB", "Pesticide+LB", "Bifenthrin"), ("Bifenthrin", "Pesticide", "Bifenthrin"),
        ("IMIDACLOPRIDANDLB", "Pesticide+LB", "Imidacloprid"), ("PERMETHRIN", "Pesticide", "Permethrin"),
        ("Malathion+LB", "Pesticide+LB", "Malathion"), ("LB + Malathion", "Pesticide+LB", "Malathion"),
        ("lb", "LB", None), ("water", "H2O", None),
    ])
    def test_classify_media(self, name, ttype, pest):
        out = media.classify_media(name)
        assert out["treatment_type"] == ttype
        assert out["pesticide"] == pest

    def test_annotate_adds_columns(self):
        df = pd.DataFrame({"media": ["LB", "MALATHIONANDLB", "H2O", "MALATHION"], "x": [1, 2, 3, 4]})
        out = media.annotate(df)
        assert list(out.treatment_type) == ["LB", "Pesticide+LB", "H2O", "Pesticide"]
        assert list(out.has_nutrients) == [True, True, False, False]

    def test_strain_colours_are_stable_and_fold_past_eight(self):
        strains = [f"S{i}" for i in range(10)]
        cmap = media.strain_color_map(strains)
        assert cmap["S0"] == "#2a78d6" and cmap["S1"] == "#eb6834"
        assert cmap["S8"] == cmap["S9"] == "#898781"
        assert media.strain_color_map(["S1", "S0"])["S1"] == "#2a78d6"  # first-seen order
        assert media.strain_color_map(strains[:3])["S2"] == cmap["S2"]  # unchanged by later strains
