"""
Smoke test of the lab interface with Streamlit's AppTest (no browser needed).

Loads the example plate, runs the analysis, visits every page, changes a
threshold, refits a curve with a manual window and builds the export bundle.
Any uncaught exception on any page fails the test.
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

os.environ.setdefault("MPLBACKEND", "Agg")


def _errors(at):
    return [str(e.value)[:300] for e in at.exception]


def _button(at, label, sidebar=False):
    src = at.sidebar if sidebar else at
    found = [b for b in src.button if b.label == label]
    assert found, f"button '{label}' not found (have {[b.label for b in src.button]})"
    return found[0]


@pytest.fixture(scope="module")
def app():
    cwd = os.getcwd()
    os.chdir(APP)
    if str(APP) not in sys.path:
        sys.path.insert(0, str(APP))
    try:
        at = AppTest.from_file(str(APP / "Home.py"), default_timeout=300)
        at.run()
        assert not _errors(at)
        _button(at, "Try the example plate").click()
        at.run()
        assert not _errors(at)
        _button(at, "Run analysis").click()
        at.run()
        assert not _errors(at)
        yield at
    finally:
        os.chdir(cwd)


def test_example_plate_runs(app):
    res = app.session_state["results"]
    assert res.n_total == 24
    assert 0 < res.n_good < 24


@pytest.mark.parametrize("page", ["views/plate_qc.py", "views/curves.py", "views/compare.py",
                                  "views/export.py", "views/how_it_works.py"])
def test_every_page_renders(app, page):
    app.switch_page(page)
    app.run()
    assert not _errors(app), _errors(app)


def test_threshold_change_reclassifies_instantly(app):
    app.switch_page("views/curves.py")
    app.run()
    before = app.session_state["results"].n_good
    slider = [w for w in app.sidebar.slider if w.label == "Minimum R²"][0]
    slider.set_value(0.80).run()
    assert not _errors(app)
    assert app.session_state["results"].n_good >= before
    assert not app.sidebar.warning, "a post-fit threshold change must not leave the results stale"
    slider.set_value(0.95).run()
    assert app.session_state["results"].n_good == before


def test_truncation_change_marks_stale_and_rerun_clears_it(app):
    radio = [w for w in app.sidebar.radio if w.label.startswith("Growth window")][0]
    radio.set_value("Global maximum").run()
    assert app.sidebar.warning, "changing the window method must flag stale results"
    _button(app, "Re-run analysis", sidebar=True).click()
    app.run()
    assert not _errors(app)
    assert not app.sidebar.warning
    radio.set_value("First peak (fast, default)").run()
    _button(app, "Re-run analysis", sidebar=True).click()
    app.run()


def test_manual_window_refit_and_verdict_override(app):
    app.switch_page("views/curves.py")
    app.run()
    _button(app, "Open").click()
    app.run()
    cid = app.session_state["selected_curve"]
    assert cid in app.session_state["results"].curves
    window = [w for w in app.slider if w.label.startswith("Time window")][0]
    lo, hi = window.value
    window.set_value((lo, max(lo + 10.0, (lo + hi) / 2))).run()
    _button(app, "Refit with this window").click()
    app.run()
    assert not _errors(app)
    assert app.session_state["overrides"][cid]["truncation"]
    _button(app, "Back to automatic").click()
    app.run()
    assert not app.session_state["overrides"][cid].get("truncation")
    [w for w in app.radio if w.label == "Call of record"][0].set_value("GOOD").run()
    [w for w in app.text_input if w.label.startswith("Why?")][0].set_value("checked by eye").run()
    _button(app, "Save verdict").click()
    app.run()
    assert app.session_state["overrides"][cid]["verdict"] == "GOOD"


def test_export_bundle_builds(app):
    app.switch_page("views/export.py")
    app.run()
    _button(app, "Build the full bundle (zip)").click()
    app.run()
    assert not _errors(app)
    assert len(app.session_state["bundle"]) > 10_000
