"""
The lab manual must render cleanly: every placeholder resolves against the
settings, the LaTeX survives formatting, the images it references exist,
the rendered docs/HOW_IT_WORKS.md is current, and the README points at
things that exist.
"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tecan_growth.manual import (MISSING, RENDERED_PATH, SECTIONS_DIR, is_current, placeholder_values,
                                 placeholders_in, render_manual, render_section, section_files, section_title)
from tecan_growth.settings import AnalysisSettings

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"


def test_sections_exist_in_order():
    files = section_files()
    assert len(files) >= 11
    numbers = [int(p.stem.split("_")[0]) for p in files]
    assert numbers == sorted(numbers) and len(set(numbers)) == len(numbers)
    for p in files:
        assert p.read_text(encoding="utf-8").lstrip().startswith("## "), f"{p.name} must start with a '## ' heading"


@pytest.mark.parametrize("path", section_files(), ids=lambda p: p.stem)
def test_section_renders_with_defaults(path):
    values = placeholder_values(AnalysisSettings.from_config(), counts={"GOOD": 1, "BAD": 2, "not fitted": 3}, run_name="x")
    text = path.read_text(encoding="utf-8")
    known = set(AnalysisSettings.from_config().to_dict()) | {"n_GOOD", "n_BAD", "n_not_fitted", "run_name"}
    unknown = [k for k in placeholders_in(text) if k not in known]
    assert not unknown, f"{path.name} uses unknown placeholders {unknown}"
    out = render_section(path, values)
    assert MISSING not in out
    # Every brace in the source is either a placeholder or a doubled LaTeX brace.
    stripped = re.sub(r"\{[A-Za-z_][A-Za-z0-9_]*\}", "", text.replace("{{", "").replace("}}", ""))
    assert "{" not in stripped and "}" not in stripped, f"{path.name} has an undoubled brace"
    for m in re.finditer(r"!\[[^\]]*\]\(images/([^)]+)\)", out):
        assert (ROOT / "docs" / "images" / m.group(1)).exists(), f"{path.name} references a missing image {m.group(1)}"


def test_thresholds_follow_the_settings():
    text = (SECTIONS_DIR / "07_the_verdict.md").read_text(encoding="utf-8")
    loose = placeholder_values(AnalysisSettings.from_config(min_r_squared=0.9, max_param_error_pct=33.0))
    out = text.format_map(loose)
    assert "≥ 0.9" in out and "33 %" in out
    assert "0.95" not in out.split("### After fitting")[1].split("The last two")[0]


def test_rendered_manual_is_current():
    assert RENDERED_PATH.exists(), "run: python -m tecan_growth.manual"
    assert is_current(), "docs/HOW_IT_WORKS.md is stale; run: python -m tecan_growth.manual"
    rendered = render_manual()
    for p in section_files():
        assert section_title(p) in rendered.lower() or section_title(p).capitalize() in rendered


def test_manual_cli_check_passes():
    proc = subprocess.run([sys.executable, "-m", "tecan_growth.manual", "--check"], cwd=ROOT,
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_readme_links_and_targets_exist():
    text = README.read_text(encoding="utf-8")
    for rel in re.findall(r"\]\(((?:docs|scripts|app|data|tests|results|synthetic_data)/[^)#]+)\)", text):
        assert (ROOT / rel).exists(), f"README links to a missing path: {rel}"
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", makefile, flags=re.M))
    for target in re.findall(r"`make ([a-zA-Z0-9_-]+)`", text):
        assert target in targets, f"README mentions `make {target}` which is not a Makefile target"
    for cmd in re.findall(r"python3? (scripts/[A-Za-z0-9_./-]+\.py)", text):
        assert (ROOT / cmd).exists(), f"README runs a missing script {cmd}"


def test_lab_protocol_exists_and_names_the_pages():
    text = (ROOT / "docs" / "LAB_PROTOCOL.md").read_text(encoding="utf-8")
    for page in ("Home", "Plate QC", "Curves", "Compare", "Export"):
        assert page in text
