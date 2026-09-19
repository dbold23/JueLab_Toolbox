"""
The lab manual ("How it works") as data.

The manual lives in ``docs/how_it_works/NN_<slug>.md``. Each section is
Markdown with ``{placeholders}`` for the analysis thresholds, so the app can
show the text with the *current* settings filled in, and this module can
render the same text with the defaults into ``docs/HOW_IT_WORKS.md`` for
readers on GitHub.

    python -m tecan_growth.manual          # rewrite docs/HOW_IT_WORKS.md
    python -m tecan_growth.manual --check  # exit 1 if the file is stale
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from .settings import AnalysisSettings

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
SECTIONS_DIR = DOCS_DIR / "how_it_works"
RENDERED_PATH = DOCS_DIR / "HOW_IT_WORKS.md"

_HEADER = """# How the analysis works

This is the lab manual for the TECAN growth-curve pipeline, rendered with
the **default** thresholds. The same text appears in the app under *How it
works*, where the numbers follow whatever settings are currently applied.
Edit the sections in `docs/how_it_works/` and run
`python -m tecan_growth.manual` to refresh this file.

"""

#: Placeholder used when a value is unknown (e.g. counts with no run loaded).
MISSING = "…"


def section_files() -> List[Path]:
    """The section files in reading order."""
    if not SECTIONS_DIR.exists():
        return []
    return sorted(p for p in SECTIONS_DIR.glob("*.md") if re.match(r"^\d+_", p.stem))


def section_title(path: Path) -> str:
    """'03_growth_window' -> 'growth window'."""
    return re.sub(r"^\d+_", "", path.stem).replace("_", " ")


def placeholder_values(settings: Optional[AnalysisSettings] = None,
                       counts: Optional[Mapping[str, int]] = None,
                       run_name: Optional[str] = None) -> Dict[str, object]:
    """
    The dictionary the sections are formatted with: every field of
    ``AnalysisSettings`` (floats printed compactly), ``n_GOOD``, ``n_BAD``,
    ``n_not_fitted`` and ``run_name``. Unknown keys render as :data:`MISSING`
    rather than raising, so a typo in a section never blanks the page.
    """
    settings = settings or AnalysisSettings.interactive_defaults()
    values: Dict[str, object] = defaultdict(lambda: MISSING)
    for k, v in settings.to_dict().items():
        if k == "extra":
            continue
        values[k] = f"{v:g}" if isinstance(v, float) else v
    for k, v in (counts or {}).items():
        values[f"n_{str(k).replace(' ', '_')}"] = v
    values["run_name"] = run_name or "(no run loaded)"
    return values


def placeholders_in(text: str) -> List[str]:
    """Names of the ``{placeholders}`` a section uses (LaTeX braces are doubled and ignored)."""
    return sorted(set(re.findall(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})", text)))


def render_section(path: Path, values: Mapping[str, object]) -> str:
    """One section with its placeholders filled in."""
    return path.read_text(encoding="utf-8").format_map(values)


def render_manual(settings: Optional[AnalysisSettings] = None, header: str = _HEADER) -> str:
    """The whole manual as one Markdown document."""
    values = placeholder_values(settings)
    parts = [header.rstrip() + "\n\n## Contents\n\n"]
    files = section_files()
    for i, p in enumerate(files, 1):
        parts.append(f"{i}. [{section_title(p).capitalize()}](#{_anchor(p)})\n")
    parts.append("\n")
    for p in files:
        body = render_section(p, values).strip() + "\n"
        # Promote the section's own '## ' heading to a numbered one for the single file.
        parts.append(f"\n---\n\n{body}\n")
    return "".join(parts)


def _anchor(path: Path) -> str:
    first = path.read_text(encoding="utf-8").lstrip().splitlines()[0]
    heading = re.sub(r"^#+\s*", "", first)
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def write_rendered(path: Path = RENDERED_PATH) -> Path:
    path.write_text(render_manual(), encoding="utf-8", newline="\n")
    return path


def is_current(path: Path = RENDERED_PATH) -> bool:
    return path.exists() and path.read_text(encoding="utf-8") == render_manual()


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        if is_current():
            print(f"{RENDERED_PATH.relative_to(REPO_ROOT)} is up to date")
            return 0
        print(f"{RENDERED_PATH.relative_to(REPO_ROOT)} is stale; run: python -m tecan_growth.manual")
        return 1
    out = write_rendered()
    print(f"wrote {out.relative_to(REPO_ROOT)} ({len(section_files())} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
