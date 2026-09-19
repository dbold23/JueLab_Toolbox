#!/usr/bin/env python3
"""
Derive the lab prior from the real plates (WP5).

Why
---
The synthetic suite's amplitude distribution is inverted relative to the lab it
is supposed to stand in for: measured over the 92 real conditions in
``data/raw/Group1..4``, 30.4 % of conditions are dead wells (rise < 0.02 OD)
and 44.6 % are full growers (> 1.2 OD), while the committed 480-curve benchmark
had 4.0 % dead and 46 % in the 0.6-1.2 OD band that is 2 % of the lab.  A
headline accuracy computed on that composition is a macro average over an
arbitrary set of scenarios, not an estimate of anything the lab will see.

This script measures the composition instead of asserting it, and writes it to
``synthetic_data/config/lab_prior.json``.  ``curve_scenarios.ScenarioConfig``
references a band by name (``lab_band``) and
``generate_comprehensive.py --weighting lab`` re-allocates the curve budget to
the measured fractions; nothing duplicates the numbers as literals, so the
prior and the catalog cannot drift apart.

Definitions
-----------
* A *condition* is one (media, strain) curve of one plate, blank-subtracted and
  replicate-averaged by ``tecan_growth.preprocess.preprocess_plate`` - exactly
  the object the analysis pipeline reads.
* *Total rise* is ``max(od) - mean(od[:5])``, which is the pipeline's own
  ``delta_od`` (01_growth_curve_analysis.py:289): using the pipeline's
  definition means the bands line up with the gate the pipeline applies.
* The band edges (0.02, 0.1, 0.3, 0.6, 1.2 OD) are the audit's, chosen around
  the 0.15 OD growth gate and the 1.2 OD "full culture" plateau.

Usage
-----
    python synthetic_data/scripts/derive_lab_prior.py            # rewrite the file
    python synthetic_data/scripts/derive_lab_prior.py --check    # verify it
    python synthetic_data/scripts/derive_lab_prior.py --data-dir DIR -o OUT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tecan_growth.io import load_plate            # noqa: E402
from tecan_growth.preprocess import preprocess_plate  # noqa: E402

#: The raw values/key pair of every real plate, relative to ``data/raw``.
#: These are the four plates the whole project is built on; a group whose files
#: are missing is reported, never silently skipped.
GROUP_FILES: Dict[str, Tuple[str, str]] = {
    'Group1': ('Group1/GrowthRate_ Group1_Values.csv', 'Group1/GROUP1_KEY_V2.csv'),
    'Group2': ('Group 2/GrowthRate_GROUP2_DATA.csv', 'Group 2/GrowthRate_GROUP2_KEY.csv'),
    'Group3': ('Group 3/GROUP3_VALUES.csv', 'Group 3/GROUP3_KEY.csv'),
    'Group4': ('Group 4/G4_VALUES.csv', 'Group 4/G4_KEY.csv'),
}

#: Band edges in OD (upper edge exclusive); the six bands are
#: <0.02, 0.02-0.1, 0.1-0.3, 0.3-0.6, 0.6-1.2, >=1.2.
BAND_EDGES = (0.02, 0.1, 0.3, 0.6, 1.2)

#: Band names, in the same order.  ``curve_scenarios.ScenarioConfig.lab_band``
#: refers to these.
BAND_NAMES = ('dead', 'trace', 'marginal', 'partial', 'moderate', 'full')

BAND_DESCRIPTIONS = (
    'no measurable rise (< 0.02 OD) - dead well or empty control',
    'trace rise (0.02-0.1 OD) - below the pipeline growth gate',
    'marginal rise (0.1-0.3 OD) - straddles the 0.15 OD gate',
    'partial growth (0.3-0.6 OD)',
    'moderate growth (0.6-1.2 OD)',
    'full culture (>= 1.2 OD)',
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / 'config' / 'lab_prior.json'


def condition_rises(data_dir: Path) -> List[Dict[str, Any]]:
    """Total rise of every real condition, as the pipeline defines it."""
    rows: List[Dict[str, Any]] = []
    for group, (values_rel, key_rel) in sorted(GROUP_FILES.items()):
        values = data_dir / values_rel
        key = data_dir / key_rel
        if not values.exists() or not key.exists():
            raise FileNotFoundError(
                f"{group}: expected {values} and {key}; the lab prior is a "
                f"measurement of the real plates and cannot be derived without them")
        plate = preprocess_plate(load_plate(values, key))
        for curve_id in plate.curve_ids:
            _t, od = plate.curve(curve_id)
            od = np.asarray(od, dtype=float)
            rows.append({
                'group': group,
                'condition': str(curve_id),
                # The pipeline's own delta_od (01_growth_curve_analysis.py:289).
                'delta_od': float(np.nanmax(od) - np.nanmean(od[:5])),
            })
    return rows


def build_prior(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Band counts and fractions over the measured conditions."""
    values = np.array([r['delta_od'] for r in rows], dtype=float)
    index = np.digitize(values, BAND_EDGES)
    n = int(values.size)

    bands = []
    for i, name in enumerate(BAND_NAMES):
        count = int(np.sum(index == i))
        low = 0.0 if i == 0 else float(BAND_EDGES[i - 1])
        high = float(BAND_EDGES[i]) if i < len(BAND_EDGES) else None
        bands.append({
            'name': name,
            'description': BAND_DESCRIPTIONS[i],
            'delta_od_min': low,
            'delta_od_max': high,
            'n': count,
            'fraction': round(count / n, 6) if n else 0.0,
        })

    per_group: Dict[str, int] = {}
    for r in rows:
        per_group[r['group']] = per_group.get(r['group'], 0) + 1

    return {
        'schema': 'lab_prior/1',
        'source': 'data/raw/Group1..4 via tecan_growth.load_plate + preprocess_plate',
        'statistic': 'delta_od = max(od) - mean(od[:5]), the pipeline definition '
                     '(01_growth_curve_analysis.py:289), on the blank-subtracted, '
                     'replicate-averaged condition',
        'band_edges_od': list(BAND_EDGES),
        'n_conditions': int(values.size),
        'conditions_per_group': dict(sorted(per_group.items())),
        'delta_od_median': round(float(np.median(values)), 6),
        'delta_od_mean': round(float(np.mean(values)), 6),
        'bands': bands,
        'derived_by': 'synthetic_data/scripts/derive_lab_prior.py',
        'note': 'Derived, never hand-edited: re-run the script to regenerate. '
                'Scenario weights reference a band by name '
                '(curve_scenarios.ScenarioConfig.lab_band).',
    }


def render(prior: Dict[str, Any]) -> str:
    """The exact bytes of config/lab_prior.json (sorted, 2-space, trailing \\n)."""
    return json.dumps(prior, indent=2, sort_keys=True) + '\n'


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', default=str(_REPO_ROOT / 'data' / 'raw'),
                        help='Directory holding Group1..Group 4 (default: data/raw)')
    parser.add_argument('-o', '--output', default=str(DEFAULT_OUTPUT),
                        help='Where to write the prior (default: config/lab_prior.json)')
    parser.add_argument('--check', action='store_true',
                        help='Do not write; exit non-zero if the file on disk differs')
    args = parser.parse_args(argv)

    rows = condition_rises(Path(args.data_dir))
    prior = build_prior(rows)
    text = render(prior)
    out = Path(args.output)

    if args.check:
        if not out.exists():
            print(f"MISSING: {out}")
            return 1
        current = out.read_text()
        if current != text:
            print(f"STALE: {out} differs from what data/raw produces")
            return 1
        print(f"OK: {out} matches data/raw ({prior['n_conditions']} conditions)")
        return 0

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"Wrote {out} from {prior['n_conditions']} real conditions")
    for band in prior['bands']:
        print(f"  {band['name']:9s} {band['n']:3d}  {100 * band['fraction']:5.1f} %  "
              f"{band['description']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
