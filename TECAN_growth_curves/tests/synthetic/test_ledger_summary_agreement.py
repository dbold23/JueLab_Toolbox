"""The ledger and the generated summary must state the same numbers.

Finding covered
---------------
`docs/BENCHMARK_LEDGER.md` scored the held-out `a_not_identifiable` block at
68/80 (0.850) while the rendered `VALIDATION_SUMMARY.md` printed 12/80 (15.0 %)
for the same eighty curves, and the ledger row pointed at the report's
`truncation` block as the source of *its* number.  A reader comparing the two
documents saw two different scores for one block with no way to tell which
question each answered.

There are genuinely two rules:

* the **flat** rule -- the headline's own accept/reject-versus-truth rule,
  which the block passes 68 times out of 80;
* the **block** rule recorded in `report.json` at
  `truncation.a_not_identifiable.rule`, which additionally asks whether the
  reported A is supportable from a truncated run, and which passes 12 times.

Both documents must now carry both numbers, and both must agree with
`report.json` and with `per_curve.csv`.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

RENDERER = SYNTH / 'scripts' / 'validation_summary.py'
BENCH_V2 = SYNTH / 'benchmark' / 'v2'
LEDGER = REPO_ROOT / 'docs' / 'BENCHMARK_LEDGER.md'

ENV = {'MPLBACKEND': 'Agg', 'PATH': '/usr/bin:/bin:/usr/local/bin',
       'HOME': str(Path.home())}

BLOCK = 'a_not_identifiable'


@pytest.fixture(scope='module')
def report():
    path = BENCH_V2 / 'report.json'
    if not path.exists():
        pytest.skip('no published benchmark report in this checkout')
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture(scope='module')
def flat_counts():
    """Flat-rule counts for the block, straight from per_curve.csv."""
    import pandas as pd

    path = BENCH_V2 / 'per_curve.csv'
    if not path.exists():
        pytest.skip('no per_curve.csv in this checkout')
    frame = pd.read_csv(path)
    sub = frame[frame['scoring_mode'].astype(str) == BLOCK]
    ok = (sub['gt_is_good'].astype(bool) == sub['pred_is_good'].astype(bool))
    return {'n': int(len(sub)), 'n_correct': int(ok.sum()),
            'accuracy': float(ok.mean())}


@pytest.fixture(scope='module')
def rendered(tmp_path_factory):
    if not (BENCH_V2 / 'report.json').exists():
        pytest.skip('no published benchmark report in this checkout')
    out = tmp_path_factory.mktemp('summary') / 'VALIDATION_SUMMARY.md'
    proc = subprocess.run(
        [sys.executable, str(RENDERER), '--run-dir', str(BENCH_V2),
         '--output', str(out)],
        cwd=str(REPO_ROOT), env=ENV, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return out.read_text(encoding='utf-8')


def _ledger_row():
    if not LEDGER.exists():
        pytest.skip('no benchmark ledger in this tree')
    rows = [line for line in LEDGER.read_text(encoding='utf-8').splitlines()
            if line.startswith(f'| `{BLOCK}` |')]
    assert len(rows) == 1, f'expected one held-out ledger row, found {len(rows)}'
    return rows[0]


def test_the_renderer_prints_both_rules_for_the_block(rendered, report,
                                                      flat_counts):
    """The summary states the flat number and the block number, both labelled."""
    block_stats = report.get('truncation', {}).get(BLOCK) or {}
    if not block_stats:
        pytest.skip('this run has no a_not_identifiable block')

    section = rendered.split('## Blocks held out of the headline', 1)[1]
    section = section.split('## Matching', 1)[0]

    flat_row = [line for line in section.splitlines()
                if line.startswith('| flat GOOD/BAD')]
    block_row = [line for line in section.splitlines()
                 if line.startswith('| block rule')]
    assert flat_row and block_row, section

    assert str(flat_counts['n_correct']) in flat_row[0]
    assert f"{flat_counts['accuracy'] * 100:.1f}%" in flat_row[0]
    assert str(block_stats['n_correct']) in block_row[0]
    assert f"{block_stats['pass_rate'] * 100:.1f}%" in block_row[0]

    # The rule the block number comes from is quoted, not merely referenced.
    assert block_stats['rule'] in section


def test_the_ledger_row_carries_both_numbers_with_their_rules(report,
                                                              flat_counts):
    block_stats = report.get('truncation', {}).get(BLOCK) or {}
    if not block_stats:
        pytest.skip('this run has no a_not_identifiable block')

    row = _ledger_row()
    cells = [c.strip() for c in row.strip('|').split('|')]
    # block | n | flat correct | flat accuracy | block correct | block rate | why
    assert len(cells) >= 6, cells
    assert cells[1] == str(flat_counts['n'])
    assert cells[2] == str(flat_counts['n_correct'])
    assert float(cells[3]) == pytest.approx(flat_counts['accuracy'], abs=5e-4)
    assert cells[4] == str(block_stats['n_correct'])
    assert float(cells[5]) == pytest.approx(block_stats['pass_rate'], abs=5e-4)

    header = LEDGER.read_text(encoding='utf-8')
    section = header.split('### Nothing is hidden by the scoring policy', 1)[1]
    section = section.split('## 2.', 1)[0]
    for phrase in ('flat rule', 'block rule'):
        assert phrase in section, phrase


def test_neither_document_states_one_of_the_two_numbers_alone(rendered):
    """Whichever document a reader opens, both numbers are in front of them."""
    ledger_section = LEDGER.read_text(encoding='utf-8').split(
        '### Nothing is hidden by the scoring policy', 1)[1].split('## 2.', 1)[0]
    summary_section = rendered.split(
        '## Blocks held out of the headline', 1)[1].split('## Matching', 1)[0]

    assert re.search(r'\b0\.850\b', ledger_section)
    assert re.search(r'\b0\.150\b', ledger_section)
    assert '85.0%' in summary_section
    assert '15.0%' in summary_section
