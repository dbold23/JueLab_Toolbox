#!/usr/bin/env python3
"""
Verify that the committed synthetic benchmark still regenerates from HEAD.
=========================================================================

``make synthetic-verify`` runs this. It regenerates the suite into a temporary
directory with the SAME generation command the committed data was made with,
and compares the two ground-truth tables.

  The command is not guessed. MANIFEST.json records ``generation_command``
  verbatim, so the replay is that command with ``-o`` pointed at the temporary
  directory; when the manifest is older and records only the settings, the
  command is rebuilt from them, and a per-scenario map that matches this tree's
  catalogue replays as no ``--n-per-scenario`` at all -- which is what ``make
  synthetic-data`` runs, and what makes each scenario keep its own ``n_default``
  (5..50 curves).  Only a uniform override replays as ``--n-per-scenario N``.
  Guessing one number here is what previously made this check impossible to
  pass: the recorded per-scenario map was collapsed to its smallest count, which
  flattened every scenario to that count, so a snapshot produced by the exact
  ``make synthetic-data`` command regenerated 350 rows against 1139 committed
  ones and the check reported a composition change of its own making.

The two tables are then compared:

  Rows are matched on a CONTENT KEY, not on curve_id.  curve_id in this suite
  is a positional index (0, 1, 2, ...) and strain_name is CURVE%04d of the same
  index, so as soon as the catalogue gains or loses a scenario every later
  curve_id points at different data and a value comparison keyed on it reports
  drift in scenarios that never moved.  The key is (scenario, per-curve seed)
  when the table carries a seed column, and (scenario, position within the
  scenario) otherwise; the one in use is printed with the result.

  HARD checks (any failure exits non-zero, and every message names the
  scenarios responsible):
    * the scenario set is identical
    * the per-scenario row counts, and so the total row count, are identical
    * the curve_id set is identical, and every curve_id still names a curve of
      the same scenario (a positional index that has been reshuffled is a
      composition change, reported as one)
    * expected_class is identical for every content key
    * true_A / true_mu / true_lambda agree to 1e-9 for every content key

  SOFT check (reported, never fatal):
    * sha256 of ground_truth.csv and of every DATA/*.csv. Byte equality is not
      a promise the project makes: numpy/BLAS versions already move the last
      bits of these floats, so a digest mismatch with the numeric check green
      is reported as "regenerated under a different numpy/BLAS", not as a
      failure wall.

The closing verdict distinguishes the two ways this can fail.  COMPOSITION
CHANGED means the committed snapshot came from a different scenario catalogue
(scenarios added or removed, or per-scenario counts changed): the snapshot is
out of date and has to be regenerated and re-published.  DRIFT means the
catalogue matches curve for curve but the numbers moved, which is a regression
in the generator unless it was intended.

``--self-check`` adds a second regeneration and compares the two: determinism
is a property of the generator alone, so it is the one question this script can
still answer while a snapshot regeneration is pending.

Usage:
    python scripts/verify_synthetic_data.py
    python scripts/verify_synthetic_data.py --self-check
    python scripts/verify_synthetic_data.py --committed synthetic_data/output/other
    python scripts/verify_synthetic_data.py --keep      # leave the regenerated tree

Exit codes: 0 all hard checks passed, 1 a hard check failed, 2 the regeneration
itself could not be run.
"""

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
SYNTH_DIR = PROJECT_DIR / 'synthetic_data'
DEFAULT_COMMITTED = SYNTH_DIR / 'output' / 'comprehensive_test'

#: Parameter columns compared numerically, and the tolerance they must meet.
PARAM_COLUMNS = ('true_A', 'true_mu', 'true_lambda')
PARAM_TOLERANCE = 1e-9

#: Failing any of these means the two suites are not the same catalogue, so the
#: value comparisons are comparing different curves and the honest verdict is
#: "regenerate", not "the numbers drifted".
#: Scenarios listed per parameter before the drift report is summarised. A
#: catalogue-wide move would otherwise print one line per scenario per column.
MAX_DRIFT_LINES = 10

COMPOSITION_CHECKS = (
    'scenario set',
    'row count (total and per scenario)',
    'curve_id set',
    'curve_id -> scenario map',
)

#: Fallback when neither MANIFEST.json nor generation_summary.json records the
#: generator settings: the documented default of generate_comprehensive.py.
#: There is no n_per_scenario fallback on purpose -- the generator's default is
#: "every scenario's own n_default", which is the empty argument list.
DEFAULT_SEED = 42


# ─── Helpers ─────────────────────────────────────────────────────────────────

def read_json(path):
    """Parse a small JSON file, or ``None`` when missing/unreadable."""
    try:
        with open(path) as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


class GenerationPlan:
    """
    The generator invocation that reproduces a committed suite.

    ``args`` are the arguments to pass to generate_comprehensive.py *besides*
    ``-o`` (the temporary output directory is the caller's business); ``seed``
    is carried separately so ``--seed`` on the command line can override it
    without re-parsing the list.  ``composition`` is the one-line description
    printed with the result, and ``notes`` carries anything the reader has to
    know about how faithful the replay is.
    """

    def __init__(self, seed, args, composition, source, notes=()):
        self.seed = int(seed)
        self.args = list(args)
        self.composition = composition
        self.source = source
        self.notes = list(notes)

    def command(self, out_dir):
        """The full argv, with ``--seed`` and ``-o`` applied last so they win."""
        return [sys.executable,
                str(SYNTH_DIR / 'scripts' / 'generate_comprehensive.py')] + \
            self.args + ['--seed', str(self.seed), '-o', str(out_dir)]

    def __repr__(self):  # pragma: no cover - debugging aid
        return (f"GenerationPlan(seed={self.seed}, args={self.args!r}, "
                f"composition={self.composition!r}, source={self.source!r})")


#: Arguments of generate_comprehensive.py that a replay must not inherit: the
#: output directory is ours, and --print-readme-tables would print and exit.
_DROPPED_REPLAY_FLAGS = {'-o': 1, '--output': 1, '--seed': 1,
                         '--print-readme-tables': 0, '-v': 0, '--verbose': 0}


def catalog_counts():
    """
    This tree's per-scenario counts with no override, or ``None``.

    That is what ``make synthetic-data`` generates (each scenario's own
    ``n_default``), so a recorded map equal to it replays as no
    ``--n-per-scenario`` flag at all.
    """
    try:
        if str(SYNTH_DIR) not in sys.path:
            sys.path.insert(0, str(SYNTH_DIR))
        from src.curve_scenarios import get_comprehensive_test_config
        return {str(k): int(v) for k, v in get_comprehensive_test_config().items()}
    except Exception:
        return None


def _replay_args(recorded_command):
    """
    ``generation_command`` from MANIFEST.json turned into replayable arguments.

    Everything is kept verbatim except the flags in ``_DROPPED_REPLAY_FLAGS``:
    the recorded command is the ground truth about how the snapshot was made,
    including ``--weighting lab``, ``--total-curves`` and ``--lab-prior``, none
    of which can be reconstructed from the counts alone.
    """
    try:
        tokens = shlex.split(str(recorded_command))
    except ValueError:
        return None
    if not tokens:
        return None
    if Path(tokens[0]).name.endswith('.py'):
        tokens = tokens[1:]
    args, index = [], 0
    while index < len(tokens):
        token = tokens[index]
        flag, _, inline = token.partition('=')
        if flag in _DROPPED_REPLAY_FLAGS:
            index += 1 + (0 if inline else _DROPPED_REPLAY_FLAGS[flag])
            continue
        args.append(token)
        index += 1
    return args


def _recorded_counts(data):
    """The per-scenario map a manifest or a generation summary recorded."""
    for key in ('n_per_scenario', 'scenarios'):
        value = data.get(key)
        if isinstance(value, dict) and value:
            try:
                return {str(k): int(v) for k, v in value.items()}
            except (TypeError, ValueError):
                return None
    return None


def generator_settings(committed_root):
    """
    The :class:`GenerationPlan` that reproduces the suite under ``committed_root``.

    MANIFEST.json first (WP5 writes it, and it records the generation command
    verbatim), generation_summary.json next, then the generator's own defaults.
    Guessing settings silently would make this check meaningless, so the source
    and the resulting composition are printed with the result.

    The composition is never collapsed to a single number.  ``make
    synthetic-data`` passes no ``--n-per-scenario`` at all, so each scenario
    keeps its own ``n_default`` (5..50 curves); replaying that as a uniform
    count -- which the first version of this function did, using the smallest
    recorded count -- regenerates a different suite and reports a composition
    change this check caused itself.
    """
    for name in ('MANIFEST.json', 'generation_summary.json'):
        data = read_json(Path(committed_root) / name)
        if not isinstance(data, dict):
            continue
        seed = int(data.get('seed', DEFAULT_SEED))
        counts = _recorded_counts(data)

        recorded = data.get('generation_command')
        if isinstance(recorded, str) and recorded.strip():
            args = _replay_args(recorded)
            if args is not None:
                composition = _describe_counts(counts)
                return GenerationPlan(
                    seed, args, composition, f"{name} (generation_command)",
                    notes=[' '.join(['replaying: generate_comprehensive.py']
                                    + args + ['--seed', str(seed), '-o', '<tmp>'])])

        # Older snapshots record the settings but not the command.
        args, notes = [], []
        weighting = str(data.get('weighting', 'uniform') or 'uniform')
        if weighting == 'lab':
            args += ['--weighting', 'lab']
            if counts:
                args += ['--total-curves', str(sum(counts.values()))]
            notes.append("the snapshot was lab-weighted; --min-per-scenario and "
                         "--lab-prior are not recorded, so the generator defaults "
                         "are used for them")
        elif counts:
            uniform = sorted({int(v) for v in counts.values()})
            catalog = catalog_counts()
            if catalog is not None and counts == catalog:
                notes.append("the recorded per-scenario counts are this tree's "
                             "catalogue defaults, so no --n-per-scenario is passed")
            elif len(uniform) == 1:
                args += ['--n-per-scenario', str(uniform[0])]
            else:
                notes.append(
                    "the recorded per-scenario counts are non-uniform and are not "
                    "this tree's catalogue defaults; --n-per-scenario cannot "
                    "express them, so the catalogue counts are regenerated and the "
                    "row-count check below reports the difference")
        return GenerationPlan(seed, args, _describe_counts(counts), name, notes)

    return GenerationPlan(DEFAULT_SEED, [], _describe_counts(None),
                          'generate_comprehensive.py defaults',
                          notes=["no MANIFEST.json and no generation_summary.json: "
                                 "assuming the documented defaults (seed 42, each "
                                 "scenario's own n_default)"])


def _describe_counts(counts):
    """One line naming the composition the committed snapshot recorded."""
    if not counts:
        return "not recorded; regenerating with each scenario's own n_default"
    values = sorted({int(v) for v in counts.values()})
    total = sum(counts.values())
    if len(values) == 1:
        return (f"{len(counts)} scenarios x {values[0]} curve(s) "
                f"= {total} requested (uniform)")
    return (f"{len(counts)} scenarios, {values[0]}..{values[-1]} curves each "
            f"= {total} requested (per-scenario n_default)")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def regenerate(out_dir, plan, verbose=True):
    """Run generate_comprehensive.py into ``out_dir``. Returns True on success."""
    cmd = plan.command(out_dir)
    if verbose:
        print(f"  Regenerating: {' '.join(cmd)}")
    env = dict(os.environ, MPLBACKEND='Agg')
    proc = subprocess.run(cmd, cwd=str(SYNTH_DIR), env=env,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print("  ERROR: the generator failed", file=sys.stderr)
        print(proc.stdout[-3000:], file=sys.stderr)
        print(proc.stderr[-3000:], file=sys.stderr)
        return False
    return True


# ─── Comparison ──────────────────────────────────────────────────────────────

class Result:
    """One named check with a pass/fail verdict and the lines explaining it."""

    def __init__(self, name, ok, lines=(), fatal=True):
        self.name = name
        self.ok = ok
        self.lines = list(lines)
        self.fatal = fatal

    def render(self):
        status = 'OK' if self.ok else ('FAIL' if self.fatal else 'DIFFERS')
        out = [f"  [{status:7s}] {self.name}"]
        out += [f"            {line}" for line in self.lines]
        return '\n'.join(out)


def content_keys(old, new):
    """
    ``(keys_old, keys_new, description)`` -- the identity rows are matched on.

    ``curve_id``/``strain_name`` are positional (0, 1, 2 ... / CURVE0000 ...),
    so they stop naming the same curve the moment the catalogue changes size.
    Preference order:

      1. ``scenario`` + the per-curve ``seed`` (ground truth v2), when both
         tables carry it: a real identity that survives reordering and
         insertion.  An artefact-carrying curve and its artefact-free twin
         share a seed by construction, so the pair is separated by the row's
         occurrence number inside its own (scenario, seed) group -- a local
         index over two rows the generator emits together, not the global
         positional index that catalogue growth invalidates.
      2. ``scenario`` + the row's position inside its scenario, which is stable
         as long as the generator emits a scenario's curves in the same order.
    """
    def seed_key(df):
        if 'seed' not in df.columns:
            return None
        seeds = df['seed']
        if seeds.isna().any():
            return None
        base = df['scenario'].astype(str) + '#seed=' + seeds.astype(str)
        return base + '#' + base.groupby(base).cumcount().astype(str)

    old_seed, new_seed = seed_key(old), seed_key(new)
    if old_seed is not None and new_seed is not None:
        twinned = bool(old_seed.str.endswith('#1').any() or
                       new_seed.str.endswith('#1').any())
        description = ('scenario + per-curve seed (+ position within a twin pair)'
                       if twinned else 'scenario + per-curve seed')
        return old_seed, new_seed, description

    def ordinal_key(df):
        position = df.groupby(df['scenario'].astype(str)).cumcount()
        return df['scenario'].astype(str) + '#' + position.astype(str)

    return ordinal_key(old), ordinal_key(new), 'scenario + position within scenario'


def compare_ground_truth(committed_csv, regenerated_csv):
    """The hard checks, in the order a reader wants to see them fail."""
    import pandas as pd

    old = pd.read_csv(committed_csv)
    new = pd.read_csv(regenerated_csv)
    results = []

    # ── The catalogue: which scenarios exist, and how many curves each has ──
    old_scen = sorted(set(old['scenario'].astype(str)))
    new_scen = sorted(set(new['scenario'].astype(str)))
    missing = [s for s in new_scen if s not in old_scen]
    dropped = [s for s in old_scen if s not in new_scen]
    lines = []
    if dropped:
        lines.append(f"only in the committed data: {', '.join(dropped)}")
    if missing:
        lines.append(f"only in the regenerated data: {', '.join(missing)}")
    if not lines:
        lines.append(f"{len(old_scen)} scenarios, identical")
    scenario_set_ok = not (missing or dropped)
    results.append(Result('scenario set', scenario_set_ok, lines))

    old_counts = old['scenario'].astype(str).value_counts().to_dict()
    new_counts = new['scenario'].astype(str).value_counts().to_dict()
    changed = sorted(name for name in set(old_counts) & set(new_counts)
                     if old_counts[name] != new_counts[name])
    lines = [f"committed {len(old)} rows, regenerated {len(new)} rows"]
    for name in changed:
        lines.append(f"{name}: {old_counts[name]} -> {new_counts[name]} curve(s)")
    counts_ok = (len(old) == len(new)) and not changed
    results.append(Result('row count (total and per scenario)', counts_ok, lines))

    # ── curve_id is positional: check the set AND what each id now points at ──
    old_ids = set(old['curve_id'].astype(str))
    new_ids = set(new['curve_id'].astype(str))
    lost, gained = sorted(old_ids - new_ids), sorted(new_ids - old_ids)
    lines = []
    if lost:
        lines.append(f"{len(lost)} committed curve_ids are gone, e.g. {', '.join(lost[:5])}")
    if gained:
        lines.append(f"{len(gained)} new curve_ids, e.g. {', '.join(gained[:5])}")
    if not lines:
        lines.append(f"{len(old_ids)} curve_ids, identical")
    results.append(Result('curve_id set', not (lost or gained), lines))

    shared_ids = sorted(old_ids & new_ids)
    if shared_ids:
        o_map = old.set_index(old['curve_id'].astype(str))['scenario'].astype(str)
        n_map = new.set_index(new['curve_id'].astype(str))['scenario'].astype(str)
        moved = [i for i in shared_ids if o_map[i] != n_map[i]]
        lines = []
        if moved:
            lines.append(f"{len(moved)}/{len(shared_ids)} curve_ids name a curve of a "
                         f"different scenario now, e.g. {moved[0]}: "
                         f"{o_map[moved[0]]} -> {n_map[moved[0]]}")
            lines.append("curve_id is a positional index, so this is a catalogue "
                         "change, not a per-curve change; the value checks below "
                         "are keyed on content instead")
        else:
            lines.append(f"{len(shared_ids)} curve_ids, same scenario in both")
        results.append(Result('curve_id -> scenario map', not moved, lines))

    # ── Values, keyed on content ────────────────────────────────────────────
    old_key, new_key, key_description = content_keys(old, new)
    o = old.set_index(old_key)
    n = new.set_index(new_key)
    o = o[~o.index.duplicated()]
    n = n[~n.index.duplicated()]
    shared = [k for k in o.index if k in set(n.index)]
    scenario_of = o.loc[shared, 'scenario'].astype(str) if shared else None

    key_note = f"matched on {key_description}: {len(shared)} curve(s) in both tables"
    if not shared:
        for name in ('expected_class per curve', f"{'/'.join(PARAM_COLUMNS)} within "
                     f"{PARAM_TOLERANCE:g}"):
            results.append(Result(name, False, [
                f"no curve matched on {key_description}"]))
        return results

    o, n = o.loc[shared], n.loc[shared]

    if 'expected_class' in o.columns and 'expected_class' in n.columns:
        flipped = [k for k, a, b in zip(shared,
                                        o['expected_class'].astype(str).values,
                                        n['expected_class'].astype(str).values)
                   if a != b]
        lines = [key_note]
        if flipped:
            by_scen = scenario_of.loc[flipped].value_counts()
            for scen, count in by_scen.items():
                lines.append(f"{scen}: {count} curve(s) changed expected_class")
        else:
            lines.append("expected_class identical for every matched curve")
        results.append(Result('expected_class per curve', not flipped, lines))
    else:
        results.append(Result('expected_class per curve', False,
                              [key_note, 'the column is missing from one of the tables']))

    drifted_lines = [key_note]
    drifted = False
    for column in PARAM_COLUMNS:
        if column not in o.columns or column not in n.columns:
            drifted = True
            drifted_lines.append(f"{column}: missing from one of the tables")
            continue
        delta = (n[column].astype(float).values - o[column].astype(float).values)
        delta = abs(delta)
        bad = delta > PARAM_TOLERANCE
        if not bool(bad.any()):
            continue
        drifted = True
        import pandas as _pd
        by_scen = _pd.DataFrame({'scenario': scenario_of.values[bad],
                                 'delta': delta[bad]}).groupby('scenario')['delta']
        table = by_scen.agg(['size', 'max']).sort_values('max', ascending=False)
        for scen, row in table.head(MAX_DRIFT_LINES).iterrows():
            drifted_lines.append(
                f"{column}: {scen} -- {int(row['size'])} curve(s), max |delta| {row['max']:.6g}")
        if len(table) > MAX_DRIFT_LINES:
            drifted_lines.append(
                f"{column}: ... and {len(table) - MAX_DRIFT_LINES} more scenario(s), "
                f"smallest max |delta| {table['max'].min():.6g}")
    if not drifted:
        drifted_lines.append(f"{len(shared)} curves x {len(PARAM_COLUMNS)} parameters, "
                             f"all within {PARAM_TOLERANCE:g}")
    results.append(Result(f"{'/'.join(PARAM_COLUMNS)} within {PARAM_TOLERANCE:g}",
                          not drifted, drifted_lines))
    return results


def compare_digests(committed_root, regenerated_root):
    """The soft check: sha256 of ground_truth.csv and every DATA/*.csv."""
    def digests(root):
        root = Path(root)
        out = {}
        gt = root / 'test_data' / 'ground_truth.csv'
        if gt.exists():
            out['ground_truth.csv'] = sha256_file(gt)
        for path in sorted((root / 'test_data' / 'DATA').glob('*.csv')):
            out[f"DATA/{path.name}"] = sha256_file(path)
        return out

    old, new = digests(committed_root), digests(regenerated_root)
    shared = sorted(set(old) & set(new))
    differing = [name for name in shared if old[name] != new[name]]
    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    lines = []
    if differing:
        lines.append(f"{len(differing)}/{len(shared)} comparable files hash "
                     f"differently: {', '.join(differing[:6])}"
                     + (' ...' if len(differing) > 6 else ''))
        lines.append("a digest difference on its own is a rebuild under a different "
                     "numpy/BLAS, not a content change -- the hard checks above are "
                     "what decide whether the data moved")
    else:
        lines.append(f"{len(shared)} comparable files, byte-identical")
    if only_old:
        lines.append(f"{len(only_old)} committed file(s) the regeneration did not "
                     f"produce: {', '.join(only_old[:6])}")
    if only_new:
        lines.append(f"{len(only_new)} new file(s): {', '.join(only_new[:6])}")
    return Result('sha256 of the generated files',
                  not (differing or only_old or only_new), lines, fatal=False)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Regenerate the synthetic benchmark and compare it with the "
                    "committed copy.")
    parser.add_argument('--committed', default=str(DEFAULT_COMMITTED),
                        help=f"Committed suite root (default: {DEFAULT_COMMITTED})")
    parser.add_argument('--seed', type=int, default=None,
                        help='Override the seed read from MANIFEST.json')
    parser.add_argument('--n-per-scenario', type=int, default=None,
                        help='Regenerate with a uniform count instead of the '
                             'composition recorded in MANIFEST.json (the default '
                             'replays the recorded generation command, which for '
                             "`make synthetic-data` means each scenario's n_default)")
    parser.add_argument('--tmp-dir', default=None,
                        help='Regenerate here instead of a temporary directory')
    parser.add_argument('--keep', action='store_true',
                        help='Keep the regenerated tree instead of deleting it')
    parser.add_argument('--skip-regenerate', action='store_true',
                        help='Compare against an existing --tmp-dir without regenerating')
    parser.add_argument('--self-check', action='store_true',
                        help='Also regenerate a second time and compare the two '
                             'regenerations, i.e. check the generator is '
                             'deterministic independently of the committed snapshot')
    args = parser.parse_args(argv)

    committed = Path(args.committed).resolve()
    committed_gt = committed / 'test_data' / 'ground_truth.csv'
    if not committed_gt.exists():
        print(f"ERROR: no committed ground truth at {committed_gt}", file=sys.stderr)
        return 2

    plan = generator_settings(committed)
    if args.seed is not None:
        plan.seed = args.seed
        plan.source += ' (+ --seed from the command line)'
    if args.n_per_scenario is not None:
        plan.args = [token for token in plan.args
                     if not token.startswith('--n-per-scenario')]
        plan.args += ['--n-per-scenario', str(args.n_per_scenario)]
        plan.composition = (f"uniform {args.n_per_scenario} per scenario "
                            f"(--n-per-scenario on the command line)")
        plan.source += ' (+ --n-per-scenario from the command line)'

    print("=" * 60)
    print("  SYNTHETIC BENCHMARK REGENERATION CHECK")
    print("=" * 60)
    print(f"  Committed:      {committed}")
    print(f"  Settings from:  {plan.source}")
    print(f"  Seed:           {plan.seed}")
    print(f"  Composition:    {plan.composition}")
    for note in plan.notes:
        print(f"                  {note}")

    tmp_dir = Path(args.tmp_dir).resolve() if args.tmp_dir else Path(
        tempfile.mkdtemp(prefix='synthetic_verify_'))
    tmp_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Regenerated to: {tmp_dir}")
    print()

    try:
        if not args.skip_regenerate:
            if not regenerate(tmp_dir, plan):
                return 2
        regenerated_gt = tmp_dir / 'test_data' / 'ground_truth.csv'
        if not regenerated_gt.exists():
            print(f"ERROR: the generator wrote no {regenerated_gt}", file=sys.stderr)
            return 2

        results = compare_ground_truth(committed_gt, regenerated_gt)
        results.append(compare_digests(committed, tmp_dir))

        print()
        print("  Committed snapshot vs this tree:")
        for result in results:
            print(result.render())
        print()

        if args.self_check:
            # Determinism is a property of the generator alone: it holds (or
            # fails) whatever state the committed snapshot is in, so it is the
            # one thing this check can still answer while a regeneration is
            # pending.
            second_dir = Path(tempfile.mkdtemp(prefix='synthetic_verify_2nd_'))
            try:
                if not regenerate(second_dir, plan):
                    return 2
                self_results = compare_ground_truth(
                    regenerated_gt, second_dir / 'test_data' / 'ground_truth.csv')
                self_results.append(compare_digests(tmp_dir, second_dir))
                print("  Generator determinism (two regenerations, same seed):")
                for result in self_results:
                    print(result.render())
                print()
                results += self_results
            finally:
                shutil.rmtree(second_dir, ignore_errors=True)

        failed = [r for r in results if r.fatal and not r.ok]
        if failed:
            names = [r.name for r in failed]
            print(f"  FAILED: {', '.join(names)}")
            if any(name in COMPOSITION_CHECKS for name in names):
                print("  COMPOSITION CHANGED: the committed snapshot came from a "
                      "different scenario")
                print("  catalogue than this tree generates -- scenarios, per-scenario "
                      "counts or the")
                print("  positional curve_id map moved, so it is out of date rather "
                      "than drifted.")
                print("  Regenerate it (make synthetic-data), retrain the classifier "
                      "and re-publish")
                print("  every number derived from it; record the move in "
                      "docs/BENCHMARK_LEDGER.md.")
            else:
                print("  DRIFT: the catalogue matches curve for curve, but the values "
                      "moved.")
                print("  The scenarios named above are the ones that changed. This is "
                      "a generator")
                print("  regression unless it was intended; if it was, regenerate "
                      "(make synthetic-data)")
                print("  and record the move in docs/BENCHMARK_LEDGER.md.")
            return 1

        soft = [r for r in results if not r.fatal and not r.ok]
        if soft:
            print("  PASSED (with a digest difference: regenerated under a "
                  "different numpy/BLAS)")
        else:
            print("  PASSED: the committed synthetic benchmark regenerates from "
                  "this tree.")
        return 0
    finally:
        if not args.keep and not args.tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        elif args.keep:
            print(f"  Regenerated tree kept at: {tmp_dir}")


if __name__ == '__main__':
    sys.exit(main())
