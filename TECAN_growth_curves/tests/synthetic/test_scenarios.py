"""WP5 acceptance tests: the scenario catalog, the suite composition and the lab prior.

Each test pins a property that was false in the shipped 480-curve benchmark,
with the measured number that showed it:

    coverage (M03)         ALL_SCENARIOS had 37 entries and generate_comprehensive
                           produced 32; bubble_spike, evaporation_drift and the
                           three haldane_* scenarios were defined, documented in
                           the README's Scenario Coverage tables and asserted on by
                           tests/test_audit_fixes_ml.py, and never benchmarked
    borderline (M16)       borderline_delta_od shipped true_A 0.320-0.380 against
                           a config of A=(0.17,0.23) - a +0.150 offset put the
                           "borderline" probe at 2.1-2.5x the 0.15 OD gate, where
                           it scored a meaningless 15/15
    artefacts (M21)        drift_positive and outlier_contaminated differed in
                           name, description and tags ONLY, and
                           outlier_contaminated contained 0.00 points beyond 5
                           robust MADs per curve - fewer than 'standard'
    initial_od (M15)       every GOOD scenario had initial_od <= 0.020 and
                           contamination had 0.238-0.483: a perfect shortcut
    incomplete runs (M07)  detect_incomplete_curve fired on 12/48 real growers
                           (25 %) and 7/421 synthetic (1.7 %), 5 of the 7 being
                           short_experiment
    lab prior (M11)        the amplitude composition was inverted: the 0.6-1.2 OD
                           band was 46 % of the benchmark and 2 % of the lab; the
                           <0.02 OD dead-well band 4 % against 30 %

The statistical tests run on ONE module-scoped suite generated at the catalog's
own composition (1105 curves, ~14 s), because "8 % of growers" and "AUC < 0.75"
are properties of the shipped composition, not of an arbitrary subsample.
"""

import importlib.util
import inspect
import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SYNTH = REPO_ROOT / 'synthetic_data'
SCRIPTS = REPO_ROOT / 'scripts'
if str(SYNTH) not in sys.path:
    sys.path.insert(0, str(SYNTH))

from src.curve_scenarios import (           # noqa: E402
    ALL_SCENARIOS, BAD_CURVE_SCENARIOS, EDGE_CASE_SCENARIOS,
    GOOD_GROWTH_SCENARIOS, LAB_PRIOR_PATH, ScenarioConfig, ScenarioSampler,
    audit_realisation, get_comprehensive_test_config, get_lab_weighted_config,
    is_twin_row, lab_band_names, load_lab_prior, NO_BLANK,
)
from src.data_generator import SyntheticGrowthCurveGenerator   # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope='module')
def suite():
    """The whole catalog at its own composition, generated once."""
    generator = SyntheticGrowthCurveGenerator(seed=42)
    return generator.generate_comprehensive_test_set(
        config=get_comprehensive_test_config(), seed=42, verbose=False)


@pytest.fixture(scope='module')
def pipeline():
    """scripts/01_growth_curve_analysis.py (module name starts with a digit)."""
    spec = importlib.util.spec_from_file_location(
        'growth_curve_analysis', str(SCRIPTS / '01_growth_curve_analysis.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The generator hooks WP5's catalog needs from WP3's data_generator (the
#: Contract A extras).  They used to be a skip condition, which kept the suite
#: green while four noise rungs all ran at one preset and 35 curves labelled
#: BAD for "two of three wells are dead" shipped as ordinary growers.  A
#: declaration nothing consumes is a defect in the DATA, so the tests below
#: fail instead: red until data_generator.py reads these fields.
GENERATOR_HOOKS = ('well_sigma_od', 'spike_window_h', 'outlier_well_A_scale',
                   'n_outlier_wells')

COMPANION_HINT = (
    "data_generator.py must consume the scenario fields the sampler emits; "
    "see WP5 companion_changes_needed (the hook is ~40 lines in "
    "generate_single_curve/_apply_noise_stack)")


def _generator_source() -> str:
    return (inspect.getsource(SyntheticGrowthCurveGenerator._apply_noise_stack)
            + inspect.getsource(SyntheticGrowthCurveGenerator.generate_single_curve))


def test_the_generator_consumes_the_declared_scenario_fields():
    """Every Contract A field the catalog declares is read by the generator.

    This is the root cause of three of the data defects below, pinned once so
    the failure names its own fix instead of surfacing as "the noise ladder is
    not a ladder".
    """
    source = _generator_source()
    missing = [h for h in GENERATOR_HOOKS if h not in source]
    assert not missing, f"data_generator ignores {missing}: {COMPANION_HINT}"


def _rows(suite, scenario):
    return suite[suite['scenario'] == scenario]


# ---------------------------------------------------------------------------
# Acceptance 1: every defined scenario is generated
# ---------------------------------------------------------------------------

def test_ground_truth_covers_every_defined_scenario(suite):
    """Every scenario in the catalog appears in the generated ground truth.

    ALL_SCENARIOS had 37 entries and only 32 were generated, so five defined,
    documented and asserted-on scenarios were never benchmarked.
    """
    assert set(suite['scenario']) == set(ALL_SCENARIOS)
    assert set(get_comprehensive_test_config()) == set(ALL_SCENARIOS)


def test_every_scenario_is_in_exactly_one_dict():
    names = list(GOOD_GROWTH_SCENARIOS) + list(BAD_CURVE_SCENARIOS) + list(EDGE_CASE_SCENARIOS)
    assert len(names) == len(set(names)) == len(ALL_SCENARIOS)
    for name, cfg in ALL_SCENARIOS.items():
        assert cfg.name == name, f"{name} carries name={cfg.name!r}"


# ---------------------------------------------------------------------------
# Acceptance 2: the borderline probe sits on its gate
# ---------------------------------------------------------------------------

def test_borderline_scenarios_sit_near_their_gate(suite):
    """borderline_delta_od must straddle the 0.15 OD rise gate, not sit at 2.4x it."""
    rows = _rows(suite, 'borderline_delta_od')
    assert len(rows) >= 10
    assert rows['delta_od'].min() > 0.15, rows['delta_od'].min()
    assert rows['delta_od'].max() < 0.30, rows['delta_od'].max()
    # ... and the truth curve's own amplitude is in the configured band, i.e.
    # nothing adds a hidden offset to it.
    assert rows['true_A'].max() < 0.30


def test_low_A_still_straddles_the_gate(suite):
    """low_A keeps curves below 0.5 OD - 22.8 % of audited real GOOD are there.

    The band was widened to (0.15, 0.5) and NOT lifted off the gate: draws below
    0.2 OD must still happen.
    """
    rows = _rows(suite, 'low_A')
    assert rows['true_A'].min() < 0.25
    assert rows['true_A'].max() <= 0.55


# ---------------------------------------------------------------------------
# Acceptance 3: artefact scenarios contain artefacts
# ---------------------------------------------------------------------------

def _tail_rise(time, od):
    """Linear fit to the last 50 % of the trace, scaled by the run length."""
    t = np.asarray(time, dtype=float)
    y = np.asarray(od, dtype=float)
    half = len(t) // 2
    slope = np.polyfit(t[half:], y[half:], 1)[0]
    return float(slope * (t[-1] - t[0]))


def _n_local_outliers(od, window=21, k=5.0):
    """Points beyond ``k`` robust MADs of the LOCAL first difference.

    Local, because the growth phase's own first differences are large: a global
    MAD flags the exponential phase of every curve, including 'standard'.
    """
    d = pd.Series(np.diff(np.asarray(od, dtype=float)))
    med = d.rolling(window, center=True, min_periods=5).median()
    mad = (d - med).abs().rolling(window, center=True, min_periods=5).median()
    scale = 1.4826 * mad
    scale = scale.where(scale > 1e-9, np.nan)
    return int((((d - med).abs() / scale) > k).sum())


def _artefact_rows(suite, scenario):
    """The artefact half of an artefact scenario (its twin carries none).

    The twin still carries the preset's own per-well drift, so "has a drift" is
    not the discriminator: the scenario's declared artefact is.
    """
    cfg = ALL_SCENARIOS[scenario]
    rows = _rows(suite, scenario)
    n_spikes = cfg.n_spikes if isinstance(cfg.n_spikes, int) else max(cfg.n_spikes)
    if n_spikes > 0:
        return rows[rows['spike_indices'].astype(str).str.len() > 0]
    if cfg.drift_od_per_h is not None:
        floor = 0.9 * min(abs(v) for v in cfg.drift_od_per_h)
        return rows[rows['drift_od_per_h'].abs() >= floor]
    return rows


def _twin_rows(suite, scenario):
    """The artefact-free half."""
    rows = _rows(suite, scenario)
    artefact = _artefact_rows(suite, scenario)
    return rows.drop(index=artefact.index)


def test_artefact_scenarios_contain_artefacts(suite):
    """evaporation_drift really drifts; the spike scenarios really spike.

    Today outlier_contaminated has 0.00 points beyond 5 robust MADs per curve,
    fewer than 'standard', because it was generatively identical to
    drift_positive.
    """
    evap = _artefact_rows(suite, 'evaporation_drift')
    assert len(evap) >= 10
    rises = np.array([_tail_rise(r.time, r.od600) for r in evap.itertuples()])
    assert rises.min() >= 0.04, f"evaporation_drift tail rise {rises.min():.4f} OD"

    for scenario in ('bubble_spike', 'outlier_contaminated'):
        rows = _artefact_rows(suite, scenario)
        assert len(rows) >= 10, scenario
        counts = np.array([_n_local_outliers(r.od600) for r in rows.itertuples()])
        assert counts.min() >= 1, f"{scenario} has a curve with no spike"


def test_artefact_free_twins_have_no_artefact(suite):
    """Every artefact scenario ships an artefact-free twin of every curve."""
    for scenario in ('evaporation_drift', 'bubble_spike', 'outlier_contaminated',
                     'drift_positive', 'negative_drift'):
        rows = _rows(suite, scenario)
        twins = _twin_rows(suite, scenario)
        assert len(twins) == len(rows) // 2, scenario
        assert (twins['twin_curve_id'].astype(str) != '').all()
        assert (twins['spike_indices'].astype(str).str.len() == 0).all(), scenario


def test_negative_drift_declines(suite):
    """G3 H2O_IMID3 falls 0.0431 -> 0.0293 over 99 h; the scenario must too.

    Per curve the decline (-0.014 OD over the run) is the same size as the
    slow instrument wander a real dead well shows, so the property is asserted
    on the population and on the applied rate, not on every single trace.
    """
    rows = _artefact_rows(suite, 'negative_drift')
    rises = np.array([_tail_rise(r.time, r.od600) for r in rows.itertuples()])
    assert np.median(rises) < 0.0, f"negative_drift median rise {np.median(rises):+.4f}"
    assert (rows['drift_od_per_h'] < 0).all()
    assert np.mean(rises < 0) >= 0.8, rises


def test_spikes_land_inside_the_declared_window(suite):
    """bubble_spike puts its bubbles in the exponential phase, not the plateau."""
    rows = _artefact_rows(suite, 'bubble_spike')
    lo, hi = ALL_SCENARIOS['bubble_spike'].spike_window_h
    for row in rows.itertuples():
        idx = [int(i) for i in str(row.spike_indices).split(';') if i != '']
        assert idx, 'no spikes recorded'
        times = np.asarray(row.time, dtype=float)[idx]
        assert times.min() >= lo - 1e-9 and times.max() <= hi + 1e-9, times


# ---------------------------------------------------------------------------
# Acceptance 4: no two scenarios are the same experiment
# ---------------------------------------------------------------------------

def test_no_two_scenarios_are_identical():
    """drift_positive and outlier_contaminated used to agree on every field."""
    seen = {}
    for name, cfg in ALL_SCENARIOS.items():
        key = cfg.generative_fingerprint()
        assert key not in seen, f"{name} is generatively identical to {seen[key]}"
        seen[key] = name


def test_instrument_noise_is_a_real_selector():
    """It was a documented no-op; it now means 'this scenario has an artefact'."""
    for name, cfg in ALL_SCENARIOS.items():
        declared = (cfg.n_spikes if isinstance(cfg.n_spikes, int) else max(cfg.n_spikes)) > 0 \
            or cfg.drift_od_per_h is not None \
            or cfg.well_sigma_od is not None
        assert cfg.instrument_noise == declared, name
    # The three scenarios tests/test_audit_fixes_ml.py pins by name: an
    # over-subtracted blank is preprocessing, not an instrument artefact, so
    # 'standard' stays False even though it goes through the blank stage.
    assert ALL_SCENARIOS['evaporation_drift'].instrument_noise is True
    assert ALL_SCENARIOS['bubble_spike'].instrument_noise is True
    assert ALL_SCENARIOS['standard'].instrument_noise is False


# ---------------------------------------------------------------------------
# Acceptance 5: initial_od is not the label
# ---------------------------------------------------------------------------

def test_initial_od_does_not_separate_classes(suite):
    """A classifier reading only initial_od must be near useless.

    Every GOOD scenario used to start below 0.020 OD and contamination at
    0.238-0.483, so one threshold on the starting OD reproduced the label.
    """
    y = (suite['expected_class'] == 'GOOD').to_numpy().astype(int)
    x = suite['initial_od'].to_numpy(dtype=float)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        model = LogisticRegression().fit(x.reshape(-1, 1), y)
        auc = roc_auc_score(y, model.predict_proba(x.reshape(-1, 1))[:, 1])
    except ImportError:                                     # pragma: no cover
        order = np.argsort(x)
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, len(x) + 1)
        n1, n0 = int(y.sum()), int((1 - y).sum())
        auc = (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        auc = max(auc, 1 - auc)
    assert auc < 0.75, f"initial_od alone scores AUC {auc:.3f}"


def test_a_high_baseline_is_not_automatically_bad():
    """The real grower baseline IQR is 0.019-0.066 OD with a maximum of 0.144."""
    good = {n: c for n, c in ALL_SCENARIOS.items() if c.expected_class == 'GOOD'}
    assert max(c.initial_od[1] for c in good.values()) >= 0.10
    assert ALL_SCENARIOS['high_baseline_good'].initial_od == (0.08, 0.15)
    # ... and the contamination boundary scenario overlaps that band.
    frac = ALL_SCENARIOS['contamination_offset'].initial_od_offset_fraction
    assert frac == (0.10, 0.20)


# ---------------------------------------------------------------------------
# Acceptance 6: incomplete runs are actually in the suite
# ---------------------------------------------------------------------------

def test_incomplete_prevalence(suite, pipeline):
    """detect_incomplete_curve must fire on a real population of growers.

    Real: 12/48 growers (25 %).  Shipped suite: 7/421 (1.7 %), and 5 of those 7
    were short_experiment - a 20 h run 98.9 % of the way through its rise, i.e.
    no synthetic curve was genuinely mid-exponential at its last point.
    """
    growers = suite[suite['delta_od'] > 0.15]
    assert len(growers) > 200
    hits = [bool(pipeline.detect_incomplete_curve(
        np.asarray(r.time, dtype=float), np.asarray(r.od600, dtype=float))[0])
        for r in growers.itertuples()]
    n_incomplete = int(np.sum(hits))
    assert n_incomplete >= 40, n_incomplete
    assert n_incomplete / len(growers) >= 0.08, n_incomplete / len(growers)


def test_incomplete_family_really_ends_mid_growth(suite):
    """The 24 h family must end at 60-85 % of A, not at 99 %."""
    rows = _rows(suite, 'incomplete_24h')
    frac = rows['true_frac_of_A_reached'].to_numpy(dtype=float)
    assert np.nanmedian(frac) < 0.90
    assert np.nanmax(frac) <= 0.95
    assert (rows['duration_hours'] == 24.0).all()
    assert (rows['n_points'] == 96).all()      # the literal Group 1 geometry
    assert (rows['scoring_mode'] == 'a_not_identifiable').all()


def test_incomplete_geometry_is_not_the_label(suite):
    """A 24 h run that DOES plateau is in the suite, so the grid cannot predict."""
    plateaued = _rows(suite, 'run_24h_plateaued')
    assert len(plateaued) >= 5
    assert (plateaued['duration_hours'] == 24.0).all()
    assert plateaued['true_frac_of_A_reached'].min() > 0.95


def test_run_ends_just_after_the_primary_phase(suite):
    """run_ends_at_plateau_edge stops a few samples after t99, not at t_inflection."""
    rows = _rows(suite, 'run_ends_at_plateau_edge')
    assert len(rows) >= 5
    frac = rows['true_frac_of_A_reached'].to_numpy(dtype=float)
    # The Gompertz inflection - the rule the plan warns against - sits at 37 %.
    assert frac.min() > 0.95
    # Each curve gets its own duration, so they are not all on one grid.
    assert rows['duration_hours'].nunique() > 1


# ---------------------------------------------------------------------------
# Acceptance 7: the lab prior is derived, not typed
# ---------------------------------------------------------------------------

def test_lab_prior_is_derived():
    """lab_prior.json must regenerate byte-identically from data/raw."""
    script = SYNTH / 'scripts' / 'derive_lab_prior.py'
    assert script.exists()
    result = subprocess.run([sys.executable, str(script), '--check'],
                            capture_output=True, text=True, cwd=str(REPO_ROOT),
                            timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr


def test_lab_prior_is_the_measured_composition():
    prior = load_lab_prior()
    assert prior['n_conditions'] == 92           # the four real plates
    fractions = {b['name']: b['fraction'] for b in prior['bands']}
    assert abs(sum(fractions.values()) - 1.0) < 1e-5   # 6-dp rounding
    # The two bands the old suite inverted, to one decimal place.
    assert round(100 * fractions['dead'], 1) == 30.4
    assert round(100 * fractions['full'], 1) == 44.6


def test_every_scenario_names_a_real_band():
    names = set(lab_band_names())
    for name, cfg in ALL_SCENARIOS.items():
        assert cfg.lab_band in names, f"{name}: lab_band={cfg.lab_band!r}"


def test_lab_weighting_follows_the_prior():
    """--weighting lab must move the dead-well band up and the 0.6-1.2 band down."""
    uniform = get_comprehensive_test_config()
    lab = get_lab_weighted_config(sum(uniform.values()))
    assert set(lab) == set(uniform)

    def band_share(config):
        total = sum(config.values())
        out = {}
        for name, n in config.items():
            band = ALL_SCENARIOS[name].lab_band
            out[band] = out.get(band, 0) + n / total
        return out

    u, l = band_share(uniform), band_share(lab)
    prior = {b['name']: b['fraction'] for b in load_lab_prior()['bands']}
    assert l['dead'] > u['dead']
    assert abs(l['dead'] - prior['dead']) < abs(u['dead'] - prior['dead'])
    assert abs(l['full'] - prior['full']) < 0.15


# ---------------------------------------------------------------------------
# Label rule, scoring mode and composition
# ---------------------------------------------------------------------------

def test_label_rule_is_declared_per_scenario_and_scoped(suite):
    """Only the scenarios that opt in are labelled by the oracle rule."""
    oracle = {n for n, c in ALL_SCENARIOS.items() if c.label_rule == 'oracle'}
    assert {'high_noise', 'borderline_noise', 'borderline_r2_bad',
            'borderline_r2_good'} <= oracle
    assert {'noise_ladder_1', 'noise_ladder_2', 'noise_ladder_3',
            'noise_ladder_4'} <= oracle
    assert 'standard' not in oracle and 'contamination' not in oracle

    for name, rows in suite.groupby('scenario'):
        cfg = ALL_SCENARIOS[name]
        assert (rows['label_rule'] == cfg.label_rule).all(), name
        if cfg.label_rule == 'scenario':
            assert (rows['expected_class'] == cfg.expected_class).all(), name


def test_the_noisy_bad_scenarios_are_relabelled_by_the_rule(suite):
    """M02: their parameters are recoverable, so 'BAD' was a statement about R².

    The three scenarios contributed 35 of the 61 headline errors while every
    curve in them has an oracle A/mu error under 20 %.
    """
    flipped = 0
    for name in ('high_noise', 'borderline_noise', 'borderline_r2_bad'):
        rows = _rows(suite, name)
        assert ALL_SCENARIOS[name].expected_class == 'BAD'
        flipped += int((rows['expected_class'] == 'GOOD').sum())
    assert flipped >= 30, flipped


def test_expected_class_stays_binary(suite):
    assert set(suite['expected_class']) <= {'GOOD', 'BAD'}
    assert set(suite['scoring_mode']) <= {
        'standard', 'a_not_identifiable', 'preprocessing_blind', 'model_mismatch'}


def test_hard_scenarios_are_not_softened():
    """sparse_data, fast_growth and low_A stay exactly as hard, in the headline."""
    assert ALL_SCENARIOS['sparse_data'].time_resolution == 1.0
    assert ALL_SCENARIOS['sparse_data'].expected_class == 'GOOD'
    assert ALL_SCENARIOS['sparse_data'].scoring_mode == 'standard'
    assert ALL_SCENARIOS['fast_growth'].mu == (0.4, 0.75)
    assert ALL_SCENARIOS['fast_growth'].scoring_mode == 'standard'
    assert ALL_SCENARIOS['low_A'].A[0] == 0.15
    assert ALL_SCENARIOS['low_A'].n_default >= 15
    # The historic catastrophic decline is kept, named and reported.
    assert ALL_SCENARIOS['death_phase_extreme'].include_death_phase is True
    assert ALL_SCENARIOS['death_phase_extreme'].k_death == (0.05, 0.1)


def test_composition_lives_on_the_scenarios():
    """n_default replaces the --extra-good / --extra-borderline flags."""
    config = get_comprehensive_test_config()
    assert config['flat_no_growth'] >= 25
    assert config['minimal_growth'] >= 25
    assert config['pesticide_only'] >= 25
    assert config['fit_failure'] >= 25
    for name in ('standard', 'very_clean', 'drift_positive', 'outlier_contaminated'):
        assert config[name] == 10, name
    assert get_comprehensive_test_config(7)['standard'] == 7


def test_bad_non_growers_are_a_real_share(suite):
    """They were 12.5 % of the suite against ~43 % of real conditions."""
    non_growers = suite[(suite['expected_class'] == 'BAD') & (suite['true_A'] < 0.15)]
    assert len(non_growers) / len(suite) >= 0.15


# ---------------------------------------------------------------------------
# The sampler's two solved quantities and the Contract A pass-through
# ---------------------------------------------------------------------------

def test_sampler_passes_every_contract_a_field():
    """A field nothing forwards is a field that does nothing (M21's mechanism)."""
    params = ScenarioSampler(ALL_SCENARIOS['bubble_spike'], seed=0).sample_parameters()
    for key in ('label_rule', 'scoring_mode', 'lab_band', 'shape', 'A2', 'mu2',
                'lambda2', 'gap_hours', 'drift_od_per_h', 'n_spikes', 'spike_od',
                'spike_decay_points', 'blank_offset_od', 'replicate_A_cv',
                'replicate_mu_cv', 'replicate_lambda_sd', 'outlier_well_prob',
                'death_loss_frac', 't_death_offset_h', 'media', 'well_sigma_od'):
        assert key in params, key
    haldane = ScenarioSampler(ALL_SCENARIOS['haldane_strong_inhibition'],
                              seed=0).sample_parameters()
    for key in ('Ks', 'Ki', 'q', 'S0'):
        assert key in haldane, key
    assert haldane['Ki'] == 0.15 and haldane['S0'] == 1.0


def test_ranges_reach_the_shape_library_as_ranges():
    """curve_shapes draws the second phase on the curve's own stream."""
    params = ScenarioSampler(ALL_SCENARIOS['diauxic_second_larger'],
                             seed=0).sample_parameters()
    assert params['A2'] == (0.8, 1.2)
    assert params['shape'] == 'diauxic_second_larger'
    # ... while the count of spikes is a scalar the generator can use directly.
    spikes = ScenarioSampler(ALL_SCENARIOS['outlier_contaminated'],
                             seed=0).sample_parameters()['n_spikes']
    assert isinstance(spikes, int) and 3 <= spikes <= 8


def test_mu_is_solved_from_the_requested_end_fraction():
    """'the run ends at 60-85 % of A' is the property; mu realises it."""
    cfg = ALL_SCENARIOS['incomplete_24h']
    for seed in range(8):
        params = ScenarioSampler(cfg, seed=seed).sample_parameters()
        A, mu, lam = params['A'], params['mu'], params['lambda_']
        t_end = params['duration_hours'] - params['time_resolution']
        value = A * np.exp(-np.exp(mu * np.e / A * (lam - t_end) + 1.0))
        assert 0.55 <= value / A <= 0.90, value / A
        assert abs(value / A - params['end_fraction_of_A']) < 1e-6


def test_contamination_offset_is_a_fraction_of_the_final_od():
    cfg = ALL_SCENARIOS['contamination_offset']
    for seed in range(8):
        p = ScenarioSampler(cfg, seed=seed).sample_parameters()
        frac = p['initial_od'] / (p['initial_od'] + p['A'])
        assert 0.10 <= frac <= 0.20, frac


def test_sampling_is_deterministic_in_the_seed():
    for name in ('standard', 'incomplete_24h', 'bubble_spike', 'noise_ladder_3'):
        a = ScenarioSampler(ALL_SCENARIOS[name], seed=11).sample_parameters()
        b = ScenarioSampler(ALL_SCENARIOS[name], seed=11).sample_parameters()
        assert a == b, name


def test_scenarios_are_constructible_from_their_own_fields():
    """The clone test tests/test_audit_fixes_ml.py runs, on the new fields too."""
    names = {f.name for f in fields(ScenarioConfig)}
    for name, cfg in ALL_SCENARIOS.items():
        clone = ScenarioConfig(**{f: getattr(cfg, f) for f in names})
        assert clone.generative_fingerprint() == cfg.generative_fingerprint(), name


def test_an_unknown_scoring_mode_is_rejected():
    with pytest.raises(ValueError):
        ScenarioConfig(name='x', description='', expected_class='GOOD',
                       scoring_mode='easy')
    with pytest.raises(ValueError):
        ScenarioConfig(name='x', description='', expected_class='MAYBE')


# ---------------------------------------------------------------------------
# The new families do what their names say
# ---------------------------------------------------------------------------

def test_the_noise_ladder_is_a_ladder(suite):
    """Rungs are ordered in realised oracle error, and none of them is the rule.

    Labels come from each curve's OWN realised error, so a rung is a
    population, not a label: measured at seed 42 the rungs put 1, 4, 8 and 12
    of 15 curves past the 20 % rule.
    """
    medians = []
    bad = []
    for rung in ('noise_ladder_1', 'noise_ladder_2', 'noise_ladder_3', 'noise_ladder_4'):
        rows = _rows(suite, rung)
        assert len(rows) >= 10, rung
        # A curve whose oracle fit did not converge has no error to rank; it
        # is BAD by the rule and is counted below, not averaged in here.
        errs = np.vstack([rows['oracle_A_relerr'].to_numpy(dtype=float),
                          rows['oracle_mu_relerr'].to_numpy(dtype=float)])
        err = np.array([np.nanmax(col) if np.isfinite(col).any() else np.nan
                        for col in errs.T])
        medians.append(float(np.nanmedian(err)))
        bad.append(int((rows['expected_class'] == 'BAD').sum()))
    assert medians == sorted(medians), medians
    assert bad == sorted(bad), bad
    assert bad[0] <= 2 and bad[-1] >= 8, bad


def test_replicate_disagreement_is_generated(suite):
    """G4 PERMETHRINANDLB-CISPERM1 = [0.016, 1.063, 0.010] is called GOOD."""
    high = _rows(suite, 'replicate_outlier_high')
    assert (high['outlier_well']).all()
    # Two dead wells of three: the condition mean is about a third of a well.
    assert high['true_frac_of_A_reached'].median() < 0.5
    drop = _rows(suite, 'replicate_dropout')
    assert 0.5 < drop['true_frac_of_A_reached'].median() < 0.8
    for name in ('replicate_outlier_high', 'replicate_dropout',
                 'edge_well_evaporation', 'contaminated_blank'):
        assert ALL_SCENARIOS[name].scoring_mode == 'preprocessing_blind', name


def test_multi_phase_scenarios_carry_a_second_phase(suite):
    """The suite has never contained the shape that drove the human calibration."""
    for name, shape in (('slow_secondary_drift', 'slow_secondary_drift'),
                        ('diauxic_second_larger', 'diauxic_second_larger'),
                        ('truncation_challenge', 'two_phase_gompertz')):
        rows = _rows(suite, name)
        assert (rows['shape'] == shape).all(), name
        assert rows['t_second_start'].notna().all(), name
        # The primary phase completes before the second one starts.
        assert (rows['t_primary_95'] < rows['t_second_start']).all(), name

    drift = _rows(suite, 'slow_secondary_drift')
    gains = []
    for row in drift.itertuples():
        t = np.asarray(row.time, dtype=float)
        od = np.asarray(row.od600, dtype=float)
        gains.append(od[-1] - od[np.searchsorted(t, 50.0)])
    # Real: 58.1 % of growers gain > 0.05 OD after t=50 h, p90 +0.20, max 0.516.
    assert np.median(gains) > 0.05
    assert max(gains) < 0.60


def test_decline_scenarios_lose_a_realistic_fraction(suite):
    """Real end-of-run loss over 48 growers: median 0.005, p90 0.186, max 0.372."""
    for name, lo, hi in (('death_phase_moderate', 0.03, 0.25),
                         ('death_phase_severe', 0.20, 0.50),
                         ('partial_lysis', 0.15, 0.40),
                         ('sudden_crash', 0.20, 0.45)):
        rows = _rows(suite, name)
        losses = []
        for row in rows.itertuples():
            od = np.asarray(row.od600, dtype=float)
            peak = float(np.max(od))
            losses.append((peak - float(np.mean(od[-5:]))) / peak if peak > 0 else 0.0)
        assert lo <= float(np.median(losses)) <= hi, (name, np.median(losses))
    # ... and the historic collapse survives, named and out of distribution.
    extreme = _rows(suite, 'death_phase_extreme')
    losses = []
    for row in extreme.itertuples():
        od = np.asarray(row.od600, dtype=float)
        losses.append((np.max(od) - np.mean(od[-5:])) / np.max(od))
    assert np.median(losses) > 0.5
    assert 'out_of_distribution' in ALL_SCENARIOS['death_phase_extreme'].tags


def test_haldane_scenarios_have_different_kinetics():
    """All three used to get Ki=10, Ks=0.1, q=0.1, S0=1.0 regardless of intent."""
    mild = ALL_SCENARIOS['haldane_mild_inhibition']
    strong = ALL_SCENARIOS['haldane_strong_inhibition']
    total = ALL_SCENARIOS['haldane_total_inhibition']
    assert mild.Ki > 10 * mild.Ks              # "Ki >> Ks"
    assert 0.5 * strong.Ks <= strong.Ki <= 3 * strong.Ks   # "Ki ~ Ks"
    assert total.S0 > mild.S0 and total.Ki < strong.Ki     # "very high substrate"
    assert len({(c.Ks, c.Ki, c.q, c.S0) for c in (mild, strong, total)}) == 3


def test_clipped_baseline_family_reaches_the_zero_floor(suite):
    """0/480 synthetic curves were ever clipped; 30.4 % of real ones are."""
    all_zero = _rows(suite, 'all_zero')
    assert all([np.all(np.asarray(r.od600) == 0.0) for r in all_zero.itertuples()])
    clipped = _rows(suite, 'clipped_baseline')
    opens = [float(np.std(np.asarray(r.od600)[:10])) for r in clipped.itertuples()]
    assert np.median(opens) <= 1e-8
    grows = _rows(suite, 'clipped_then_grows')
    assert (grows['shape'] == 'clipped_then_grows').all()
    assert grows['delta_od'].min() > 0.15


def test_the_suite_contains_clipped_baselines_across_the_board(suite):
    """The pipeline's noise-floor branch must be entered by more than dead wells."""
    def clipped(od):
        od = np.asarray(od, dtype=float)
        k = min(10, max(1, len(od) // 5))
        return float(np.std(od[:k])) <= 1e-8
    flags = np.array([clipped(r.od600) for r in suite.itertuples()])
    assert flags.mean() >= 0.08, flags.mean()
    # ... and not only in the scenarios whose name says "clipped".
    named = suite['scenario'].isin(['all_zero', 'clipped_baseline', 'clipped_then_grows'])
    assert flags[~named.to_numpy()].sum() >= 20


# ---------------------------------------------------------------------------
# The data has to carry what the catalog declares, and the labels have to
# follow the curve that was written
# ---------------------------------------------------------------------------

def test_the_suite_carries_what_the_scenarios_declare(suite):
    """No scenario ships a property it only declared.

    ``audit_realisation`` measures the ground-truth rows - spikes, spike
    windows, signed drifts, per-well sigma, dead replicate wells and shapes -
    against the config that asked for them.  Six declarations were inert when
    this was written: the four ``noise_ladder`` rungs all ran at one preset
    (realised sigma 0.030-0.068 for a declared 0.06/0.10/0.25/0.35), both
    replicate scenarios produced ordinary growers, and the spike windows were
    ignored.  ``generate_comprehensive.py`` refuses to write a suite that fails
    this, so the check and the shipped data cannot disagree.
    """
    problems = audit_realisation(suite)
    assert problems == [], '\n'.join(problems)


def test_class_balance_sits_in_the_lab_band(suite):
    """The default composition is 49-60 % GOOD, the lab's own prevalence.

    The shipped 480-curve benchmark was 71.9 % GOOD and the rebuild's first
    pass was 71.3 % under ``--weighting uniform`` (the default, and what
    ``make synthetic-data`` runs), so a headline published from the default
    command still carried the old prevalence.  The real plates put 30.4 % of
    conditions in the dead band and 13.0 % in the trace band
    (config/lab_prior.json), leaving 49-60 % GOOD depending on where the
    marginal 0.1-0.3 OD band falls.
    """
    good = float((suite['expected_class'] == 'GOOD').mean())
    assert 0.49 <= good <= 0.60, f"{good:.3f} GOOD"


def test_no_bad_row_is_an_unexplained_grower(suite):
    """Every BAD row can point at the property of ITS OWN curve that made it BAD.

    50 of 1105 rows failed this in the first pass: 35 replicate curves whose
    dead wells were never generated (true_A 0.78-1.35, oracle error 0.9 %) and
    the 15 artefact-free twins of ``contaminated_blank``, which carry neither
    the drift nor the blank ramp and were BAD only because their primary was.
    """
    bad = suite[suite['expected_class'] == 'BAD']
    reasons = {
        # no usable amplitude
        'flat': bad['delta_od'] < 0.15,
        # the parameters are not recoverable from the curve
        'unrecoverable': (~bad['oracle_fit_ok'].astype(bool))
                         | (bad['oracle_A_relerr'] > 0.20)
                         | (bad['oracle_mu_relerr'] > 0.20),
        # something is in the well before the run starts
        'offset': bad['initial_od'] / (bad['initial_od'] + bad['true_A'].fillna(0.0)) > 0.08,
        # a preprocessing artefact the trace cannot show: the condition mean is
        # not the biology (dead wells averaged in) or a subtracted blank grew
        'preprocessing': ((bad['scoring_mode'] == 'preprocessing_blind')
                          & ((bad['true_frac_of_A_reached'] < 0.80)
                             | (bad['drift_od_per_h'].abs() > 1e-4))),
    }
    explained = np.zeros(len(bad), dtype=bool)
    for mask in reasons.values():
        explained |= mask.fillna(False).to_numpy()
    unexplained = bad[~explained]
    assert unexplained.empty, (
        unexplained.groupby('scenario').size().to_dict())


def test_artefact_free_twins_are_not_labelled_by_an_artefact_they_lack(suite):
    """A twin is labelled by its own curve, not by the artefact it was stripped of.

    ``evaporation_drift`` and ``negative_drift`` twins are honestly BAD - their
    base curves are dead wells at delta_od 0.009-0.012.  ``contaminated_blank``
    is the case where the base curve is a healthy grower, so it ships without a
    twin instead of shipping 15 growers labelled BAD.
    """
    rows = suite.to_dict('records')
    by_scenario = {}
    for row in rows:
        by_scenario.setdefault(row['scenario'], []).append(row)
    offenders = []
    for name, srows in by_scenario.items():
        for row in srows:
            if not is_twin_row(row, srows):
                continue
            if row['expected_class'] != 'BAD':
                continue
            recoverable = (bool(row['oracle_fit_ok'])
                           and float(row['oracle_A_relerr']) <= 0.20
                           and float(row['oracle_mu_relerr']) <= 0.20
                           and float(row['delta_od']) >= 0.15)
            if recoverable:
                offenders.append((name, row['strain_name']))
    assert not offenders, offenders
    assert ALL_SCENARIOS['contaminated_blank'].generate_twin is False


def test_the_replicate_family_really_loses_its_wells(suite):
    """The condition mean is what the pipeline reads, and it must be wrong.

    G4 PERMETHRINANDLB-CISPERM1 = [0.016, 1.063, 0.010] is called GOOD off one
    well of three.  Two dead wells of three put the mean at ~1/3 of a well;
    one uninoculated well of three puts it at ~2/3.
    """
    high = _rows(suite, 'replicate_outlier_high')
    ratio = (high['true_A'] / high['requested_A']).median()
    assert 0.25 <= ratio <= 0.45, ratio
    drop = _rows(suite, 'replicate_dropout')
    ratio = (drop['true_A'] / drop['requested_A']).median()
    assert 0.55 <= ratio <= 0.80, ratio


def test_the_no_blank_sentinel_is_the_generators():
    """``NO_BLANK`` is spelled as a literal to avoid a circular import; pin it.

    The blank stage is on by default, so a scenario whose ``initial_od`` IS a
    post-blank baseline has to opt out explicitly.  If the two constants ever
    drift apart the generator raises on the value, which is loud - but this
    says which two lines have to agree.
    """
    from src import data_generator
    assert NO_BLANK == data_generator.BLANK_OFF
    for name in ('negative_drift', 'evaporation_drift'):
        assert ALL_SCENARIOS[name].blank_offset_od == NO_BLANK, name


def test_the_readme_tables_are_rendered_from_the_catalog():
    """``--print-readme-tables`` covers every scenario, with its count.

    synthetic_data/README.md is WP8's file and went stale as soon as the
    catalog grew (it documented 9/8/20 scenarios against 10/15/39, which is
    what tests/test_audit_fixes_ml.py::TestScenarioDocs fails on).  The tables
    are therefore GENERATED here; this test pins the generator, and the README
    is a paste of its output.
    """
    proc = subprocess.run(
        [sys.executable, str(SYNTH / 'scripts' / 'generate_comprehensive.py'),
         '--print-readme-tables'],
        capture_output=True, text=True, cwd=str(SYNTH))
    assert proc.returncode == 0, proc.stderr
    text = proc.stdout
    good = text.split('### Bad Curve Scenarios', 1)[0]
    bad = text.split('### Bad Curve Scenarios', 1)[1].split('### Edge Case', 1)[0]
    edge = text.split('### Edge Case', 1)[1]
    for name in GOOD_GROWTH_SCENARIOS:
        assert f'| {name} |' in good, name
    for name in BAD_CURVE_SCENARIOS:
        assert f'| {name} |' in bad, name
    for name, cfg in EDGE_CASE_SCENARIOS.items():
        assert f'| {name} | {cfg.expected_class} | {cfg.n_default} |' in edge, name
    assert f'{len(ALL_SCENARIOS)} scenarios' in text


def test_the_readme_scenario_coverage_is_the_rendered_block():
    """synthetic_data/README.md carries the rendered tables verbatim.

    tests/test_audit_fixes_ml.py::TestScenarioDocs only checks that every
    scenario name appears somewhere in the right table, so the README can
    still drift on counts, ranges and descriptions.  This pins the whole
    block: the text between ``## Scenario Coverage`` and ``## Noise Models``
    must be exactly what ``--print-readme-tables`` prints today (plus the
    short "regenerate me" note), so a catalog edit that is not pasted back
    fails here instead of in a stale document.
    """
    proc = subprocess.run(
        [sys.executable, str(SYNTH / 'scripts' / 'generate_comprehensive.py'),
         '--print-readme-tables'],
        capture_output=True, text=True, cwd=str(SYNTH))
    assert proc.returncode == 0, proc.stderr

    readme = (SYNTH / 'README.md').read_text()
    block = readme.split('## Scenario Coverage', 1)[1].split('## Noise Models', 1)[0]
    rendered = proc.stdout

    assert '--print-readme-tables' in block, (
        'the README must say how its tables are regenerated')
    missing = [line for line in rendered.splitlines()
               if line.strip() and line.strip() not in block]
    assert not missing, (
        'synthetic_data/README.md is stale against the catalog; paste the '
        'output of `python scripts/generate_comprehensive.py '
        '--print-readme-tables` between "## Scenario Coverage" and '
        f'"## Noise Models".  Missing lines: {missing[:5]}')
