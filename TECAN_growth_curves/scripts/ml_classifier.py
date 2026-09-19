"""
ML-based growth curve classifier.

Two-stage classification:
  1. PreFitGate: rejects obvious junk before expensive truncation/fitting
  2. PostFitClassifier: three-class (GOOD/BORDERLINE/BAD) after Gompertz fit

Falls back to rule-based classification if model files are missing.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------

#: Strain-name metadata. NOT model features -- see the note below. They are
#: still extracted (and written into the feature dicts) because they are useful
#: for stratifying a real-plate report by control/treatment, and because the
#: KEY/plate path may one day supply them honestly.
#:
#: They were members of PREFIT_FEATURES and ALL_POSTFIT_FEATURES until the
#: synthetic-suite audit (M27). Every synthetic training column is named
#: ``SYNTHETIC_400pts_99h_CURVE0330_blanked``, which
#: :func:`extract_metadata_features` deliberately refuses to parse, so both
#: features were NaN for all 480 training rows; the trainer's
#: ``clean_feature_matrix`` then froze them to the constant 0.0. The shipped
#: models carry ``_bin_mapper.n_bins_non_missing_ == 1`` for both, and
#: ``predict_proba`` is bit-identical (max |dp| = 0.0 over 200 random rows)
#: whether they are 0/0, 1/10 or NaN -- so they are provably inert in training
#: while taking real values (is_control in {0,1}, concentration 1..10) at
#: inference: training/serving skew with no upside.
#:
#: They are also unsupported on real plates: over the 84 decided curves of
#: results/tables/validation_audit.csv, is_control has ROC-AUC 0.45 and mutual
#: information 0.0046 with the audited label, and concentration_numeric has
#: AUC 0.524. Restore them only if a metadata-only model clears AUC >= 0.55 on
#: that external test, and then only via the KEY/plate path where media comes
#: from the KEY file -- never from the column name, which is the exact
#: mechanism that produced the earlier curve-id leak.
METADATA_FEATURES = [
    'is_control',           # 1 if H2O/LB control, 0 if treatment, NaN if unknown
    'concentration_numeric', # parsed numeric concentration from strain name
]

PREFIT_FEATURES = [
    'raw_delta_od', 'raw_max_od', 'raw_snr', 'raw_monotone_fraction',
    'raw_baseline_std', 'raw_baseline_mean', 'n_points', 'time_span',
]

POSTFIT_DIRECT_FEATURES = [
    'fit_r_squared', 'fit_rmse', 'fit_mae',
    'a_err_pct', 'mu_err_pct',
    'snr', 'delta_od', 'max_od', 'points_used',
    'gompertz_a', 'gompertz_mu', 'gompertz_lambda',
    'truncation_time',
]

POSTFIT_SECONDARY_FEATURES = [
    'baseline_std', 'monotone_fraction',
    'residual_autocorr', 'delta_od_ci_lower',
]

POSTFIT_DERIVED_FEATURES = [
    'mu_over_a', 'lambda_over_trunc_time', 'rmse_over_delta_od',
    'err_product', 'points_per_hour',
]

ALL_POSTFIT_FEATURES = (
    POSTFIT_DIRECT_FEATURES + POSTFIT_SECONDARY_FEATURES + POSTFIT_DERIVED_FEATURES
)


# ---------------------------------------------------------------------------
# Truncation mode provenance
# ---------------------------------------------------------------------------

#: Truncation modes a run can resolve to (the vocabulary of
#: run_settings.json's ``truncation_mode``).
TRUNCATION_MODES = ('adaptive', 'first_peak', 'global_max')


def check_truncation_mode(trained_mode, deployed_mode, source='post-fit classifier'):
    """Warn when a model is being used under a different truncation mode.

    A model trained on first-peak features sees a different ``truncation_time``,
    ``points_used`` and fit-quality distribution than an adaptive run produces,
    so the mode is part of the model's contract. Returns the warning message
    (also logged) or None when the modes agree or either is unknown.
    """
    if not trained_mode or not deployed_mode:
        return None
    if str(trained_mode) == str(deployed_mode):
        return None
    msg = (
        f"{source}: truncation mode mismatch -- the model was trained on "
        f"features built with truncation mode '{trained_mode}' but this run "
        f"resolves to '{deployed_mode}'. Fit-quality features are not "
        f"comparable across modes; retrain with --truncation-mode "
        f"{deployed_mode} or run the pipeline in '{trained_mode}' mode."
    )
    logger.warning(msg)
    return msg


def _load_feature_config(path):
    """Read a models/feature_config.json, or {} when it is absent/unreadable."""
    try:
        with open(path) as fh:
            cfg = json.load(fh)
    except (OSError, ValueError):
        return {}
    return cfg if isinstance(cfg, dict) else {}


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

# Media tokens used by the lab naming convention 'MEDIA-STRAINdigits'
# (e.g. 'H2O-MAL10', 'MALATHIONANDLB-MAL4'). Anything else — synthetic curve
# ids, plate labels, free-form names — yields NaN metadata rather than a
# number that happens to be parseable.
CONTROL_MEDIA_TOKENS = frozenset({'H2O', 'LB'})

TREATMENT_MEDIA_TOKENS = frozenset({
    'BIFENTHRIN', 'CISPERMETHRIN', 'CYPERMETHRIN', 'FLUPYRADIFURONE',
    'IMIDACLOPRID', 'LAMBDACYHALOTHRIN', 'MALATHION', 'PERMETHRIN',
})

# Pesticide-in-LB media are written '<PESTICIDE>ANDLB' and are treatments,
# not controls.
KNOWN_MEDIA_TOKENS = frozenset(
    CONTROL_MEDIA_TOKENS
    | TREATMENT_MEDIA_TOKENS
    | {f'{t}ANDLB' for t in TREATMENT_MEDIA_TOKENS}
)

# Strain part of the lab convention: letters then at most 3 digits of
# concentration (the real names top out at 2, e.g. 'MAL10', 'LCY01').
_STRAIN_CONCENTRATION_RE = re.compile(r'^[A-Z]+(\d{1,3})$')


def extract_metadata_features(strain_name: Optional[str]) -> Dict[str, float]:
    """Extract biological metadata from a strain name.

    Only the lab naming convention 'MEDIA-STRAINdigits' is parsed, where MEDIA
    is a known media token (see KNOWN_MEDIA_TOKENS): 'H2O-MAL10' -> control,
    concentration 10; 'MALATHIONANDLB-MAL4' -> treatment, concentration 4.

    Names that do not follow the convention return NaN for both features. This
    matters for synthetic training curves ('CURVE0330'): parsing their trailing
    digits would hand the model the curve id, which encodes the scenario and
    therefore the label.

    Returns is_control (1/0/NaN) and concentration_numeric (float/NaN).
    """
    nan = float('nan')
    features = {'is_control': nan, 'concentration_numeric': nan}

    if not strain_name:
        return features

    name = str(strain_name).strip().upper()

    # Exactly one dash separates media from strain in the lab convention
    if name.count('-') != 1:
        return features

    media, strain = name.split('-')
    if media not in KNOWN_MEDIA_TOKENS:
        return features

    features['is_control'] = 1.0 if media in CONTROL_MEDIA_TOKENS else 0.0

    m = _STRAIN_CONCENTRATION_RE.match(strain)
    if m:
        features['concentration_numeric'] = float(m.group(1))

    return features


def extract_prefit_features(
    time: np.ndarray,
    od600: np.ndarray,
    strain_name: Optional[str] = None,
) -> Dict[str, float]:
    """Extract features from raw OD data before fitting."""
    import pandas as pd

    n_baseline = min(10, len(od600) // 5, len(od600))
    baseline = od600[:n_baseline]
    baseline_mean = float(np.mean(baseline))
    baseline_std = float(np.std(baseline)) if float(np.std(baseline)) > 1e-8 else 1e-8
    max_od = float(np.max(od600))
    delta_od = max_od - baseline_mean
    snr = delta_od / baseline_std

    # Monotonicity of smoothed signal (first 60%)
    smoothed = pd.Series(od600).rolling(window=5, center=True, min_periods=1).mean().values
    check_end = max(10, int(len(smoothed) * 0.6))
    diffs = np.diff(smoothed[:check_end])
    monotone_frac = float(np.sum(diffs > 0) / max(len(diffs), 1))

    features = {
        'raw_delta_od': delta_od,
        'raw_max_od': max_od,
        'raw_snr': snr,
        'raw_monotone_fraction': monotone_frac,
        'raw_baseline_std': baseline_std,
        'raw_baseline_mean': baseline_mean,
        'n_points': len(od600),
        'time_span': float(time[-1] - time[0]) if len(time) > 1 else 0.0,
    }
    features.update(extract_metadata_features(strain_name))
    return features


def extract_postfit_features(
    fit_result,
    classification_metrics: Dict,
    truncation_time: Optional[float] = None,
    points_used: Optional[int] = None,
) -> Dict[str, float]:
    """Extract ML feature vector from fit results and classification metrics."""
    features = {}

    # Direct features from fit result
    if fit_result and fit_result.success:
        features['fit_r_squared'] = fit_result.r_squared
        features['fit_rmse'] = fit_result.rmse
        features['fit_mae'] = fit_result.mae
        features['gompertz_a'] = fit_result.a_opt
        features['gompertz_mu'] = fit_result.mu_opt
        features['gompertz_lambda'] = fit_result.lambda_opt
    else:
        for k in ['fit_r_squared', 'fit_rmse', 'fit_mae',
                   'gompertz_a', 'gompertz_mu', 'gompertz_lambda']:
            features[k] = float('nan')

    # From classification metrics
    for key in ['a_err_pct', 'mu_err_pct', 'snr', 'delta_od', 'max_od',
                'baseline_std', 'monotone_fraction', 'residual_autocorr',
                'delta_od_ci_lower']:
        features[key] = classification_metrics.get(key, float('nan'))

    features['truncation_time'] = truncation_time if truncation_time is not None else float('nan')
    features['points_used'] = float(points_used) if points_used is not None else float('nan')

    # Derived ratio features
    features.update(compute_derived_features(features))

    return features


def compute_derived_features(features: Dict[str, float]) -> Dict[str, float]:
    """Compute ratio/interaction features from base features."""
    def safe_div(a, b):
        if b is None or b == 0 or np.isnan(b) or a is None or np.isnan(a):
            return float('nan')
        return a / b

    a = features.get('gompertz_a', float('nan'))
    mu = features.get('gompertz_mu', float('nan'))
    lam = features.get('gompertz_lambda', float('nan'))
    trunc = features.get('truncation_time', float('nan'))
    rmse = features.get('fit_rmse', float('nan'))
    delta = features.get('delta_od', float('nan'))
    a_err = features.get('a_err_pct', float('nan'))
    mu_err = features.get('mu_err_pct', float('nan'))
    pts = features.get('points_used', float('nan'))

    return {
        'mu_over_a': safe_div(mu, a),
        'lambda_over_trunc_time': safe_div(lam, trunc),
        'rmse_over_delta_od': safe_div(rmse, delta),
        'err_product': (a_err * mu_err) if not (np.isnan(a_err) or np.isnan(mu_err)) else float('nan'),
        'points_per_hour': safe_div(pts, trunc),
    }


# ---------------------------------------------------------------------------
# Pre-fit gate
# ---------------------------------------------------------------------------

class PreFitGate:
    """
    Rejects obvious junk before expensive truncation/fitting.

    Tuned for HIGH RECALL — only rejects curves that are clearly bad.
    Better to fit a few extra bad curves than miss a good one.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        reject_threshold: float = 0.2,
        feature_config_path: Optional[str] = None,
        truncation_mode: Optional[str] = None,
    ):
        self.reject_threshold = reject_threshold
        self.model = None
        #: Feature order the *loaded* model was trained with. Read from
        #: feature_config.json when present so a model shipped with an older
        #: feature list keeps working after the list changes in code.
        self.feature_names = list(PREFIT_FEATURES)
        self.trained_truncation_mode = None
        self.truncation_mode_warning = None

        self._good_class_idx = 1  # Default: class 1 = GOOD

        path = Path(model_path) if model_path else self._default_path()
        if path.exists():
            import joblib
            self.model = joblib.load(path)
            # Verify class ordering — find the index for class 1 (GOOD)
            if hasattr(self.model, 'classes_'):
                self._good_class_idx = list(self.model.classes_).index(1)
            logger.info(f"Loaded pre-fit gate from {path}")
        else:
            logger.warning(f"Pre-fit gate model not found at {path}, gate disabled")

        cfg_path = Path(feature_config_path) if feature_config_path else self._default_config_path()
        cfg = _load_feature_config(cfg_path)
        names = cfg.get('prefit_feature_names')
        if names:
            self.feature_names = list(names)
            if list(names) != list(PREFIT_FEATURES):
                logger.warning(
                    "Pre-fit gate was trained on a different feature list than "
                    "this code defines (model: %s; code: %s). Using the model's "
                    "list; retrain to bring them back in step.",
                    list(names), list(PREFIT_FEATURES),
                )
        self.trained_truncation_mode = cfg.get('truncation_mode')
        self.truncation_mode_warning = check_truncation_mode(
            self.trained_truncation_mode, truncation_mode, source='pre-fit gate'
        )

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).parent.parent / 'models' / 'prefit_gate.joblib'

    @staticmethod
    def _default_config_path() -> Path:
        return Path(__file__).parent.parent / 'models' / 'feature_config.json'

    def should_skip(
        self,
        time: np.ndarray,
        od600: np.ndarray,
        strain_name: Optional[str] = None,
    ) -> bool:
        """Return True if this curve should be skipped (obvious junk)."""
        if self.model is None:
            return False

        features = extract_prefit_features(time, od600, strain_name=strain_name)
        X = np.array([[features.get(f, float('nan')) for f in self.feature_names]])

        p_good = self.model.predict_proba(X)[0, self._good_class_idx]
        return p_good <= self.reject_threshold


# ---------------------------------------------------------------------------
# Post-fit classifier
# ---------------------------------------------------------------------------

class PostFitClassifier:
    """
    Three-class classifier: GOOD / BORDERLINE / BAD.

    Uses P(good) probability with two thresholds:
      P >= good_threshold → GOOD
      P <= bad_threshold  → BAD
      otherwise           → BORDERLINE (needs manual review)
    """

    #: The DEPLOYED decision gate. These are policy, not a model artefact:
    #: changing them changes which curves a run calls GOOD, so they move only
    #: by an explicit config/caller value, never because a retrain wrote a
    #: different number into feature_config.json. A model's own
    #: ``recommended_thresholds`` are read and reported (see
    #: :attr:`recommended_thresholds`) but are advisory only.
    DEFAULT_GOOD_THRESHOLD = 0.7
    DEFAULT_BAD_THRESHOLD = 0.3

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        good_threshold: Optional[float] = DEFAULT_GOOD_THRESHOLD,
        bad_threshold: Optional[float] = DEFAULT_BAD_THRESHOLD,
        truncation_mode: Optional[str] = None,
    ):
        self.model = None
        self.feature_names = list(ALL_POSTFIT_FEATURES)
        self.trained_truncation_mode = None
        self.truncation_mode_warning = None
        #: Thresholds the trainer re-derived on a lab-weighted validation
        #: split, or None when the model predates the sweep.
        self.recommended_thresholds = None
        self._good_class_idx = 1  # Default: class 1 = GOOD

        path = Path(model_path) if model_path else self._default_model_path()
        if path.exists():
            import joblib
            self.model = joblib.load(path)
            # Verify class ordering — find the index for class 1 (GOOD)
            if hasattr(self.model, 'classes_'):
                self._good_class_idx = list(self.model.classes_).index(1)
            logger.info(f"Loaded post-fit classifier from {path}")

        # Load feature config if available
        cfg_path = Path(feature_config_path) if feature_config_path else self._default_config_path()
        cfg = _load_feature_config(cfg_path)
        self.feature_names = list(cfg.get('postfit_feature_names', self.feature_names))

        # Thresholds. The active gate is the caller's value, and nothing else:
        # a model card's recommendation is recorded and logged for the operator
        # but never applied, so a retrain cannot silently move the deployed
        # GOOD gate (e.g. 0.70 -> 0.15) without a policy decision.
        rec = cfg.get('recommended_thresholds') or {}
        if rec.get('good_threshold') is not None and rec.get('bad_threshold') is not None:
            self.recommended_thresholds = {
                'good_threshold': float(rec['good_threshold']),
                'bad_threshold': float(rec['bad_threshold']),
            }
        self.good_threshold = self._resolve_threshold(
            good_threshold, 'good_threshold', self.DEFAULT_GOOD_THRESHOLD
        )
        self.bad_threshold = self._resolve_threshold(
            bad_threshold, 'bad_threshold', self.DEFAULT_BAD_THRESHOLD
        )

        # Truncation mode: the fit-quality features a model was trained on are
        # mode-specific, so using it under another mode is a provenance error.
        self.trained_truncation_mode = cfg.get('truncation_mode')
        self.truncation_mode_warning = check_truncation_mode(
            self.trained_truncation_mode, truncation_mode,
            source='post-fit classifier',
        )

    def _resolve_threshold(self, supplied, key, fallback):
        """Resolve one decision threshold.

        The caller's value wins; ``None`` (a caller or config field that never
        got filled in) falls back to the deployed default, NOT to the model's
        recommendation. A model that recommends something else is reported, so
        an operator can decide to adopt it, but adopting it is a config change.
        """
        rec = (self.recommended_thresholds or {}).get(key)
        if supplied is None:
            active = float(fallback)
            logger.warning(
                "Post-fit %s was not supplied; using the deployed default "
                "%.3f.", key, active,
            )
        else:
            active = float(supplied)
        if rec is not None and abs(active - float(rec)) > 1e-9:
            logger.info(
                "Post-fit %s: using %.3f; the loaded model recommends %.3f "
                "(recommended_thresholds in feature_config.json). The "
                "recommendation is advisory - set it in config.yaml to adopt "
                "it.", key, active, float(rec),
            )
        return active

    @staticmethod
    def _default_model_path() -> Path:
        return Path(__file__).parent.parent / 'models' / 'postfit_classifier.joblib'

    @staticmethod
    def _default_config_path() -> Path:
        return Path(__file__).parent.parent / 'models' / 'feature_config.json'

    def classify(
        self,
        fit_result,
        classification_metrics: Dict,
        truncation_time: Optional[float] = None,
        points_used: Optional[int] = None,
        strain_name: Optional[str] = None,
    ) -> Dict:
        """
        Classify a curve. Returns dict with p_good, ml_classification, is_good.

        If model is not loaded, returns None (caller should fall back to rules).
        """
        if self.model is None:
            return None

        features = extract_postfit_features(
            fit_result, classification_metrics, truncation_time, points_used
        )
        # Add metadata features from strain name
        features.update(extract_metadata_features(strain_name))

        X = np.array([[features.get(f, float('nan')) for f in self.feature_names]])

        p_good = float(self.model.predict_proba(X)[0, self._good_class_idx])

        if p_good >= self.good_threshold:
            ml_class = 'GOOD'
            is_good = True
        elif p_good <= self.bad_threshold:
            ml_class = 'BAD'
            is_good = False
        else:
            ml_class = 'BORDERLINE'
            is_good = False  # Conservative: borderline treated as not-good for pipeline

        return {
            'p_good': p_good,
            'ml_classification': ml_class,
            'is_good': is_good,
            'reason': f"{ml_class} (ML): P(good)={p_good:.3f}",
        }
