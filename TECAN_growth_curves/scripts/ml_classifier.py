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

METADATA_FEATURES = [
    'is_control',           # 1 if H2O/LB control, 0 if treatment, NaN if unknown
    'concentration_numeric', # parsed numeric concentration from strain name
]

# Genomic features (optional — NaN when genomic data unavailable)
# These are added to the postfit feature set when a model is trained with them.
# HistGradientBoosting handles NaN natively, so backward compatibility is maintained.
GENOMIC_FEATURES = [
    'n_degradation_genes',
    'has_carboxylesterase', 'has_opd_mpd', 'has_pyrethroid_hydrolase',
    'has_cytochrome_p450', 'has_nitroreductase',
    'max_pident_carboxylesterase', 'max_pident_opd_mpd',
    'pesticide_gene_relevance_score',
]

PREFIT_FEATURES = [
    'raw_delta_od', 'raw_max_od', 'raw_snr', 'raw_monotone_fraction',
    'raw_baseline_std', 'raw_baseline_mean', 'n_points', 'time_span',
] + METADATA_FEATURES

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
    + METADATA_FEATURES
)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def extract_metadata_features(strain_name: Optional[str]) -> Dict[str, float]:
    """Extract biological metadata from strain name.

    Parses naming conventions like 'H2O-MAL10', 'LB-CISPERM1', 'Bifenthrin-BIF3'.
    Returns is_control (1/0) and numeric concentration.
    """
    if not strain_name:
        return {'is_control': float('nan'), 'concentration_numeric': float('nan')}

    name = str(strain_name).strip().upper()

    # Detect controls: H2O or LB prefix (before the dash)
    prefix = name.split('-')[0] if '-' in name else name
    is_control = 1.0 if prefix in ('H2O', 'LB') else 0.0

    # Extract trailing number as concentration
    m = re.search(r'(\d+)\s*$', name)
    concentration = float(m.group(1)) if m else float('nan')

    return {'is_control': is_control, 'concentration_numeric': concentration}


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


def extract_genomic_features_for_classifier(
    strain_name: Optional[str],
    genomic_df: Optional['pd.DataFrame'] = None,
) -> Dict[str, float]:
    """
    Look up pre-computed genomic features for a strain.

    Args:
        strain_name: Full pipeline strain name (e.g., 'BifenthrinANDLB-BIF2')
        genomic_df: DataFrame indexed by biological strain ID with genomic features.
            If None, returns NaN for all genomic features.

    Returns:
        Dict of genomic feature name -> value (float or NaN)
    """
    nan_features = {f: float('nan') for f in GENOMIC_FEATURES}

    if genomic_df is None or strain_name is None:
        return nan_features

    try:
        from genomic_features import resolve_strain_id
        bio_id = resolve_strain_id(strain_name)
    except ImportError:
        return nan_features

    if bio_id not in genomic_df.index:
        return nan_features

    row = genomic_df.loc[bio_id]
    features = {}
    for feat in GENOMIC_FEATURES:
        val = row.get(feat, float('nan'))
        features[feat] = float(val) if not isinstance(val, float) else val

    return features


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
    ):
        self.reject_threshold = reject_threshold
        self.model = None

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

    @staticmethod
    def _default_path() -> Path:
        return Path(__file__).parent.parent / 'models' / 'prefit_gate.joblib'

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
        X = np.array([[features.get(f, float('nan')) for f in PREFIT_FEATURES]])

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

    def __init__(
        self,
        model_path: Optional[str] = None,
        feature_config_path: Optional[str] = None,
        good_threshold: float = 0.7,
        bad_threshold: float = 0.3,
    ):
        self.good_threshold = good_threshold
        self.bad_threshold = bad_threshold
        self.model = None
        self.feature_names = ALL_POSTFIT_FEATURES
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
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = json.load(f)
            self.feature_names = cfg.get('postfit_feature_names', self.feature_names)

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
