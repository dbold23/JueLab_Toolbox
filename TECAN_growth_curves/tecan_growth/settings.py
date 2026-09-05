"""
Analysis settings: one dataclass that mirrors ``scripts/config.yaml`` and the
CLI flags of ``01_growth_curve_analysis.py`` so the interface, the CLI and the
tests all build the same threshold dictionaries.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .legacy import REPO_ROOT

DEFAULT_CONFIG_PATH = REPO_ROOT / "scripts" / "config.yaml"


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


@dataclass
class AnalysisSettings:
    """All knobs that influence a per-curve analysis, with lab-facing docs."""

    # --- fit-quality gates (classification section) ---
    min_r_squared: float = 0.95           # fit must explain >= this fraction of variance
    max_param_error_pct: float = 20.0     # standard error of A and mu, as % of the estimate
    min_points_for_fit: int = 15          # minimum usable data points per curve

    # --- secondary quality gates (quality_gates section) ---
    min_snr: float = 5.0
    min_delta_od_ci: float = 0.1
    min_absolute_delta_od: float = 0.15
    pre_fit_snr_threshold: float = 3.0
    pre_fit_delta_od_override: float = 0.5
    excellent_r2_threshold: float = 0.98
    min_monotone_fraction: float = 0.55
    max_residual_autocorr: float = 0.7

    # --- truncation section ---
    smoothing_window: int = 5
    buffer_points: int = 3
    min_points_after_truncation: int = 20
    use_first_peak: bool = True
    use_adaptive: bool = False            # MCCV truncation (the --adaptive flag of the batch pipeline)
    try_multi_model: bool = False

    # --- ML classifier ---
    use_ml: bool = False                  # only possible when models/*.joblib exist
    prefit_reject_threshold: float = 0.2
    postfit_good_threshold: float = 0.7
    postfit_bad_threshold: float = 0.3

    # --- preprocessing ---
    clip_negative: bool = True
    blank_mode: str = "per_media"

    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------ builders
    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None, **overrides) -> "AnalysisSettings":
        """Build from a config.yaml dict (defaults to the repo's) plus keyword overrides."""
        cfg = load_config() if config is None else config
        c = cfg.get("classification", {}) or {}
        g = cfg.get("quality_gates", {}) or {}
        t = cfg.get("truncation", {}) or {}
        m = cfg.get("ml_classifier", {}) or {}
        kw = dict(
            min_r_squared=c.get("min_r_squared", 0.95),
            max_param_error_pct=c.get("max_param_error_pct", 20.0),
            min_points_for_fit=c.get("min_points_for_fit", 15),
            min_snr=g.get("min_snr", 5.0),
            min_delta_od_ci=g.get("min_delta_od_ci", 0.1),
            min_absolute_delta_od=g.get("min_absolute_delta_od", 0.15),
            pre_fit_snr_threshold=g.get("pre_fit_snr_threshold", 3.0),
            pre_fit_delta_od_override=g.get("pre_fit_delta_od_override", 0.5),
            excellent_r2_threshold=g.get("excellent_r2_threshold", 0.98),
            min_monotone_fraction=g.get("min_monotone_fraction", 0.55),
            max_residual_autocorr=g.get("max_residual_autocorr", 0.7),
            smoothing_window=t.get("smoothing_window", 5),
            buffer_points=t.get("buffer_points", 3),
            min_points_after_truncation=t.get("min_points_for_fit", 20),
            use_first_peak=t.get("use_first_peak", True),
            # config.yaml default (first peak). The batch pipeline passes --adaptive explicitly;
            # the app offers it as an option because it is ~100x slower.
            use_adaptive=t.get("use_adaptive", False),
            try_multi_model=(t.get("mccv", {}) or {}).get("try_multi_model", False),
            use_ml=bool(m.get("enabled", False)),
            prefit_reject_threshold=(m.get("prefit_gate", {}) or {}).get("reject_threshold", 0.2),
            postfit_good_threshold=(m.get("postfit_classifier", {}) or {}).get("good_threshold", 0.7),
            postfit_bad_threshold=(m.get("postfit_classifier", {}) or {}).get("bad_threshold", 0.3),
        )
        kw.update(overrides)
        return cls(**kw)

    # ---------------------------------------------------- legacy dictionaries
    def fit_quality_thresholds(self) -> Dict[str, Any]:
        """Dictionary in the shape ``01_growth_curve_analysis.main()`` builds."""
        return {
            "min_r_squared": self.min_r_squared,
            "max_param_error_pct": self.max_param_error_pct,
            "min_points_for_fit": self.min_points_for_fit,
            "min_snr": self.min_snr,
            "min_delta_od_ci": self.min_delta_od_ci,
            "min_absolute_delta_od": self.min_absolute_delta_od,
            "pre_fit_snr_threshold": self.pre_fit_snr_threshold,
            "pre_fit_delta_od_override": self.pre_fit_delta_od_override,
            "excellent_r2_threshold": self.excellent_r2_threshold,
            "min_monotone_fraction": self.min_monotone_fraction,
            "max_residual_autocorr": self.max_residual_autocorr,
        }

    def truncation_params(self) -> Dict[str, Any]:
        return {
            "smoothing_window": self.smoothing_window,
            "buffer_points": self.buffer_points,
            "min_points_for_fit": self.min_points_after_truncation,
            "use_first_peak": self.use_first_peak,
            "use_adaptive": self.use_adaptive,
            "try_multi_model": self.try_multi_model,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def describe_changes(self) -> Dict[str, Any]:
        """Settings that differ from the repository defaults (for run manifests)."""
        base = AnalysisSettings.from_config()
        return {k: v for k, v in self.to_dict().items() if base.to_dict().get(k) != v and k != "extra"}
