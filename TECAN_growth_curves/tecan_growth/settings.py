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

    def to_config(self, base: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        These settings in the shape of ``scripts/config.yaml``, so a run can be
        reproduced from the command line with ``--config``. Sections the
        settings do not cover (paths, groups, haldane, advanced, the model
        file locations under ``ml_classifier``) are copied from ``base``
        (default: the repository's config.yaml).
        """
        import copy
        cfg = copy.deepcopy(base if base is not None else load_config())
        cfg["classification"] = {**(cfg.get("classification") or {}),
                                 "min_r_squared": self.min_r_squared,
                                 "max_param_error_pct": self.max_param_error_pct,
                                 "min_points_for_fit": self.min_points_for_fit}
        cfg["quality_gates"] = {**(cfg.get("quality_gates") or {}),
                                "min_snr": self.min_snr,
                                "min_delta_od_ci": self.min_delta_od_ci,
                                "min_absolute_delta_od": self.min_absolute_delta_od,
                                "pre_fit_snr_threshold": self.pre_fit_snr_threshold,
                                "pre_fit_delta_od_override": self.pre_fit_delta_od_override,
                                "excellent_r2_threshold": self.excellent_r2_threshold,
                                "min_monotone_fraction": self.min_monotone_fraction,
                                "max_residual_autocorr": self.max_residual_autocorr}
        cfg["truncation"] = {**(cfg.get("truncation") or {}),
                             "smoothing_window": self.smoothing_window,
                             "buffer_points": self.buffer_points,
                             "min_points_for_fit": self.min_points_after_truncation,
                             "use_first_peak": self.use_first_peak,
                             "use_adaptive": self.use_adaptive}
        trunc_mccv = dict((cfg["truncation"].get("mccv") or {}))
        trunc_mccv["try_multi_model"] = self.try_multi_model
        cfg["truncation"]["mccv"] = trunc_mccv
        ml = dict(cfg.get("ml_classifier") or {})
        ml["enabled"] = self.use_ml
        ml["prefit_gate"] = {**(ml.get("prefit_gate") or {}), "reject_threshold": self.prefit_reject_threshold}
        ml["postfit_classifier"] = {**(ml.get("postfit_classifier") or {}),
                                    "good_threshold": self.postfit_good_threshold,
                                    "bad_threshold": self.postfit_bad_threshold}
        cfg["ml_classifier"] = ml
        cfg["preprocessing"] = {"clip_negative": self.clip_negative, "blank_mode": self.blank_mode}
        return cfg

    def config_roundtrip(self) -> "AnalysisSettings":
        """``from_config(self.to_config())``: what the CLI would read back."""
        return AnalysisSettings.from_config(self.to_config(), clip_negative=self.clip_negative,
                                            blank_mode=self.blank_mode)

    @classmethod
    def interactive_defaults(cls, **overrides) -> "AnalysisSettings":
        """
        What the lab interface starts with: config.yaml's thresholds with the
        fast first-peak window rule. config.yaml may ask the batch pipeline for
        the cross-validated (adaptive) rule, which costs one to three minutes
        per plate and is offered in the interface as an option instead.
        """
        overrides.setdefault("use_adaptive", False)
        return cls.from_config(**overrides)

    def describe_changes(self, reference: Optional["AnalysisSettings"] = None) -> Dict[str, Any]:
        """
        Settings that differ from ``reference`` (default: the interactive
        defaults), for run manifests and the methods paragraph.
        """
        base = (reference or AnalysisSettings.interactive_defaults()).to_dict()
        return {k: v for k, v in self.to_dict().items() if base.get(k) != v and k != "extra"}
