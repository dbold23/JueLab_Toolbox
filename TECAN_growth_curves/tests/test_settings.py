"""AnalysisSettings: config round trip, interactive defaults, change description."""
from tecan_growth.settings import AnalysisSettings, load_config


def test_interactive_defaults_use_the_fast_rule_whatever_config_says():
    s = AnalysisSettings.interactive_defaults()
    assert s.use_adaptive is False
    assert s.use_first_peak is True
    # every other threshold still comes from config.yaml
    cfg = load_config()
    assert s.min_r_squared == cfg["classification"]["min_r_squared"]
    assert AnalysisSettings.interactive_defaults(use_adaptive=True).use_adaptive is True


def test_to_config_round_trips_every_field():
    s = AnalysisSettings.interactive_defaults(min_r_squared=0.9, max_param_error_pct=33.0, use_first_peak=False,
                                              min_monotone_fraction=0.4, use_ml=False, try_multi_model=True)
    back = AnalysisSettings.from_config(s.to_config())
    for k, v in s.to_dict().items():
        if k in ("extra", "clip_negative", "blank_mode"):
            continue
        assert back.to_dict()[k] == v, k
    assert s.config_roundtrip().to_dict() == s.to_dict()


def test_to_config_keeps_the_sections_it_does_not_own():
    cfg = AnalysisSettings.interactive_defaults().to_config()
    for section in ("paths", "groups", "haldane"):
        assert section in cfg
    assert "feature_config_path" in cfg["ml_classifier"]


def test_describe_changes_is_relative_to_the_interactive_defaults():
    assert AnalysisSettings.interactive_defaults().describe_changes() == {}
    assert AnalysisSettings.interactive_defaults(min_r_squared=0.9).describe_changes() == {"min_r_squared": 0.9}
    batch = AnalysisSettings.from_config()
    changes = batch.describe_changes()
    assert changes == ({"use_adaptive": True} if batch.use_adaptive else {})
    assert batch.describe_changes(reference=batch) == {}
