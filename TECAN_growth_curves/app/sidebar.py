"""Sidebar shared by every page: run name, verdict counts, the settings that matter, stale banner."""
from __future__ import annotations

import streamlit as st

import _bootstrap  # noqa: F401
from state import (S, TRUNCATION_CHOICES, apply_setting_changes, counts, is_stale,
                   run_analysis, truncation_choice_label, update_setting)
from tecan_growth import legacy
from tecan_growth.settings import AnalysisSettings


def _on_change(field: str, widget_key: str):
    def _cb():
        update_setting(**{field: st.session_state[widget_key]})
        outcome = apply_setting_changes()
        if outcome == "reclassified":
            st.toast("Verdicts updated with the new threshold", icon=":material/check_circle:")
    return _cb


def _on_truncation_change():
    cfg = TRUNCATION_CHOICES[st.session_state["trunc_choice"]]
    update_setting(**cfg)
    apply_setting_changes()


def _slider(label: str, field: str, lo: float, hi: float, step: float, help_text: str, fmt: str = "%.3f"):
    s = S()
    key = f"set_{field}"
    st.slider(label, lo, hi, float(getattr(s["settings"], field)), step=step, key=key, help=help_text,
              on_change=_on_change(field, key), format=fmt)


def render_sidebar() -> None:
    s = S()
    with st.sidebar:
        st.markdown("### TECAN growth curves")
        if s.get("run_name_pending"):
            st.session_state["run_name_input"] = s.pop("run_name_pending")
        st.text_input("Run name", key="run_name_input", help="Used for the export folder and file names",
                      on_change=lambda: s.__setitem__("run_name", st.session_state["run_name_input"]))
        s["run_name"] = st.session_state.get("run_name_input", s.get("run_name", ""))

        models = legacy.model_files_present()
        mode = "Rules + ML opinion" if (models and s["settings"].use_ml) else "Rules"
        st.caption(f"Classifier: **{mode}**" + ("" if models else "  ·  no ML models on disk"))

        c = counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("GOOD", c["GOOD"])
        m2.metric("BAD", c["BAD"])
        m3.metric("Not fitted", c["not fitted"])

        if is_stale():
            st.warning("Settings changed. Results are stale.", icon=":material/warning:")
            if st.button("Re-run analysis", type="primary", width="stretch", key="rerun_sidebar"):
                with st.spinner("Re-running every curve…"):
                    run_analysis()
                st.rerun()

        st.markdown("#### Verdict thresholds")
        _slider("Minimum R²", "min_r_squared", 0.80, 0.995, 0.005,
                "A fit must explain at least this fraction of the variation in the data to be called GOOD.")
        _slider("Max parameter error (%)", "max_param_error_pct", 5.0, 50.0, 1.0,
                "Standard error of A and of μ, as a percentage of the estimate.", fmt="%.0f")
        _slider("Minimum growth ΔOD", "min_absolute_delta_od", 0.02, 0.50, 0.01,
                "The curve must rise at least this much above its baseline (OD600 units).", fmt="%.2f")

        st.radio("Growth window (truncation)", list(TRUNCATION_CHOICES),
                 index=list(TRUNCATION_CHOICES).index(truncation_choice_label(s["settings"])),
                 key="trunc_choice", on_change=_on_truncation_change,
                 help="Where the fit stops: at the first growth peak (default), at the global maximum, "
                      "or at the cross-validated optimum used by the batch pipeline (slow).")

        with st.expander("Advanced"):
            unlocked = st.checkbox("I understand these change the call of record", key="adv_unlock")
            dis = not unlocked
            st.number_input("Smoothing window (points)", 3, 15, int(s["settings"].smoothing_window), 2, key="set_smoothing_window",
                            on_change=_on_change("smoothing_window", "set_smoothing_window"), disabled=dis)
            st.number_input("Buffer points after peak", 0, 10, int(s["settings"].buffer_points), 1, key="set_buffer_points",
                            on_change=_on_change("buffer_points", "set_buffer_points"), disabled=dis)
            st.number_input("Minimum points after truncation", 5, 60, int(s["settings"].min_points_after_truncation), 1,
                            key="set_min_points_after_truncation",
                            on_change=_on_change("min_points_after_truncation", "set_min_points_after_truncation"), disabled=dis)
            st.slider("Excellent-fit R² (bypasses noise gates)", 0.90, 0.999, float(s["settings"].excellent_r2_threshold), 0.001,
                      key="set_excellent_r2_threshold", format="%.3f",
                      on_change=_on_change("excellent_r2_threshold", "set_excellent_r2_threshold"), disabled=dis)
            st.slider("Minimum signal-to-noise", 1.0, 20.0, float(s["settings"].min_snr), 0.5, key="set_min_snr",
                      on_change=_on_change("min_snr", "set_min_snr"), disabled=dis)
            st.slider("Minimum ΔOD − 2×noise", 0.0, 0.5, float(s["settings"].min_delta_od_ci), 0.01, key="set_min_delta_od_ci",
                      on_change=_on_change("min_delta_od_ci", "set_min_delta_od_ci"), disabled=dis)
            st.slider("Minimum monotone fraction", 0.3, 0.9, float(s["settings"].min_monotone_fraction), 0.05,
                      key="set_min_monotone_fraction",
                      on_change=_on_change("min_monotone_fraction", "set_min_monotone_fraction"), disabled=dis)
            st.slider("Maximum residual autocorrelation", 0.3, 0.99, float(s["settings"].max_residual_autocorr), 0.01,
                      key="set_max_residual_autocorr",
                      on_change=_on_change("max_residual_autocorr", "set_max_residual_autocorr"), disabled=dis)
            st.slider("Pre-fit signal-to-noise threshold", 0.5, 10.0, float(s["settings"].pre_fit_snr_threshold), 0.5,
                      key="set_pre_fit_snr_threshold",
                      on_change=_on_change("pre_fit_snr_threshold", "set_pre_fit_snr_threshold"), disabled=dis)
            st.checkbox("Clip negative blank-subtracted values to 0 (legacy)", value=bool(s["settings"].clip_negative),
                        key="set_clip_negative", on_change=_on_change("clip_negative", "set_clip_negative"), disabled=dis,
                        help="The original pipeline floors blanked OD at zero. Turning this off keeps the real baseline noise.")
            st.selectbox("Blank subtraction", ["per_media", "global", "none"],
                         index=["per_media", "global", "none"].index(s["settings"].blank_mode), key="set_blank_mode",
                         on_change=_on_change("blank_mode", "set_blank_mode"), disabled=dis)
            st.checkbox("ML second opinion (needs models/*.joblib)", value=bool(s["settings"].use_ml and models),
                        key="set_use_ml", on_change=_on_change("use_ml", "set_use_ml"), disabled=dis or not models,
                        help="Adds a P(good) column from the trained classifier. Rules stay the call of record.")
            if st.button("Reset to defaults", key="reset_defaults", disabled=dis):
                s["settings"] = AnalysisSettings.interactive_defaults()
                for k in list(st.session_state.keys()):
                    if k.startswith("set_") or k == "trunc_choice":
                        del st.session_state[k]
                apply_setting_changes()
                st.rerun()

        if s.get("flips"):
            with st.expander(f"{len(s['flips'])} verdict(s) changed", expanded=True):
                for cid, a, b, gate in s["flips"][:12]:
                    st.caption(f"{cid}: {a} → {b}" + (f" ({gate})" if gate else ""))
