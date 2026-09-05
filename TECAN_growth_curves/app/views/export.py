"""Export: everything needed for a report, and to reproduce the run."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import streamlit as st

import _bootstrap  # noqa: F401
from _bootstrap import RUNS_DIR
from state import S, get_processed
from tecan_growth import export

s = S()
st.title("Export")
results = s.get("results")
if results is None:
    st.info("Run an analysis on the Home page first.")
    st.page_link("Home.py", label="Home", icon=":material/home:")
    st.stop()
run, key, processed = s["run"], s["key"], get_processed()
overrides = s["overrides"]
name = (s.get("run_name") or "run").strip().replace("/", "_") or "run"

st.markdown("Every table below comes from the analysis you see in the app. The run folder also carries a "
            "**manifest** (software versions, input file hash, every setting, every override) and the processed "
            "curves in the format the command-line pipeline reads, so anyone can reproduce the numbers.")

c1, c2, c3 = st.columns(3)
c1.download_button("Results table (CSV)", export.results_table(results, overrides).to_csv(index=False).encode(),
                   file_name=f"{name}_results.csv", mime="text/csv", width="stretch", key="dl_results")
c2.download_button("Clean summary, GOOD only (CSV)", export.clean_summary(results, overrides).to_csv(index=False).encode(),
                   file_name=f"{name}_summary.csv", mime="text/csv", width="stretch", key="dl_clean")
c3.download_button("Curves, long format (CSV)", export.curves_long(processed, results).to_csv(index=False).encode(),
                   file_name=f"{name}_curves.csv", mime="text/csv", width="stretch", key="dl_curves")


@st.cache_data(show_spinner=False, max_entries=4)
def _bundle(run_name: str, input_sha: str, settings_key: str, overrides_json: str) -> bytes:
    tmp = Path(tempfile.mkdtemp())
    export.write_run_folder(tmp / run_name, s["run"], s["key"], get_processed(), s["results"],
                            overrides=s["overrides"], input_sha256=input_sha)
    data = export.bundle_zip(tmp / run_name)
    shutil.rmtree(tmp, ignore_errors=True)
    return data


b1, b2 = st.columns(2)
with b1:
    if st.button("Build the full bundle (zip)", width="stretch", key="build_bundle"):
        with st.spinner("Writing tables, workbook and manifest…"):
            s["bundle"] = _bundle(name, s.get("input_sha256", ""), s.get("run_settings_hash", ""),
                                  json.dumps(overrides, sort_keys=True, default=str))
    if s.get("bundle"):
        st.download_button("Download bundle", s["bundle"], file_name=f"{name}.zip", mime="application/zip",
                           width="stretch", key="dl_bundle")
with b2:
    if st.button("Save a run folder next to the repository (runs/)", width="stretch", key="save_run"):
        out = RUNS_DIR / name
        with st.spinner(f"Writing {out}…"):
            files = export.write_run_folder(out, run, key, processed, results, overrides=overrides,
                                            input_sha256=s.get("input_sha256"))
        st.success(f"Wrote {len(files)} files to {out}")

st.subheader("Methods paragraph")
st.code(export.methods_paragraph(s["run_settings"] if "run_settings" in s else s["settings"], results), language=None)

with st.expander("Run manifest (JSON)"):
    manifest = export.run_manifest(run, key, processed, results, overrides=overrides, input_sha256=s.get("input_sha256"))
    st.json(manifest, expanded=False)
