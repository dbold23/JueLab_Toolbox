"""How it works: the lab manual, with the thresholds of the current run filled in."""
from __future__ import annotations

import re

import streamlit as st

import _bootstrap  # noqa: F401
from _bootstrap import DOCS_DIR
from state import S, counts
from tecan_growth.manual import placeholder_values, render_section, section_files, section_title

s = S()
st.title("How the analysis works")

sections = section_files()
if not sections:
    st.info("The explanation pages are in docs/how_it_works/. None were found.")
    st.stop()

values = placeholder_values(s["settings"], counts=counts(), run_name=s.get("run_name"))
st.caption("Numbers in this text are the thresholds of the **current settings**; change a slider and the text follows.")

toc = [section_title(p) for p in sections]
choice = st.radio("Section", toc, horizontal=True, key="hiw_section", label_visibility="collapsed")
path = sections[toc.index(choice)]
try:
    text = render_section(path, values)
except (KeyError, ValueError, IndexError):
    text = path.read_text(encoding="utf-8")
for m in re.finditer(r"!\[[^\]]*\]\(images/([^)]+)\)", text):
    img = DOCS_DIR / "images" / m.group(1)
    if img.exists():
        st.image(str(img))
text = re.sub(r"!\[[^\]]*\]\(images/[^)]+\)", "", text)
st.markdown(text)

idx = toc.index(choice)
prev_col, _, next_col = st.columns([1, 4, 1])
if idx > 0 and prev_col.button(f"← {toc[idx - 1]}", key="hiw_prev"):
    st.session_state["hiw_section"] = toc[idx - 1]
    st.rerun()
if idx < len(toc) - 1 and next_col.button(f"{toc[idx + 1]} →", key="hiw_next"):
    st.session_state["hiw_section"] = toc[idx + 1]
    st.rerun()
