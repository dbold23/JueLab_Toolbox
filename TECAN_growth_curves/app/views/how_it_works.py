"""How it works: the lab manual, with the thresholds of the current run filled in."""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import streamlit as st

import _bootstrap  # noqa: F401
from _bootstrap import DOCS_DIR
from state import S, counts

s = S()
st.title("How the analysis works")

values = defaultdict(lambda: "…")
values.update({k: (f"{v:g}" if isinstance(v, float) else v) for k, v in s["settings"].to_dict().items() if k != "extra"})
values.update({f"n_{k.replace(' ', '_')}": v for k, v in counts().items()})
values["run_name"] = s.get("run_name") or "(no run loaded)"

sections = sorted((DOCS_DIR / "how_it_works").glob("*.md")) if (DOCS_DIR / "how_it_works").exists() else []
if not sections:
    st.info("The explanation pages are in docs/how_it_works/. None were found.")
    st.stop()

st.caption("Numbers in this text are the thresholds of the **current settings**; change a slider and the text follows.")
images = {p.stem: p for p in (DOCS_DIR / "images").glob("*.svg")} if (DOCS_DIR / "images").exists() else {}

toc = [re.sub(r"^\d+_", "", p.stem).replace("_", " ") for p in sections]
choice = st.radio("Section", toc, horizontal=True, key="hiw_section", label_visibility="collapsed")
path = sections[toc.index(choice)]
text = path.read_text(encoding="utf-8")
try:
    text = text.format_map(values)
except (KeyError, ValueError):
    pass
for m in re.finditer(r"!\[[^\]]*\]\(images/([^)]+)\)", text):
    img = DOCS_DIR / "images" / m.group(1)
    if img.exists():
        st.image(str(img))
text = re.sub(r"!\[[^\]]*\]\(images/[^)]+\)", "", text)
st.markdown(text)
