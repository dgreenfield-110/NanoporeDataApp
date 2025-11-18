# Home.py
from __future__ import annotations

import streamlit as st
from pathlib import Path
from lib.config import RUNS_DIR, DB_PATH
from lib.data_io import init_catalog, count_experiments

st.set_page_config(page_title="Nanopore App", page_icon="🧬", layout="wide")
st.title("🧬 Nanopore App — Offline Workflow")

init_catalog()  # ensure DB exists

c1, c2 = st.columns([2, 1])
with c1:
    st.markdown(
        "Use **Upload & Ingest** to add results you've computed offline "
        "(trace + events). Then explore and review events in the other pages."
    )
    # ✅ fix: use a triple-quoted f-string (or \n) instead of a raw newline in quotes
    st.markdown(
        f"""**Runs directory:** `{RUNS_DIR}`  
**Catalog:** `{DB_PATH}`"""
    )

with c2:
    st.metric("Experiments in catalog", f"{count_experiments():,}")

st.divider()

# Core pages (always present)
st.page_link("pages/1_Upload_and_Ingest.py", label="Upload & Ingest (offline)", icon="📤")
st.page_link("pages/2_Experiment_Explorer.py", label="Experiment Explorer", icon="🔎")

# Comparison/Reviewer may be numbered differently; link whichever exists
if Path("pages/3_Experiment_Comparison.py").exists():
    st.page_link("pages/3_Experiment_Comparison.py", label="Experiment Comparison", icon="🧪")
if Path("pages/4_Experiment_Comparison.py").exists():
    st.page_link("pages/4_Experiment_Comparison.py", label="Experiment Comparison", icon="🧪")

if Path("pages/3_Event_Reviewer.py").exists():
    st.page_link("pages/3_Event_Reviewer.py", label="Event Reviewer", icon="🖼️")
if Path("pages/4_Event_Reviewer.py").exists():
    st.page_link("pages/4_Event_Reviewer.py", label="Event Reviewer", icon="🖼️")

st.caption("Tip: change paths in `lib/config.py`.")


with st.sidebar:
    with st.expander("⚠️ Admin"):
        if st.button("Reset catalog (DB only)"):
            from lib.data_io import reset_catalog
            # Call reset_catalog() compatibly with old/new signatures
            try:
                reset_catalog(hard=False)   # works if the function supports it
            except TypeError:
                reset_catalog()             # fallback for older versions
            
