from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from lib.data_io import get_experiment
from lib.viz import ensure_event_times, event_window, event_metrics

st.set_page_config(page_title="Event Reviewer", page_icon="🖼️", layout="wide")
st.title("🖼️ Event Reviewer")

# ---------- helpers ----------
def _fast_event_count(parq_path: str | Path | None) -> int:
    try:
        if parq_path is None:
            return 0
        p = Path(parq_path)
        if not p.exists():
            return 0
        try:
            import pyarrow.parquet as pq  # type: ignore
            return int(pq.ParquetFile(str(p)).metadata.num_rows)
        except Exception:
            return int(len(pd.read_parquet(p)))
    except Exception:
        return 0

def _ensure_time_current_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Time[s]" not in out.columns:
        for cand in ("t", "Time (s)"):
            if cand in out.columns:
                out = out.rename(columns={cand: "Time[s]"})
                break
        else:
            out = out.rename(columns={out.columns[0]: "Time[s]"})
    if ("current_pA" not in out.columns) and ("I_filt" not in out.columns):
        for cand in ("current_pA", "I_filt", "current", "I", "current_nA"):
            if cand in out.columns:
                out = out.rename(columns={cand: "current_pA"})
                break
    return out

@st.cache_data(show_spinner=False)
def _read_trace(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(Path(path))
    return _ensure_time_current_cols(df)

@st.cache_data(show_spinner=False)
def _read_events(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(Path(path))

# ---------- load selection ----------
qp = st.query_params.to_dict()
run_id = qp.get("run_id") or st.text_input("Enter run_id to load", value="")
if not run_id:
    st.info("Provide a run_id (from Explorer) using the link or text box above.")
    st.stop()

row = get_experiment(run_id)
if not row:
    st.error(f"Run '{run_id}' not found in catalog.")
    st.stop()

trace_path = Path(row.get("trace_path", ""))
events_path = Path(row.get("events_path", "")) if row.get("events_path") else None

# Header (safe n_events)
n_ev_saved = row.get("n_events")
if n_ev_saved is None:
    n_ev_saved = _fast_event_count(events_path)
st.caption(f"**Loaded:** {row.get('title','(untitled)')}  |  {row.get('detector','')}  |  n_events={int(n_ev_saved)}")

# ---------- load data ----------
try:
    tdf = _read_trace(trace_path)
except Exception as e:
    st.error(f"Failed to load trace: {e}")
    st.stop()

if not events_path or not events_path.exists():
    st.info("This run has no events file.")
    st.stop()

try:
    ev = _read_events(events_path)
except Exception as e:
    st.error(f"Failed to load events: {e}")
    st.stop()

# Prepare arrays
if "Time[s]" not in tdf.columns:
    st.error("Trace is missing a recognizable time column.")
    st.stop()

t_arr = pd.to_numeric(tdf["Time[s]"], errors="coerce").to_numpy(float)
cur_col = "current_pA" if "current_pA" in tdf.columns else ("I_filt" if "I_filt" in tdf.columns else None)
if cur_col is None:
    st.error("Trace is missing a recognizable current column (current_pA / I_filt).")
    st.stop()
I_arr = pd.to_numeric(tdf[cur_col], errors="coerce").to_numpy(float)

# Ensure event timing + dwell
try:
    ev = ensure_event_times(ev, t_arr).copy()
except Exception as e:
    st.warning(f"Event time mapping issue: {e}")
if "dwell_s" not in ev.columns and {"start_time","end_time"}.issubset(ev.columns):
    ev["dwell_s"] = pd.to_numeric(ev["end_time"], errors="coerce") - pd.to_numeric(ev["start_time"], errors="coerce")

# ---------- sidebar filters ----------
with st.sidebar:
    st.markdown("**Filters**")
    pad_s = st.slider("Context padding (± s)", 0.1, 5.0, 2.0, 0.1)
    min_dwell = st.number_input("Min dwell (ms)", min_value=0.0, value=0.0, step=0.1)
    max_dwell = st.number_input("Max dwell (ms, 0=∞)", min_value=0.0, value=0.0, step=0.1)
    min_amp   = st.number_input("Min mean depth (pA)", value=0.0, step=0.5)
    sort_by   = st.selectbox("Sort by", ["start_time","dwell_s"], index=0)
    sample_n  = st.slider("Sample (0=all)", 0, 2000, 0, step=50)

    st.markdown("---")
    clamp_y = st.checkbox("Clamp Y axis to 0–250 pA", value=True)

# Build working view
ev_view = ev.copy()
if "dwell_s" in ev_view.columns:
    ev_view["dwell_s"] = pd.to_numeric(ev_view["dwell_s"], errors="coerce")
if "blockade_depth_mean_pA" in ev_view.columns:
    ev_view["blockade_depth_mean_pA"] = pd.to_numeric(ev_view["blockade_depth_mean_pA"], errors="coerce")
elif {"event_mean_pA","baseline_pA"}.issubset(ev_view.columns):
    ev_view["blockade_depth_mean_pA"] = pd.to_numeric(ev_view["baseline_pA"], errors="coerce") - pd.to_numeric(ev_view["event_mean_pA"], errors="coerce")

if min_dwell > 0 and "dwell_s" in ev_view.columns:
    ev_view = ev_view[ev_view["dwell_s"] >= (min_dwell/1000.0)]
if max_dwell > 0 and "dwell_s" in ev_view.columns:
    ev_view = ev_view[ev_view["dwell_s"] <= (max_dwell/1000.0)]
if "blockade_depth_mean_pA" in ev_view.columns and min_amp > 0:
    ev_view = ev_view[ev_view["blockade_depth_mean_pA"] >= min_amp]
if sort_by in ev_view.columns:
    ev_view = ev_view.sort_values(sort_by).reset_index(drop=True)
if sample_n and len(ev_view) > sample_n:
    ev_view = ev_view.sample(sample_n, random_state=0).sort_values(sort_by).reset_index(drop=True)

n_events = len(ev_view)
st.write(f"**Events after filters:** {n_events}")
if n_events == 0:
    st.info("No events after filters.")
    st.stop()

# ---------- state & callbacks (modify before instantiating the number_input) ----------
if "rev_idx" not in st.session_state:
    st.session_state["rev_idx"] = 0
if "labels" not in st.session_state:
    st.session_state["labels"] = {}

def _advance(delta: int = 1):
    # advance index safely
    st.session_state["rev_idx"] = int((int(st.session_state.get("rev_idx", 0)) + delta) % n_events)

def _set_label_and_advance(label: str):
    # label current event id then advance
    cur_idx = int(st.session_state.get("rev_idx", 0))
    ev_id = int(ev_view.iloc[cur_idx].get("event_id", cur_idx))
    st.session_state["labels"][ev_id] = label
    _advance(+1)

# Determine current row for titles/metrics
idx = int(st.session_state["rev_idx"])
row_ev = ev_view.iloc[idx]

# ---------- controls (callbacks fire BEFORE widgets created below) ----------
c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    st.button("⟵ Prev", use_container_width=True, key="btn_prev",
              on_click=_advance, kwargs={"delta": -1})
with c2:
    st.button("Next ⟶", use_container_width=True, key="btn_next",
              on_click=_advance, kwargs={"delta": +1})
with c3:
    st.button("✅ Accept", use_container_width=True, key="btn_accept",
              on_click=_set_label_and_advance, args=("accept",))
with c4:
    st.button("🗑️ Reject", use_container_width=True, key="btn_reject",
              on_click=_set_label_and_advance, args=("reject",))

# Now it's safe to instantiate the number_input bound to the SAME key
st.number_input(
    "Event #",
    min_value=0,
    max_value=max(n_events - 1, 0),
    value=int(st.session_state["rev_idx"]),
    step=1,
    key="rev_idx",
)

# ---------- plot ----------
metrics = event_metrics(row_ev, t_arr, I_arr, pre_pad_s=0.010)
baseline_for_plot = metrics.get("baseline_pA", float(np.nanmedian(I_arr)))

fig_event = event_window(
    t_arr, I_arr, row_ev, pad_s=pad_s, baseline_pa=baseline_for_plot,
    title=f"{row.get('title','(untitled)')} | Event {int(st.session_state['rev_idx'])}/{n_events-1} "
          f"| start={row_ev.start_time:.3f}s dwell={row_ev.dwell_s:.4f}s"
)
if clamp_y:
    fig_event.update_yaxes(range=[0, 250])

st.plotly_chart(fig_event, use_container_width=True, config={"displaylogo": False})

# ---------- metrics ----------
show_metrics = st.checkbox("Show metrics", value=True, key="show_metrics_cb")
if show_metrics:
    colA, colB, colC, colD = st.columns(4)
    colA.metric("Dwell (ms)", f"{row_ev.dwell_s*1e3:.2f}")
    colB.metric("Mean depth (pA)", f"{metrics['amp_mean_pA']:.2f}")
    colC.metric("Min depth (pA)", f"{metrics['amp_min_pA']:.2f}")
    colD.metric("Charge deficit (pA·s)", f"{metrics['charge_deficit_pA_s']:.3g}")

# ---------- export labels ----------
lab = pd.DataFrame([{"event_id": k, "label": v} for k, v in st.session_state["labels"].items()])
st.download_button(
    "⬇️ Download labels.csv",
    data=lab.to_csv(index=False),
    file_name=f"{run_id}_labels.csv",
    mime="text/csv",
    use_container_width=True
)
