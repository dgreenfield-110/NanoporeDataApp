# pages/1_Upload_and_Ingest.py
from __future__ import annotations

import io, time, re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from lib.config import RUNS_DIR
from lib.data_io import new_experiment_row, insert_experiment, get_next_run_id, list_experiments
from lib.viz import trace_view, ensure_event_times, event_window, event_metrics

# ==============================================================================
# Page setup
# ==============================================================================
st.set_page_config(page_title="Upload & Ingest (Offline)", page_icon="📤", layout="wide")
st.title("📤 Upload & Ingest — Offline results")

# ==============================================================================
# Helpers
# ==============================================================================

_VOLTTOK_RE = re.compile(r'^(-?\d{1,3})\s*m[vV]$')
#_ANYMV_RE   = re.compile(r'(?<!\d)(-?\d{1,3})\s*m[vV]\b')
_ANYMV_RE   = re.compile(r'(?<!\d)(-?\d{1,3})\s*m[vV](?=_|\b|$)')


def _infer_voltage_from_names(*names) -> int | None:
    s = " ".join([Path(str(n)).name for n in names if n])
    m = _ANYMV_RE.search(s)
    return int(m.group(1)) if m else None

def _infer_fs_from_df(df: pd.DataFrame) -> float | None:
    try:
        t = pd.to_numeric(df.get("Time[s]", df.get("t")), errors="coerce").to_numpy()
        if t.size >= 2:
            dt = float(np.nanmedian(np.diff(t)))
            if np.isfinite(dt) and dt > 0:
                return float(1.0 / dt)
    except Exception:
        pass
    return None

def _baseline_from_df(df: pd.DataFrame) -> float:
    try:
        I = pd.to_numeric(df.get("I_filt", df.get("current_pA")), errors="coerce").to_numpy()
        if I.size:
            return float(np.nanmedian(I))
    except Exception:
        pass
    return float("nan")

def _coerce_amp(ev: pd.DataFrame, run_baseline: float) -> pd.Series:
    if "blockade_depth_mean_pA" in ev.columns:
        return pd.to_numeric(ev["blockade_depth_mean_pA"], errors="coerce")
    if {"event_mean_pA", "baseline_pA"}.issubset(ev.columns):
        return pd.to_numeric(ev["baseline_pA"], errors="coerce") - pd.to_numeric(ev["event_mean_pA"], errors="coerce")
    if "event_mean_pA" in ev.columns:
        return float(run_baseline) - pd.to_numeric(ev["event_mean_pA"], errors="coerce")
    if "delta_pA" in ev.columns:
        return -pd.to_numeric(ev["delta_pA"], errors="coerce")
    return pd.Series(np.nan, index=ev.index)

# ==============================================================================
# Sidebar inputs
# ==============================================================================
with st.sidebar:
    st.markdown("**Metadata**")
    operator   = st.text_input("Operator", value="")
    pore       = st.text_input("Pore", value="")
    analyte    = st.text_input("Analyte", value="")
    conc       = st.number_input("Concentration (mg/mL)", min_value=0.0, step=1.0, value=0.0)
    denat      = st.text_input("Electrolyte Solution", value="")
    voltage_ui = st.number_input("Voltage (mV)", step=10, value=60)  # not written directly; used only if filenames fail
    notes      = st.text_area("Notes", value="")

    st.markdown("---")
    fs_hz_ui   = st.number_input("Sampling rate (Hz) — used if trace lacks time", step=1, value=10_000)

    st.markdown("---")
    view_len   = st.slider("View window length (s)", 1.0, 60.0, 8.0, step=1.0, key="upl_viewlen")
    max_pts    = st.slider("Max plot points (window)", 10_000, 120_000, 40_000, step=5_000, key="upl_maxpts")
    show_preview = st.checkbox("Show preview", value=True, key="upl_showprev")

c1, c2, c3 = st.columns(3)
with c1:
    trace_file = st.file_uploader("Trace (.npy preferred, also supports .parquet)", type=["npy","parquet"], key="trace_upl")
with c2:
    events_pelt_file = st.file_uploader("Events — PELT (.parquet)", type=["parquet"], key="pelt_upl")
with c3:
    events_hyst_file = st.file_uploader("Events — Hysteresis (.parquet)", type=["parquet"], key="hyst_upl")

if trace_file is None:
    st.info("Upload a trace to begin.")
    st.stop()

# ==============================================================================
# Loaders
# ==============================================================================
@st.cache_data(show_spinner=False)
def _load_trace(file, fs_fallback: float) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(file.getvalue()))
        if "Time[s]" not in df.columns:
            if "t" in df.columns: tcol = "t"
            elif "Time (s)" in df.columns: tcol = "Time (s)"
            else: tcol = df.columns[0]
            df = df.rename(columns={tcol: "Time[s]"})
        if "I_filt" not in df.columns and "current_pA" in df.columns:
            df["I_filt"] = df["current_pA"]
        if "current_pA" not in df.columns and "I_filt" in df.columns:
            df["current_pA"] = df["I_filt"]
        cols = ["Time[s]"] + [c for c in ["I_filt","current_pA"] if c in df.columns]
        return df[cols].copy()

    # .npy
    obj = np.load(io.BytesIO(file.getvalue()), allow_pickle=True)
    t = None; I = None
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        if obj.ndim == 1:
            I = obj.astype(float)
            t = np.arange(len(I), dtype=float) / float(fs_fallback)
        elif obj.ndim == 2 and obj.shape[1] >= 2:
            t = obj[:,0].astype(float); I = obj[:,1].astype(float)
    else:
        try:
            d = obj.item() if hasattr(obj, "item") else dict(obj)
            for k in ["Time[s]","t","time","Time (s)"]:
                if k in d: t = np.asarray(d[k], dtype=float); break
            for k in ["I_filt","current_pA","current","I"]:
                if k in d: I = np.asarray(d[k], dtype=float); break
        except Exception:
            pass

    if I is None:
        raise ValueError("Could not parse .npy trace. Supported: 1D current; 2D [t, I]; or dict with Time[s] + I_filt/current_pA.")
    if t is None:
        t = np.arange(len(I), dtype=float) / float(fs_fallback)
    df = pd.DataFrame({"Time[s]": t, "I_filt": I})
    df["current_pA"] = df["I_filt"]
    return df

@st.cache_data(show_spinner=False)
def _load_events_parquet(file) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(file.getvalue()))

# ==============================================================================
# Load inputs
# ==============================================================================
try:
    work_df = _load_trace(trace_file, fs_fallback=float(fs_hz_ui))
    st.success(f"Trace loaded: {len(work_df):,} samples")
except Exception as e:
    st.exception(e)
    st.stop()

baseline_pa = float(np.nanmedian(work_df["current_pA"])) if "current_pA" in work_df else float("nan")
sigma_open  = float(np.nanstd(work_df.get("current_pA", work_df.get("I_filt")).to_numpy(dtype=float)))

events_pelt = None; events_hyst = None
if events_pelt_file is not None:
    try: events_pelt = _load_events_parquet(events_pelt_file)
    except Exception as e: st.warning(f"Failed to read PELT events: {e}")
if events_hyst_file is not None:
    try: events_hyst = _load_events_parquet(events_hyst_file)
    except Exception as e: st.warning(f"Failed to read Hysteresis events: {e}")

available = []
if isinstance(events_pelt, pd.DataFrame) and not events_pelt.empty: available.append("pelt")
if isinstance(events_hyst, pd.DataFrame) and not events_hyst.empty: available.append("hysteresis")

if not available:
    st.info("No events uploaded yet. You can still preview the trace.")
    active_detector = None
    events_df = pd.DataFrame(columns=["start_time","end_time"])
else:
    active_detector = st.radio("Active events to preview/save", available, horizontal=True, index=0, key="active_events_radio")
    events_df = events_pelt if active_detector == "pelt" else events_hyst

# Ensure event times + dwell
try:
    t_arr = work_df["Time[s]"].to_numpy(float)
    if not events_df.empty:
        events_df = ensure_event_times(events_df, t_arr)
        if "dwell_s" not in events_df.columns:
            events_df["dwell_s"] = events_df["end_time"] - events_df["start_time"]
except Exception as e:
    st.warning(f"Event time mapping issue: {e}")
    events_df = pd.DataFrame(columns=["start_time","end_time","dwell_s"])

# ==============================================================================
# Preview
# ==============================================================================
if show_preview:
    try:
        tmin, tmax = float(work_df["Time[s]"].min()), float(work_df["Time[s]"].max())
        key = "trace_view"
        if key not in st.session_state:
            st.session_state[key] = (tmin, min(tmin + view_len, tmax))
        v0, v1 = st.slider("View range (s)", min_value=tmin, max_value=tmax,
                           value=st.session_state[key],
                           step=max((tmax - tmin) / 1000.0, 0.001), key="trace_view_slider")
        if abs((v1 - v0) - view_len) > 1e-9:
            v1 = min(v0 + view_len, tmax)
        st.session_state[key] = (v0, v1)

        # Preview = no baseline guideline; event plot will include baseline
        fig = trace_view(
            work_df, events_df if not events_df.empty else None,
            baseline_pa=None, view=(v0, v1), max_points=max_pts,
            title=f"Preview ({active_detector or 'no events'})"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    except Exception as e:
        st.warning(f"Preview failed: {e}")

# ==============================================================================
# Event Annotation Potential
# ==============================================================================
st.markdown("---")
st.subheader("🧪 Event Annotation Potential")

c_f1, c_f2, c_f3, c_f4 = st.columns(4)
with c_f1:
    ap_min_dwell_ms = st.number_input("Min dwell (ms)", 0.0, 1e6, 0.2, 0.1, key="ap_min_dw")
with c_f2:
    ap_max_dwell_ms = st.number_input("Max dwell (ms, 0=∞)", 0.0, 1e6, 150.0, 0.1, key="ap_max_dw")
with c_f3:
    ap_min_amp_pa   = st.number_input("Min mean depth (pA)", 0.0, 1e6, 2.0, 0.1, key="ap_min_amp")
with c_f4:
    ap_sample_limit = st.number_input("Sample limit for preview (0=all)", 0, 100000, 0, 100, key="ap_smpl")

n_total = int(len(events_df)) if isinstance(events_df, pd.DataFrame) else 0
if n_total == 0:
    st.info("No events loaded — upload a PELT/Hysteresis events parquet to estimate annotation potential.")
else:
    evp = events_df.copy()
    evp["blockade_depth_mean_pA"] = _coerce_amp(evp, baseline_pa)
    if "dwell_s" not in evp.columns:
        evp["dwell_s"] = pd.to_numeric(evp.get("duration_s"), errors="coerce")
    evp["dwell_ms"] = pd.to_numeric(evp["dwell_s"], errors="coerce") * 1000.0

    mask = pd.Series(True, index=evp.index)
    if ap_min_dwell_ms > 0:
        mask &= evp["dwell_ms"] >= ap_min_dwell_ms
    if ap_max_dwell_ms > 0:
        mask &= evp["dwell_ms"] <= ap_max_dwell_ms
    if ap_min_amp_pa > 0:
        mask &= evp["blockade_depth_mean_pA"] >= ap_min_amp_pa

    ev_good = evp[mask].dropna(subset=["dwell_ms", "blockade_depth_mean_pA"])
    n_good = int(len(ev_good))
    ratio = (n_good / n_total * 100.0) if n_total else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Total events", f"{n_total:,}")
    m2.metric("Meets thresholds", f"{n_good:,}", f"{ratio:.1f}%")
    m3.metric("Baseline (pA)", f"{baseline_pa:.2f}")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    if ap_sample_limit and len(ev_good) > ap_sample_limit:
        ev_good = ev_good.sample(ap_sample_limit, random_state=0)

    fig_ap = make_subplots(rows=1, cols=2, subplot_titles=("Dwell (ms)", "Mean depth (pA)"))
    fig_ap.add_trace(go.Histogram(x=ev_good["dwell_ms"], opacity=0.75, name="dwell"), row=1, col=1)
    fig_ap.add_trace(go.Histogram(x=ev_good["blockade_depth_mean_pA"], opacity=0.75, name="amp"), row=1, col=2)
    fig_ap.update_layout(barmode="overlay", margin=dict(l=40, r=20, t=40, b=40), height=340, showlegend=False)
    fig_ap.update_xaxes(title_text="ms", row=1, col=1)
    fig_ap.update_xaxes(title_text="pA", row=1, col=2)
    st.plotly_chart(fig_ap, use_container_width=True, config={"displaylogo": False})

# ==============================================================================
# Slideshow (manual review)
# ==============================================================================
st.markdown("---")
st.subheader("Event slideshow")

with st.sidebar:
    st.markdown("---")
    st.markdown("**Slideshow preview**")
    pad_s = st.slider("Context padding (± s)", 0.2, 5.0, 2.0, 0.1, key="pad_slider")
    min_dwell = st.number_input("Min dwell (ms)", min_value=0.0, value=0.0, step=0.1, key="min_dwell_in")
    max_dwell = st.number_input("Max dwell (ms, 0=∞)", min_value=0.0, value=0.0, step=0.1, key="max_dwell_in")
    min_amp   = st.number_input("Min mean depth (pA)", value=0.0, step=0.5, key="min_amp_in")
    sort_by   = st.selectbox("Sort events by", ["start_time","dwell_s"], index=0, key="sort_by_sel")
    sample_n  = st.slider("Sample (0=all)", 0, 2000, 0, step=50, key="sample_slider")
    clamp_y   = st.checkbox("Clamp Y to 0–250 pA", value=False, key="slideshow_clamp_y")

# Prepare arrays
t_arr = work_df["Time[s]"].to_numpy(float)
I_arr = (work_df["I_filt"] if "I_filt" in work_df else work_df["current_pA"]).to_numpy(float)

# Build working view of events
ev_view = events_df.copy()
if not ev_view.empty:
    if "dwell_s" not in ev_view.columns:
        if {"start_time", "end_time"}.issubset(ev_view.columns):
            ev_view["dwell_s"] = pd.to_numeric(ev_view["end_time"], errors="coerce") - pd.to_numeric(ev_view["start_time"], errors="coerce")
        else:
            ev_view["dwell_s"] = pd.to_numeric(ev_view.get("duration_s"), errors="coerce")

    if "blockade_depth_mean_pA" not in ev_view.columns:
        if {"event_mean_pA","baseline_pA"}.issubset(ev_view.columns):
            ev_view["blockade_depth_mean_pA"] = pd.to_numeric(ev_view["baseline_pA"], errors="coerce") - pd.to_numeric(ev_view["event_mean_pA"], errors="coerce")
        elif "delta_pA" in ev_view.columns:
            ev_view["blockade_depth_mean_pA"] = -pd.to_numeric(ev_view["delta_pA"], errors="coerce")

    ev_view["dwell_s"] = pd.to_numeric(ev_view["dwell_s"], errors="coerce")
    if "blockade_depth_mean_pA" in ev_view.columns:
        ev_view["blockade_depth_mean_pA"] = pd.to_numeric(ev_view["blockade_depth_mean_pA"], errors="coerce")

    if min_dwell > 0:
        ev_view = ev_view[ev_view["dwell_s"] >= (min_dwell/1000.0)]
    if max_dwell > 0:
        ev_view = ev_view[ev_view["dwell_s"] <= (max_dwell/1000.0)]
    if "blockade_depth_mean_pA" in ev_view.columns and min_amp > 0:
        ev_view = ev_view[ev_view["blockade_depth_mean_pA"] >= min_amp]
    if sort_by in ev_view.columns:
        ev_view = ev_view.sort_values(sort_by).reset_index(drop=True)
    if sample_n and len(ev_view) > sample_n:
        ev_view = ev_view.sample(sample_n, random_state=0).sort_values(sort_by).reset_index(drop=True)

# Count AFTER filtering and reset_index
n_ev = int(len(ev_view))
st.write(f"**Slideshow:** {n_ev} event(s) after filters")
if n_ev == 0:
    st.info("No events to show.")
    st.stop()

# ---------- state, CLAMP, and callbacks ----------
if "slideshow_idx" not in st.session_state:
    st.session_state["slideshow_idx"] = 0

# CLAMP the index BEFORE any widget is created to avoid out-of-bounds
_prev_idx = int(st.session_state.get("slideshow_idx", 0))
_clamped = 0 if n_ev == 0 else max(0, min(_prev_idx, n_ev - 1))
if _clamped != _prev_idx:
    st.session_state["slideshow_idx"] = _clamped
cur_idx_for_widgets = int(st.session_state["slideshow_idx"])

def _advance(delta: int = 1):
    n = int(len(ev_view))
    if n == 0:
        return
    st.session_state["slideshow_idx"] = int((int(st.session_state.get("slideshow_idx", 0)) + delta) % n)

def _label_and_advance(label: str):
    n = int(len(ev_view))
    if n == 0:
        return
    idx = int(st.session_state.get("slideshow_idx", 0))
    if idx >= n:
        idx = n - 1
        st.session_state["slideshow_idx"] = idx  # safe: no widget yet in callback context
    ev_id = int(ev_view.iloc[idx].get("event_id", idx))
    if "labels" not in st.session_state:
        st.session_state["labels"] = {}
    st.session_state["labels"][ev_id] = label
    _advance(+1)

# ---------- top controls (callbacks fire BEFORE number_input) ----------
#c1, c2, c3, c4 = st.columns([1,1,1,1])
c1, c2  = st.columns([1,1])
with c1:
    st.button("⟵ Prev", use_container_width=True, key="upl_prev", on_click=_advance, kwargs={"delta": -1})
with c2:
    st.button("Next ⟶", use_container_width=True, key="upl_next", on_click=_advance, kwargs={"delta": +1})
#with c3:
#    st.button("✅ Accept", use_container_width=True, key="upl_accept", on_click=_label_and_advance, args=("accept",))
#with c4:
#    st.button("🗑️ Reject", use_container_width=True, key="upl_reject", on_click=_label_and_advance, args=("reject",))

# Number input bound to the SAME key, created AFTER clamping
st.number_input(
    "Event #",
    min_value=0,
    max_value=max(n_ev - 1, 0),
    value=cur_idx_for_widgets,
    step=1,
    key="slideshow_idx",
)

# Use the (now safe) index to select the row
cur_idx = int(st.session_state["slideshow_idx"])
row = ev_view.iloc[cur_idx]

# ---------- metrics & plot ----------
metrics = event_metrics(row, t_arr, I_arr, pre_pad_s=0.010)
baseline_for_plot = metrics.get("baseline_pA", float(np.nanmedian(I_arr)))

fig_event = event_window(
    t_arr, I_arr, row, pad_s=pad_s, baseline_pa=baseline_for_plot,
    title=f"Event {cur_idx}/{n_ev-1} | start={row.start_time:.3f}s  dwell={row.dwell_s:.4f}s"
)
if st.session_state.get("slideshow_clamp_y", False):
    fig_event.update_yaxes(range=[0, 250])

st.plotly_chart(fig_event, use_container_width=True, config={"displaylogo": False})

colA, colB, colC, colD = st.columns(4)
colA.metric("Dwell (ms)", f"{row.dwell_s*1e3:.2f}")
colB.metric("Mean depth (pA)", f"{metrics['amp_mean_pA']:.2f}")
colC.metric("Min depth (pA)", f"{metrics['amp_min_pA']:.2f}")
colD.metric("Charge deficit (pA·s)", f"{metrics['charge_deficit_pA_s']:.3g}")

# ==============================================================================
# Persist to catalog
# ==============================================================================
st.markdown("---")

def _base_stem_for(src_file) -> str:
    if src_file is not None and hasattr(src_file, "name"):
        return Path(src_file.name).stem
    if trace_file is not None and hasattr(trace_file, "name"):
        return Path(trace_file.name).stem
    return time.strftime("run-%Y%m%d-%H%M%S")

def _active_source_file():
    try:
        if active_detector == "pelt" and events_pelt_file is not None:
            return events_pelt_file
        if active_detector == "hysteresis" and events_hyst_file is not None:
            return events_hyst_file
    except NameError:
        pass
    return trace_file

def _save_one(det_name: str, ev_df: pd.DataFrame | None, src_file) -> int:
    if ev_df is None:
        return 0

    base_title = _base_stem_for(src_file)
    run_id = get_next_run_id()  # numeric string '1','2',...

    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    trace_path  = run_dir / "trace_work.parquet"
    events_path = run_dir / "events.parquet"

    work_df.to_parquet(trace_path)
    (ev_df if not ev_df.empty else pd.DataFrame(columns=["start_time","end_time"])).to_parquet(events_path)

    fs_hz    = _infer_fs_from_df(work_df) or float(fs_hz_ui)
    baseline = _baseline_from_df(work_df)
    volt_guess = _infer_voltage_from_names(
        getattr(src_file, "name", None),
        base_title,
    )
    
    meta = dict(
        run_id=run_id,
        title=base_title,
        operator=operator,
        pore=pore,
        analyte=analyte,
        concentration_nM=conc,
        denaturant=denat,
        voltage_mV = int(volt_guess if volt_guess is not None else voltage_ui),  # <-- set explicit voltage
        fs_hz=float(fs_hz) if fs_hz else None,
        detector=det_name,
        cutoff_hz=None,
        baseline_pa=float(baseline) if np.isfinite(baseline) else None,
        notes=notes,
    )
    row_meta = new_experiment_row(meta, trace_path, events_path, stats={})
    insert_experiment(row_meta)
    return 1

save_ok = st.button("💾 Save uploaded runs (PELT & Hysteresis separately)", type="primary", key="save_run")
if save_ok:
    try:
        saved = 0
        if isinstance(events_pelt, pd.DataFrame):
            saved += _save_one("pelt", events_pelt, events_pelt_file)
        if isinstance(events_hyst, pd.DataFrame):
            saved += _save_one("hysteresis", events_hyst, events_hyst_file)
        if saved == 0:
            saved += _save_one("precomputed", pd.DataFrame(columns=["start_time","end_time"]), _active_source_file())
        st.success(f"Saved {saved} run(s) to catalog.")
    except Exception as e:
        st.exception(e)

# ==============================================================================
# Agreement (uploaded OR catalog)
# ==============================================================================
st.markdown("---")
st.subheader("Detector agreement (event overlap)")

@st.cache_data(show_spinner=False)
def _prep_events_for_agreement_from_row(run_row: pd.Series) -> pd.DataFrame:
    try:
        tdf = pd.read_parquet(Path(run_row["trace_path"]))
    except Exception:
        return pd.DataFrame(columns=["start_time","end_time","dwell_s"])
    try:
        ev = pd.read_parquet(Path(run_row.get("events_path", "")))
    except Exception:
        ev = pd.DataFrame(columns=["start_time","end_time","dwell_s"])
    try:
        t = pd.to_numeric(tdf["Time[s]"], errors="coerce").to_numpy(float)
        ev = ensure_event_times(ev, t)
    except Exception:
        pass
    if "dwell_s" not in ev.columns and {"start_time","end_time"}.issubset(ev.columns):
        ev["dwell_s"] = pd.to_numeric(ev["end_time"], errors="coerce") - pd.to_numeric(ev["start_time"], errors="coerce")
    out = ev[["start_time","end_time","dwell_s"]].copy()
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["start_time","end_time"])
    out = out[out["end_time"] > out["start_time"]].reset_index(drop=True)
    return out

def _prep_from_df_for_agreement(ev_df: pd.DataFrame, t_arr: np.ndarray) -> pd.DataFrame:
    if ev_df is None or ev_df.empty:
        return pd.DataFrame(columns=["start_time","end_time","dwell_s"])
    ev = ensure_event_times(ev_df.copy(), t_arr)
    if "dwell_s" not in ev.columns and {"start_time","end_time"}.issubset(ev.columns):
        ev["dwell_s"] = pd.to_numeric(ev["end_time"], errors="coerce") - pd.to_numeric(ev["start_time"], errors="coerce")
    return ev[["start_time","end_time","dwell_s"]].replace([np.inf,-np.inf], np.nan).dropna().reset_index(drop=True)

_use_uploaded_for_agree = (
    isinstance(events_pelt, pd.DataFrame) and not events_pelt.empty
    and isinstance(events_hyst, pd.DataFrame) and not events_hyst.empty
)

if _use_uploaded_for_agree:
    st.caption("**Agreement source:** comparing **uploaded** PELT vs Hysteresis on this page.")
    t_arr_local = pd.to_numeric(work_df["Time[s]"], errors="coerce").to_numpy(float)
    evA = _prep_from_df_for_agreement(events_pelt, t_arr_local)
    evB = _prep_from_df_for_agreement(events_hyst, t_arr_local)
    titleA, detA = (events_pelt_file.name if events_pelt_file is not None else "PELT (uploaded)"), "pelt"
    titleB, detB = (events_hyst_file.name if events_hyst_file is not None else "Hysteresis (uploaded)"), "hysteresis"
else:
    view = list_experiments()
    options = list(view["run_id"]) if ("run_id" in view.columns) else []
    if len(options) < 2:
        st.info("Need at least two saved runs to compute agreement. Save PELT & Hysteresis first.")
        st.stop()
    left, right = st.columns(2)
    with left:
        runA_id = st.selectbox("Run A (reference)", options=options, index=0, key="agree_runA")
    with right:
        runB_id = st.selectbox("Run B (comparison)", options=options, index=min(1, len(options)-1), key="agree_runB")
    rowA = view.loc[view["run_id"] == runA_id].iloc[0]
    rowB = view.loc[view["run_id"] == runB_id].iloc[0]
    evA = _prep_events_for_agreement_from_row(rowA)
    evB = _prep_events_for_agreement_from_row(rowB)
    titleA = rowA.get("title", str(runA_id)); detA = rowA.get("detector", "")
    titleB = rowB.get("title", str(runB_id)); detB = rowB.get("detector", "")

if evA.empty or evB.empty:
    st.info("One or both event sets are empty—cannot compute agreement.")
    st.stop()

def _merge_intervals(intervals: list[tuple[float,float]]) -> list[tuple[float,float]]:
    if not intervals:
        return []
    iv = sorted(intervals)
    merged = [iv[0]]
    for s, e in iv[1:]:
        ls, le = merged[-1]
        if s <= le:
            merged[-1] = (ls, max(le, e))
        else:
            merged.append((s, e))
    return merged

def _total_len(intervals: list[tuple[float,float]]) -> float:
    return float(sum(max(0.0, e - s) for s, e in intervals))

def _intersections_len(A: list[tuple[float,float]], B: list[tuple[float,float]]) -> float:
    i = j = 0
    inter = 0.0
    while i < len(A) and j < len(B):
        a1, a2 = A[i]; b1, b2 = B[j]
        s = max(a1, b1); e = min(a2, b2)
        if e > s:
            inter += (e - s)
        if a2 <= b2:
            i += 1
        else:
            j += 1
    return float(inter)

def _match_greedy_iou(A: pd.DataFrame, B: pd.DataFrame, pad_s: float = 0.0, offsetB_s: float = 0.0) -> pd.DataFrame:
    if A.empty or B.empty:
        return pd.DataFrame(columns=["a_idx","b_idx","a_start","a_end","b_start","b_end","overlap_s","union_s","iou"])
    a = A.copy().reset_index(drop=True)
    b = B.copy().reset_index(drop=True)
    a["a_start"] = a["start_time"] - pad_s
    a["a_end"]   = a["end_time"]   + pad_s
    b["b_start"] = b["start_time"] + offsetB_s - pad_s
    b["b_end"]   = b["end_time"]   + offsetB_s + pad_s
    a["a_idx"] = np.arange(len(a))
    b["b_idx"] = np.arange(len(b))
    cand = []
    for i, rA in a.iterrows():
        sA, eA = float(rA["a_start"]), float(rA["a_end"])
        for j, rB in b.iterrows():
            sB, eB = float(rB["b_start"]), float(rB["b_end"])
            ov = max(0.0, min(eA, eB) - max(sA, sB))
            if ov <= 0:
                continue
            un = max(eA, eB) - min(sA, sB)
            iou = ov / un if un > 0 else 0.0
            cand.append((iou, i, j, sA, eA, sB, eB, ov, un))
    if not cand:
        return pd.DataFrame(columns=["a_idx","b_idx","a_start","a_end","b_start","b_end","overlap_s","union_s","iou"])
    cand.sort(key=lambda x: x[0], reverse=True)
    usedA, usedB, out = set(), set(), []
    for iou, i, j, sA, eA, sB, eB, ov, un in cand:
        if (i in usedA) or (j in usedB):
            continue
        usedA.add(i); usedB.add(j)
        out.append((i, j, sA, eA, sB, eB, ov, un, iou))
    return pd.DataFrame(out, columns=[
        "a_idx","b_idx","a_start","a_end","b_start","b_end","overlap_s","union_s","iou"
    ])

# Controls
c1, c2, c3, c4 = st.columns(4)
with c1:
    iou_min = st.slider("Min IoU to count as a TP", 0.0, 1.0, 0.2, 0.05, key="agree_iou_min")
with c2:
    pad_s = st.slider("Pad each event (± s)", 0.0, 0.050, 0.005, 0.001, key="agree_pad")
with c3:
    offsetB = st.slider("Time offset for Run B (s)", -0.5, 0.5, 0.0, 0.001, key="agree_offset")
with c4:
    show_table = st.checkbox("Show matched table", True, key="agree_show_tbl")

matches = _match_greedy_iou(evA, evB, pad_s=pad_s, offsetB_s=offsetB)
tp_tbl = matches.loc[matches["iou"] >= iou_min].copy()

TP = int(len(tp_tbl))
FN = int(len(evA) - TP)
FP = int(len(evB) - TP)

precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
f1        = 2*precision*recall / (precision + recall) if (precision + recall) > 0 else 0.0
mean_iou  = float(tp_tbl["iou"].mean()) if TP > 0 else 0.0

A_union = _merge_intervals(list(zip(evA["start_time"], evA["end_time"])))
B_union = _merge_intervals(list(zip(evB["start_time"], evB["end_time"])))
A_len = _total_len(A_union)
B_len = _total_len(B_union)
inter_len = _intersections_len(A_union, B_union)
dice_time = 2*inter_len / (A_len + B_len) if (A_len + B_len) > 0 else 0.0

st.caption(
    f"**Comparing:** A = *{(locals().get('titleA') or 'Run A')}* ({locals().get('detA','')}, n={len(evA)})  vs  "
    f"B = *{(locals().get('titleB') or 'Run B')}* ({locals().get('detB','')}, n={len(evB)})"
)

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Dice (time coverage)", f"{dice_time:.3f}")
m2.metric("Precision", f"{precision:.3f}", f"TP={TP}, FP={FP}")
m3.metric("Recall", f"{recall:.3f}", f"FN={FN}")
m4.metric("F1", f"{f1:.3f}")
m5.metric("Mean IoU (TPs)", f"{mean_iou:.3f}")

import plotly.graph_objects as go
if st.session_state.get("agree_show_tbl", True):
    st.dataframe(
        tp_tbl[["a_idx","b_idx","a_start","a_end","b_start","b_end","overlap_s","union_s","iou"]],
        use_container_width=True, hide_index=True
    )
    st.download_button(
        "⬇️ Download matched_pairs.csv",
        data=tp_tbl.to_csv(index=False),
        file_name="matched_pairs.csv",
        mime="text/csv",
        use_container_width=True
    )

fig_iou = go.Figure()
if TP > 0:
    fig_iou.add_trace(go.Histogram(x=tp_tbl["iou"], nbinsx=30, opacity=0.8, name="IoU (TPs)"))
fig_iou.update_layout(
    title="Distribution of IoU for true matches",
    xaxis_title="IoU", yaxis_title="Count", barmode="overlay",
    margin=dict(l=50, r=10, t=40, b=40),
)
st.plotly_chart(fig_iou, use_container_width=True, config={"displaylogo": False})
