# pages/3_Experiment_Comparison.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from lib.data_io import list_experiments
from lib.viz import trace_view  # for overlay

st.set_page_config(page_title="Experiment Comparison", page_icon="🧪", layout="wide")
st.title("🧪 Experiment Comparison")

# -------------------------- helpers: robust metadata --------------------------
_VOLTS_SUFFIX = re.compile(r'_(\d{2,3})mV(?:\b|$)', re.IGNORECASE)
_VOLTS_FREE   = re.compile(r'(?<!\d)(\d{1,3})\s*m[vV]\b')

def _guess_voltage_any(*names) -> int | None:
    s = " ".join([str(x) for x in names if x])
    m = _VOLTS_SUFFIX.search(s) or _VOLTS_FREE.search(s)
    return int(m.group(1)) if m else None

def _ensure_cols(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize key columns and backfill missing values (incl. n_events and voltage)."""
    out = frame.copy()

    if "run_id" not in out.columns:
        out["run_id"] = out.index.astype(str)

    if "title" not in out.columns:
        out["title"] = out["run_id"].astype(str)

    for c in ("operator", "pore", "analyte", "detector", "notes"):
        if c not in out.columns:
            out[c] = ""

    # voltage: prefer canonical measurement_voltage_mV, otherwise parse from names/paths
    if "measurement_voltage_mV" in out.columns:
        out["voltage_mV"] = pd.to_numeric(out.get("measurement_voltage_mV"), errors="coerce")
    else:
        out["voltage_mV"] = pd.to_numeric(out.get("voltage_mV"), errors="coerce")

    def _v_row(row):
        if pd.notna(row.get("voltage_mV")):
            try:
                return int(row["voltage_mV"])
            except Exception:
                pass
        return _guess_voltage_any(
            row.get("run_id", ""), row.get("title", ""),
            row.get("events_path", ""), row.get("trace_path", "")
        )

    out["voltage_mV"] = out.apply(_v_row, axis=1).astype("Int64")

    # n_events: count from parquet if missing
    if "n_events" not in out.columns:
        out["n_events"] = np.nan

    def _fast_count(p):
        try:
            if isinstance(p, str) and p and Path(p).exists():
                try:
                    import pyarrow.parquet as pq  # fast path
                    return int(pq.ParquetFile(p).metadata.num_rows)
                except Exception:
                    return int(pd.read_parquet(p).shape[0])
        except Exception:
            pass
        return 0

    miss_ne = out.get("n_events")
    if miss_ne is None or pd.isna(miss_ne).any():
        if "events_path" in out.columns:
            out["n_events"] = [
                _fast_count(p) if pd.isna(n) else int(n)
                for p, n in zip(out["events_path"].fillna(""), out.get("n_events", pd.Series(index=out.index)))
            ]
        else:
            out["n_events"] = 0

    return out

# -------------------------- load catalog + filters --------------------------
df = list_experiments()
if df is None or df.empty:
    st.info("Catalog is empty. Ingest some runs first.")
    st.stop()

df = _ensure_cols(df)

with st.sidebar:
    st.markdown("**Filter by metadata**")
    q = st.text_input("Search (title / operator / analyte / pore / notes)", "")
    ops  = st.multiselect("Operator", sorted([x for x in df["operator"].dropna().unique() if x]))
    pores = st.multiselect("Pore", sorted([x for x in df["pore"].dropna().unique() if x]))
    analytes = st.multiselect("Analyte", sorted([x for x in df["analyte"].dropna().unique() if x]))
    detectors = st.multiselect("Detector", sorted([x for x in df["detector"].dropna().unique() if x]))
    min_events = st.number_input("Min events", value=0, step=1)

    st.markdown("---")
    st.markdown("**Trace overlay controls**")
    overlay_len = st.slider("Overlay window length (s)", 0.5, 30.0, 5.0, step=0.5)
    max_points_overlay = st.slider("Max plot points (overlay)", 10_000, 120_000, 40_000, step=5_000)

    st.markdown("---")
    st.markdown("**Histogram controls**")
    bins_dwell = st.slider("Bins (dwell ms)", 10, 200, 60, step=5)
    bins_amp   = st.slider("Bins (amp pA)", 10, 200, 60, step=5)
    log_dwell  = st.checkbox("Log x-axis for dwell histogram", value=False)

view = df.copy()

if q:
    ql = q.lower()
    view = view[view.apply(lambda r:
        any(ql in str(r.get(c, "")).lower() for c in ["title","operator","analyte","pore","notes"]), axis=1
    )]

if ops:
    view = view[view["operator"].isin(ops)]
if pores:
    view = view[view["pore"].isin(pores)]
if analytes:
    view = view[view["analyte"].isin(analytes)]
if detectors:
    view = view[view["detector"].isin(detectors)]

view = view[pd.to_numeric(view["n_events"], errors="coerce").fillna(0) >= int(min_events)].reset_index(drop=True)

st.caption(f"Matches: **{len(view)}**")
if view.empty:
    st.stop()

show_cols = ["run_id","title","created_at","operator","pore","analyte","detector","voltage_mV","fs_hz","n_events"]
have_cols = [c for c in show_cols if c in view.columns]
st.dataframe(view[have_cols], use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Select experiments to compare")

# -------------------------- selection --------------------------
sel_ids = st.multiselect(
    "Choose up to 6 runs",
    options=list(view["run_id"].astype(str)),
    default=list(view["run_id"].astype(str).head(2)),
    max_selections=6,
)
if not sel_ids:
    st.info("Select at least one run.")
    st.stop()

sel_df = view[view["run_id"].astype(str).isin([str(x) for x in sel_ids])].copy()

# -------------------------- summary table --------------------------
st.markdown("### Summary")
summary_cols = ["run_id","title","operator","pore","analyte","detector","voltage_mV","fs_hz","baseline_pa","n_events","notes"]
have_sum = [c for c in summary_cols if c in sel_df.columns]
st.dataframe(sel_df[have_sum], use_container_width=True, hide_index=True)

# -------------------------- loaders --------------------------
def _ensure_time_current_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Time
    if "Time[s]" not in out.columns:
        for cand in ("t","Time (s)"):
            if cand in out.columns:
                out = out.rename(columns={cand: "Time[s]"})
                break
        if "Time[s]" not in out.columns:
            out = out.rename(columns={out.columns[0]: "Time[s]"})
    # Current
    if ("current_pA" not in out.columns) and ("I_filt" not in out.columns):
        for cand in ("current_pA","I_filt","current","I","current_nA"):
            if cand in out.columns:
                out = out.rename(columns={cand: "current_pA"})
                break
    return out

def _load_trace_events(trace_path: str | Path, events_path: str | Path | None):
    tdf = pd.read_parquet(Path(trace_path))
    tdf = _ensure_time_current_cols(tdf)
    ev = None
    if events_path:
        p = Path(str(events_path))
        if p.exists():
            try:
                ev = pd.read_parquet(p)
            except Exception:
                ev = None
    return tdf, ev

# -------------------------- overlay: traces --------------------------
def overlay_traces(sel_rows: pd.DataFrame, window_len_s: float, max_points: int):
    tmins, tmaxs = [], []
    traces = []
    for _, r in sel_rows.iterrows():
        tdf, _ = _load_trace_events(r["trace_path"], r.get("events_path"))
        if "Time[s]" not in tdf.columns:
            continue
        t = pd.to_numeric(tdf["Time[s]"], errors="coerce").to_numpy(float)
        cur_col = "current_pA" if "current_pA" in tdf.columns else ("I_filt" if "I_filt" in tdf.columns else None)
        if cur_col is None:
            continue
        I = pd.to_numeric(tdf[cur_col], errors="coerce").to_numpy(float)
        if len(t) == 0 or len(I) == 0:
            continue
        tmins.append(float(np.nanmin(t))); tmaxs.append(float(np.nanmax(t)))
        traces.append((r["run_id"], r["title"], t, I))
    if not traces:
        st.info("No traces could be loaded for overlay.")
        return

    v0 = max(tmins); v1 = min(tmaxs)
    if not np.isfinite(v0) or not np.isfinite(v1) or v1 <= v0:
        v0, v1 = float(traces[0][2].min()), float(traces[0][2].min()) + window_len_s
    else:
        v1 = min(v0 + window_len_s, v1)

    fig = go.Figure()
    for rid, title, t, I in traces:
        m = (t >= v0) & (t <= v1)
        if not np.any(m):
            continue
        tx = t[m]; Ix = I[m]
        # min/max downsample
        n = len(Ix)
        if n > max_points > 0:
            bins = np.linspace(tx[0], tx[-1], max_points + 1)
            idx = np.searchsorted(tx, bins); idx[-1] = n
            tout, yout = [], []
            for i in range(max_points):
                a, b = idx[i], idx[i+1]
                if b <= a: continue
                seg = Ix[a:b]
                jmin = a + int(np.argmin(seg)); jmax = a + int(np.argmax(seg))
                if jmin < jmax:
                    tout.extend([tx[jmin], tx[jmax]]); yout.extend([Ix[jmin], Ix[jmax]])
                else:
                    tout.extend([tx[jmax], tx[jmin]]); yout.extend([Ix[jmax], Ix[jmin]])
            tx, Ix = np.asarray(tout), np.asarray(yout)
        fig.add_trace(go.Scattergl(x=tx, y=Ix, mode="lines", name=f"{title} ({rid})", line={"width":1}))

    fig.update_layout(
        title=f"Trace overlay [{v0:.3f}s → {v1:.3f}s]",
        xaxis_title="Time (s)", yaxis_title="Current (pA)",
        hovermode="x unified", margin=dict(l=50,r=10,t=40,b=40), showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

st.markdown("### Trace overlay")
overlay_traces(sel_df, overlay_len, max_points_overlay)

# -------------------------- collect events (ROBUST) --------------------------
def _coerce_amp(ev: pd.DataFrame, run_baseline: float) -> pd.Series:
    """Return a Series of mean blockade depth (pA) per event, if possible."""
    if "blockade_depth_mean_pA" in ev.columns:
        return pd.to_numeric(ev["blockade_depth_mean_pA"], errors="coerce")
    if {"event_mean_pA", "baseline_pA"}.issubset(ev.columns):
        return pd.to_numeric(ev["baseline_pA"], errors="coerce") - pd.to_numeric(ev["event_mean_pA"], errors="coerce")
    if "event_mean_pA" in ev.columns:
        return float(run_baseline) - pd.to_numeric(ev["event_mean_pA"], errors="coerce")
    if "delta_pA" in ev.columns:
        # convention: negative delta means downward blockade
        return -pd.to_numeric(ev["delta_pA"], errors="coerce")
    return pd.Series(np.nan, index=ev.index)

def _collect_events_from_runs(sel_rows: pd.DataFrame) -> pd.DataFrame:
    """Return tidy table across selected runs with robust dwell/amp handling."""
    rows = []
    for _, r in sel_rows.iterrows():
        tdf, ev = _load_trace_events(r["trace_path"], r.get("events_path"))
        if ev is None or ev.empty:
            continue

        # baseline for fallback amp
        cur_col = "current_pA" if "current_pA" in tdf.columns else ("I_filt" if "I_filt" in tdf.columns else None)
        baseline_pa = float(np.nanmedian(pd.to_numeric(tdf[cur_col], errors="coerce"))) if cur_col else np.nan

        tmp = ev.copy()

        # dwell (prefer dwell_s; else end-start; else duration_s)
        dwell_s = pd.to_numeric(tmp.get("dwell_s"), errors="coerce")
        if dwell_s is None or dwell_s.isna().all():
            if {"start_time","end_time"}.issubset(tmp.columns):
                dwell_s = pd.to_numeric(tmp["end_time"], errors="coerce") - pd.to_numeric(tmp["start_time"], errors="coerce")
            else:
                dwell_s = pd.to_numeric(tmp.get("duration_s"), errors="coerce")

        # amplitude (mean depth, pA)
        amp_pa = _coerce_amp(tmp, baseline_pa)

        # build tidy
        rows.append(pd.DataFrame({
            "run_id": str(r["run_id"]),
            "title": r.get("title", r["run_id"]),
            "detector": (r.get("detector") or "unknown"),
            "voltage_mV": int(r["voltage_mV"]) if pd.notna(r["voltage_mV"]) else None,
            "dwell_ms": dwell_s * 1000.0,
            "amp_pa": pd.to_numeric(amp_pa, errors="coerce"),
        }))
    if not rows:
        return pd.DataFrame(columns=["run_id","title","detector","voltage_mV","dwell_ms","amp_pa"])
    out = pd.concat(rows, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["dwell_ms","amp_pa"])
    out = out[(out["dwell_ms"] > 0)]  # keep only positive dwell
    return out

ev_all = _collect_events_from_runs(sel_df)

st.markdown("### Event distributions")
if ev_all.empty:
    st.info("No events available in selected runs.")
else:
    # ---------------- dwell ----------------
    fig_dwell = go.Figure()
    for rid, grp in ev_all.groupby("run_id"):
        fig_dwell.add_trace(go.Histogram(
            x=grp["dwell_ms"], name=str(rid), nbinsx=int(bins_dwell), opacity=0.6
        ))
    if log_dwell:
        fig_dwell.update_xaxes(type="log")
    fig_dwell.update_layout(
        barmode="overlay",
        title="Dwell time distribution (ms)",
        xaxis_title="Dwell (ms)", yaxis_title="Count", margin=dict(l=50,r=10,t=40,b=40)
    )
    st.plotly_chart(fig_dwell, use_container_width=True, config={"displaylogo": False})

    # ---------------- amplitude (pA) ----------------
    fig_amp = go.Figure()
    for rid, grp in ev_all.groupby("run_id"):
        fig_amp.add_trace(go.Histogram(
            x=grp["amp_pa"], name=str(rid), nbinsx=int(bins_amp), opacity=0.6
        ))
    fig_amp.update_layout(
        barmode="overlay",
        title="Mean blockade depth distribution (pA)",
        xaxis_title="Blockade depth (pA)", yaxis_title="Count", margin=dict(l=50,r=10,t=40,b=40)
    )
    st.plotly_chart(fig_amp, use_container_width=True, config={"displaylogo": False})

# -------------------------- quick links to reviewer --------------------------
def _open_in_reviewer(run_id: str):
    try:
        st.query_params.update({"run_id": run_id})
    except Exception:
        st.experimental_set_query_params(run_id=run_id)
    st.session_state["review_run_id"] = run_id
    try:
        st.switch_page("pages/4_Event_Reviewer.py")
    except Exception:
        pass  # ok on older Streamlit; user can click sidebar link

st.markdown("### Open selected runs in Reviewer")
cols = st.columns(min(4, len(sel_ids)))
for i, rid in enumerate(sel_ids):
    with cols[i % len(cols)]:
        if st.button(f"🖼️ Open Reviewer: {rid}", key=f"open_rev_{rid}", use_container_width=True):
            _open_in_reviewer(str(rid))

st.markdown("---")
# -------------------------- export selected metadata --------------------------
if st.button("⬇️ Export selected metadata (.csv)"):
    out_cols = have_cols + (["notes"] if "notes" in sel_df.columns else [])
    out = sel_df[out_cols].copy()
    st.download_button(
        "Download comparison_metadata.csv",
        data=out.to_csv(index=False),
        file_name="comparison_metadata.csv",
        mime="text/csv",
        use_container_width=True
    )
