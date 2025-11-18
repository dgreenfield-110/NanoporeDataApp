# pages/2_Experiment_Explorer.py
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.data_io import list_experiments
from lib.viz import trace_view

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Experiment Explorer", page_icon="🔎", layout="wide")
st.title("🔎 Experiment Explorer")

# --------------------------------------------------------------------------------------
# UI helpers
# --------------------------------------------------------------------------------------
def enlarge_plotly(fig, base=18, title=22, legend=16, tick=None, margin=(70, 40, 80, 50)):
    if tick is None:
        tick = max(10, base - 2)
    l, r, t, b = margin
    fig.update_layout(
        font=dict(size=base),
        title=dict(font=dict(size=title)),
        legend=dict(font=dict(size=legend)),
        margin=dict(l=l, r=r, t=t, b=b),
    )
    fig.update_xaxes(title_font=dict(size=base), tickfont=dict(size=tick))
    fig.update_yaxes(title_font=dict(size=base), tickfont=dict(size=tick))
    if getattr(fig.layout, "annotations", None):
        for ann in fig.layout.annotations:
            if getattr(ann, "text", None):
                ann.font.size = title
    return fig

st.markdown("""
<style>
  html, body, [class*="css"] { font-size: 17px; }
  .stMetric { font-size: 16px !important; }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------------------------------
# Load catalog
# --------------------------------------------------------------------------------------
df = list_experiments()
if df is None or df.empty:
    st.info("Catalog is empty. Ingest some runs in the Upload & Ingest page.")
    st.stop()

# --------------------------------------------------------------------------------------
# Normalization helpers (voltage, counts, basic columns)
# --------------------------------------------------------------------------------------
# Parse voltage from: "_90mV" or "... 90 mV ..."
_VOLTS_SUFFIX = re.compile(r'_(\d{2,3})mV(?:\b|$)', re.IGNORECASE)
_VOLTS_FREE   = re.compile(r'(?<!\d)(\d{1,3})\s*m[vV]\b')

def parse_voltage_anywhere(*texts) -> float | np.nan:
    blob = " ".join(str(t) for t in texts if t)
    m = _VOLTS_SUFFIX.search(blob) or _VOLTS_FREE.search(blob)
    return float(m.group(1)) if m else np.nan

def _ensure_cols(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    # run_id & title fallbacks
    if "run_id" not in out.columns:
        rid = out.get("title", pd.Series(index=out.index, dtype=object)).copy()
        if "trace_path" in out.columns:
            rid = rid.fillna(out["trace_path"].map(lambda p: Path(str(p)).stem))
        out["run_id"] = rid.fillna("run").astype(str)
    if "title" not in out.columns:
        out["title"] = out["run_id"].astype(str)

    # created_at
    if "created_at" not in out.columns:
        out["created_at"] = pd.to_datetime("now")

    # common text cols
    for c in ("operator", "analyte", "pore", "detector", "notes"):
        if c not in out.columns:
            out[c] = ""

    # voltage: prefer canonical measurement_voltage_mV; else legacy voltage_mV; else parse from names
    mv = pd.to_numeric(out.get("measurement_voltage_mV"), errors="coerce")
    lv = pd.to_numeric(out.get("voltage_mV"), errors="coerce")
    v_guess = [
        parse_voltage_anywhere(row.get("title",""), row.get("events_path",""), row.get("trace_path",""))
        for _, row in out.iterrows()
    ]
    out["voltage_mV"] = mv.fillna(lv).fillna(pd.Series(v_guess, index=out.index)).astype("Int64")

    # events count (n_events)
    if "n_events" not in out.columns:
        def _count_events(p):
            try:
                if isinstance(p, str) and p and Path(p).exists():
                    # fast path if pyarrow is present
                    try:
                        import pyarrow.parquet as pq  # type: ignore
                        return int(pq.ParquetFile(str(p)).metadata.num_rows)
                    except Exception:
                        return int(pd.read_parquet(Path(p)).shape[0])
            except Exception:
                pass
            return 0
        out["n_events"] = out.get("events_path", pd.Series(index=out.index, dtype=object)).map(_count_events)

    # baseline if missing
    if "baseline_pa" not in out.columns:
        out["baseline_pa"] = np.nan

    return out

df = _ensure_cols(df)

# --------------------------------------------------------------------------------------
# Sidebar filters
# --------------------------------------------------------------------------------------
with st.sidebar:
    q = st.text_input("Search title / operator / analyte", value="")
    min_events = st.number_input("Min events", value=0, step=1)
    det_opts = sorted([d for d in df.get("detector", pd.Series(dtype=object)).dropna().unique() if d])
    det_sel = st.multiselect("Detector", det_opts, [])
    st.markdown("---")
    st.caption("Quick Annotation Potential (selected run)")
    ap_min_dw = st.number_input("Min dwell (ms)", 0.0, 1e6, 0.2, 0.1, key="exp_ap_min_dw")
    ap_max_dw = st.number_input("Max dwell (ms, 0=∞)", 0.0, 1e6, 10.0, 0.1, key="exp_ap_max_dw")
    ap_min_amp = st.number_input("Min mean depth (pA)", 0.0, 1e6, 2.0, 0.1, key="exp_ap_min_amp")
    st.markdown("---")
    if st.button("🔁 Clear cached data for Explorer"):
        st.cache_data.clear()
        st.rerun()

# --------------------------------------------------------------------------------------
# Filter view
# --------------------------------------------------------------------------------------
view = df.copy()
if q:
    ql = q.lower()
    def _match(row):
        return any(ql in str(row.get(col, "")).lower() for col in ("title","operator","analyte"))
    view = view[view.apply(_match, axis=1)]
if det_sel:
    view = view[view["detector"].isin(det_sel)]
view = view[view["n_events"] >= int(min_events)]
view = view.sort_values("created_at", ascending=False).reset_index(drop=True)

# Keep only existing traces
if "trace_path" in view.columns:
    def _ok_trace(p):
        try: return isinstance(p, str) and p and Path(p).exists()
        except Exception: return False
    view = view[view["trace_path"].map(_ok_trace)].reset_index(drop=True)

if view.empty:
    st.info("No matching runs / missing trace files.")
    st.stop()

# --------------------------------------------------------------------------------------
# Top table
# --------------------------------------------------------------------------------------
cols_want = ["run_id","title","created_at","operator","analyte","detector","n_events","voltage_mV"]
cols_have = [c for c in cols_want if c in view.columns]
st.dataframe(
    view[cols_have].rename(columns={"n_events":"events"}),
    use_container_width=True,
    height=260
)

# --------------------------------------------------------------------------------------
# Quick preview
# --------------------------------------------------------------------------------------
def _ensure_time_and_current_cols(df_trace: pd.DataFrame) -> pd.DataFrame:
    out = df_trace.copy()
    if "Time[s]" not in out.columns:
        if "t" in out.columns: out = out.rename(columns={"t": "Time[s]"})
        elif "Time (s)" in out.columns: out = out.rename(columns={"Time (s)": "Time[s]"})
        else: out = out.rename(columns={out.columns[0]: "Time[s]"})
    if ("I_filt" not in out.columns) and ("current_pA" not in out.columns):
        for cand in ("current_pA", "current", "I", "current_nA"):
            if cand in out.columns:
                out = out.rename(columns={cand: "current_pA"})
                break
    return out

opts = list(dict.fromkeys(view["run_id"].astype(str).tolist()))
sel = st.selectbox("Select run", options=opts, index=0, key="explorer_sel_run")
row_match = view.loc[view["run_id"].astype(str) == str(sel)]
row = row_match.iloc[0] if not row_match.empty else view.iloc[0]

trace_path = Path(str(row.get("trace_path", "")))
events_path = Path(str(row.get("events_path", ""))) if pd.notna(row.get("events_path", "")) else None
if not trace_path.exists():
    st.error(f"Trace parquet not found on disk:\n{trace_path}")
    st.stop()

try:
    tdf = pd.read_parquet(trace_path)
    tdf = _ensure_time_and_current_cols(tdf)
except Exception as e:
    st.error(f"Failed to load trace: {e}")
    st.stop()

ev = None
if events_path and events_path.exists():
    try:
        ev = pd.read_parquet(events_path)
    except Exception as e:
        st.warning(f"Failed to load events parquet: {e}")

with st.sidebar:
    view_len = st.slider("View window length (s)", 1.0, 60.0, 8.0, step=1.0, key="expl_viewlen")
    max_pts  = st.slider("Max plot points (window)", 10_000, 120_000, 40_000, step=5_000, key="expl_maxpts")

tmin = float(pd.to_numeric(tdf["Time[s]"], errors="coerce").min())
tmax = float(pd.to_numeric(tdf["Time[s]"], errors="coerce").max())
key = "explorer_view"
if key not in st.session_state:
    st.session_state[key] = (tmin, min(tmin + view_len, tmax))
v0, v1 = st.slider("View range (s)", min_value=tmin, max_value=tmax,
                   value=st.session_state[key], step=max((tmax - tmin)/1000.0, 0.001), key="expl_viewrng")
if abs((v1 - v0) - view_len) > 1e-9:
    v1 = min(v0 + view_len, tmax)
st.session_state[key] = (v0, v1)

cur_col = "current_pA" if "current_pA" in tdf.columns else ("I_filt" if "I_filt" in tdf.columns else None)
baseline_pa = float(np.nanmedian(pd.to_numeric(tdf[cur_col], errors="coerce"))) if cur_col else np.nan

title = f"{row.get('title','(untitled)')} — {row.get('detector') or 'no events'}"
fig = trace_view(tdf, ev, baseline_pa=baseline_pa, view=(v0, v1), max_points=max_pts, title=title)
enlarge_plotly(fig, base=18, title=22, legend=16)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# --------------------------------------------------------------------------------------
# Quick annotation potential (selected run)
# --------------------------------------------------------------------------------------
st.markdown("#### 🧮 Annotation potential (selected run)")
if ev is None or len(ev) == 0:
    st.info("Selected run has no events parquet on disk.")
else:
    evp = ev.copy()
    # dwell: ALWAYS prefer end - start; else duration_s; else dwell_s
    if {"start_time","end_time"}.issubset(evp.columns):
        evp["dwell_s"] = (pd.to_numeric(evp["end_time"], errors="coerce")
                          - pd.to_numeric(evp["start_time"], errors="coerce"))
    elif "duration_s" in evp.columns:
        evp["dwell_s"] = pd.to_numeric(evp["duration_s"], errors="coerce")
    elif "dwell_s" in evp.columns:
        evp["dwell_s"] = pd.to_numeric(evp["dwell_s"], errors="coerce")

    # mean blockade (pA)
    def _coerce_amp(df: pd.DataFrame, baseline_pa_val: float) -> pd.Series:
        if "blockade_depth_mean_pA" in df.columns:
            return pd.to_numeric(df["blockade_depth_mean_pA"], errors="coerce")
        if {"event_mean_pA", "baseline_pA"}.issubset(df.columns):
            return pd.to_numeric(df["baseline_pA"], errors="coerce") - pd.to_numeric(df["event_mean_pA"], errors="coerce")
        if "event_mean_pA" in df.columns:
            return float(baseline_pa_val) - pd.to_numeric(df["event_mean_pA"], errors="coerce")
        if "delta_pA" in df.columns:
            return -pd.to_numeric(df["delta_pA"], errors="coerce")
        return pd.Series(np.nan, index=df.index)

    evp["dwell_ms"] = pd.to_numeric(evp["dwell_s"], errors="coerce") * 1000.0
    evp["blockade_depth_mean_pA"] = _coerce_amp(evp, baseline_pa)

    mask = pd.Series(True, index=evp.index)
    if ap_min_dw > 0: mask &= evp["dwell_ms"] >= ap_min_dw
    if ap_max_dw > 0: mask &= evp["dwell_ms"] <= ap_max_dw
    if ap_min_amp > 0: mask &= evp["blockade_depth_mean_pA"] >= ap_min_amp
    good = evp[mask].dropna(subset=["dwell_ms","blockade_depth_mean_pA"])

    cA, cB, cC = st.columns(3)
    cA.metric("Total", f"{len(evp):,}")
    cB.metric("Meets thresholds", f"{len(good):,}", f"{(len(good)/len(evp)*100):.1f}%")
    cC.metric("Baseline (pA)", f"{baseline_pa:.2f}")

# --------------------------------------------------------------------------------------
# Simple navigation to reviewer
# --------------------------------------------------------------------------------------
c_go, _ = st.columns([1,3])
with c_go:
    if st.button("🖼️ Open in Event Reviewer", use_container_width=True, key="open_reviewer_btn"):
        try:
            st.query_params.update({"run_id": sel})
        except Exception:
            st.experimental_set_query_params(run_id=sel)
        st.session_state["review_run_id"] = sel
        try:
            st.switch_page("pages/4_Event_Reviewer.py")
        except Exception:
            pass

# --------------------------------------------------------------------------------------
# Shared plotting helper: robust log-binned histogram drawn with Bars
# --------------------------------------------------------------------------------------
def _add_log_binned_hist(fig, series_ms, bins, name, legendgroup, row, col, showlegend=False):
    dm = pd.to_numeric(series_ms, errors="coerce")
    dm = dm[np.isfinite(dm) & (dm > 0)]
    if dm.empty:
        return
    lo = float(np.nanquantile(dm, 0.01))
    hi = float(np.nanquantile(dm, 0.99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(dm.min()), float(dm.max())
        if hi <= lo:
            return
    edges = np.logspace(np.log10(lo), np.log10(hi), int(bins) + 1)
    counts, _ = np.histogram(dm, bins=edges)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths  = edges[1:] - edges[:-1]
    fig.add_trace(
        go.Bar(
            x=centers, y=counts, width=widths,
            name=name, legendgroup=legendgroup,
            opacity=0.55, showlegend=showlegend,
            hovertemplate="dwell=%{x:.3g} ms<br>count=%{y}<extra>"+str(name)+"</extra>"
        ),
        row=row, col=col
    )

# --------------------------------------------------------------------------------------
# Aggregator 1: across selected runs, grouped by *voltage*
# (Fixes dwell by always preferring end_time - start_time)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _collect_events_for_runs(view_df: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    frames = []
    for rid in runs:
        rrow = view_df.loc[view_df["run_id"].astype(str) == str(rid)]
        if rrow.empty:
            continue
        rr = rrow.iloc[0]
        evp = rr.get("events_path")
        if not isinstance(evp, str) or not Path(evp).exists():
            continue
        try:
            e = pd.read_parquet(evp)
        except Exception:
            continue
        if e is None or e.empty:
            continue

        # voltage
        v_mV = rr.get("voltage_mV")
        if pd.isna(v_mV):
            v_mV = parse_voltage_anywhere(rr.get("title",""), rr.get("events_path",""), rr.get("trace_path",""))

        # baseline per event (fallback to run-level)
        base_ev = pd.to_numeric(e.get("baseline_pA"), errors="coerce")
        if base_ev.isna().all():
            base_ev = pd.Series(float(rr.get("baseline_pa", np.nan)), index=e.index)

        # amplitude
        if "blockade_depth_mean_pA" in e.columns:
            amp = pd.to_numeric(e["blockade_depth_mean_pA"], errors="coerce")
        elif "delta_pA" in e.columns:
            amp = -pd.to_numeric(e["delta_pA"], errors="coerce")
        elif "event_mean_pA" in e.columns:
            amp = base_ev - pd.to_numeric(e["event_mean_pA"], errors="coerce")
        else:
            continue

        # dwell (ALWAYS prefer end - start)
        if {"start_time","end_time"}.issubset(e.columns):
            dwell_s = (pd.to_numeric(e["end_time"], errors="coerce")
                       - pd.to_numeric(e["start_time"], errors="coerce"))
        elif "duration_s" in e.columns:
            dwell_s = pd.to_numeric(e["duration_s"], errors="coerce")
        elif "dwell_s" in e.columns:
            dwell_s = pd.to_numeric(e["dwell_s"], errors="coerce")
        else:
            continue

        tidy = pd.DataFrame({
            "run_id": str(rid),
            "title": rr.get("title", str(rid)),
            "voltage_mV": v_mV,
            "dwell_ms": dwell_s * 1000.0,
            "amp_norm": amp / base_ev.replace(0, np.nan)
        })
        frames.append(tidy)

    if not frames:
        return pd.DataFrame(columns=["run_id","title","voltage_mV","dwell_ms","amp_norm"])

    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["dwell_ms","amp_norm"])
    out = out[(out["dwell_ms"] > 0.05) & (out["amp_norm"] >= 0) & (out["amp_norm"] <= 1.5)]
    out["voltage_mV"] = pd.to_numeric(out["voltage_mV"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["voltage_mV"]); out["voltage_mV"] = out["voltage_mV"].astype(int)
    return out

# --------------------------------------------------------------------------------------
# Aggregator 2: across selected runs, grouped by *detector*
# (Fixes dwell by always preferring end_time - start_time)
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _collect_events_for_runs_by_detector(view_df: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    frames = []
    for rid in runs:
        rrow = view_df.loc[view_df["run_id"].astype(str) == str(rid)]
        if rrow.empty:
            continue
        rr = rrow.iloc[0]
        evp = rr.get("events_path")
        if not isinstance(evp, str) or not Path(evp).exists():
            continue
        try:
            e = pd.read_parquet(evp)
        except Exception:
            continue
        if e is None or e.empty:
            continue

        # baseline per event (fallback to run-level)
        base_ev = pd.to_numeric(e.get("baseline_pA"), errors="coerce")
        if base_ev.isna().all():
            base_ev = pd.Series(float(rr.get("baseline_pa", np.nan)), index=e.index)

        # amplitude
        if "blockade_depth_mean_pA" in e.columns:
            amp = pd.to_numeric(e["blockade_depth_mean_pA"], errors="coerce")
        elif "delta_pA" in e.columns:
            amp = -pd.to_numeric(e["delta_pA"], errors="coerce")
        elif "event_mean_pA" in e.columns:
            amp = base_ev - pd.to_numeric(e["event_mean_pA"], errors="coerce")
        else:
            continue

        # dwell (ALWAYS prefer end - start)
        if {"start_time","end_time"}.issubset(e.columns):
            dwell_s = (pd.to_numeric(e["end_time"], errors="coerce")
                       - pd.to_numeric(e["start_time"], errors="coerce"))
        elif "duration_s" in e.columns:
            dwell_s = pd.to_numeric(e["duration_s"], errors="coerce")
        elif "dwell_s" in e.columns:
            dwell_s = pd.to_numeric(e["dwell_s"], errors="coerce")
        else:
            continue

        frames.append(pd.DataFrame({
            "detector": (rr.get("detector") or "unknown").lower(),
            "dwell_ms": dwell_s * 1000.0,
            "amp_norm": amp / base_ev.replace(0, np.nan),
        }))

    if not frames:
        return pd.DataFrame(columns=["detector","dwell_ms","amp_norm"])

    out = pd.concat(frames, ignore_index=True)
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["dwell_ms","amp_norm"])
    out = out[(out["dwell_ms"] > 0.05) & (out["amp_norm"] >= 0) & (out["amp_norm"] <= 1.5)]
    out["detector"] = out["detector"].astype(str)
    return out

# --------------------------------------------------------------------------------------
# Across-run analysis (by voltage)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Across-run analysis (by voltage)")

with st.expander("Build figure from selected runs (by voltage)", expanded=True):
    default_runs = [sel] if "sel" in locals() else []
    picks = st.multiselect(
        "Select runs to include",
        list(view["run_id"].astype(str)),
        default=default_runs,
        key="byvoltage_picks"
    )
    if not picks:
        st.info("Select one or more runs.")
        st.stop()

    chosen = view[view["run_id"].astype(str).isin(picks)]
    if chosen.empty:
        st.info("No usable runs found for the current selection.")
        st.stop()

    # Compact listing of the runs included
    def _fast_count_rows(parquet_path: str | Path) -> int:
        try:
            import pyarrow.parquet as pq  # type: ignore
            return int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)
        except Exception:
            try:
                return int(len(pd.read_parquet(parquet_path)))
            except Exception:
                return 0

    lines = []
    for _, rr in chosen.iterrows():
        rid = str(rr.get("run_id"))
        title = str(rr.get("title"))
        v = rr.get("voltage_mV")
        vtxt = f"{int(v)} mV" if pd.notna(v) else "—"
        n = _fast_count_rows(rr.get("events_path","")) if rr.get("events_path") else 0
        lines.append(f"- **{rid}** — {title} ({vtxt}; n={n})")
    st.markdown("**Runs included:**\n" + "\n".join(lines))

    c1, c2, _ = st.columns([1,1,3])
    with c1:
        amp_bins = st.slider("Amplitude bins", min_value=5, max_value=80, value=25, step=1,
                             help="Fewer bins produce wider bars.")
    with c2:
        dwell_bins = st.slider("Dwell bins (log-x)", min_value=5, max_value=80, value=22, step=1,
                               help="Fewer bins produce wider bars. Still plotted on log-x.")

    data = _collect_events_for_runs(view, picks)
    if data.empty:
        st.info("No usable events found for the selected runs.")
        st.stop()

    volts = sorted(data["voltage_mV"].unique())

    fig2 = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Normalized amplitude vs dwell (log-x)",
            "Amplitude histogram (normalized)",
            "Dwell histogram (ms, log-x)",
            "Mean dwell vs voltage (ms)"
        )
    )

    # Scatter per voltage
    for v in volts:
        dv = data[data["voltage_mV"] == v]
        fig2.add_trace(
            go.Scattergl(
                x=dv["dwell_ms"], y=dv["amp_norm"], mode="markers",
                name=f"{int(v)} mV (n={len(dv)})", legendgroup=f"v{v}",
                opacity=0.6, marker={"size":5}
            ),
            row=1, col=1
        )
    fig2.update_xaxes(type="log", title_text="Dwell (ms)", row=1, col=1)
    fig2.update_yaxes(title_text="Normalized jump amplitude", row=1, col=1)

    # Optional overall OLS line on log10(dwell)
    x = np.log10(pd.to_numeric(data["dwell_ms"], errors="coerce").to_numpy(dtype=float))
    y = pd.to_numeric(data["amp_norm"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        A = np.vstack([x[mask], np.ones(mask.sum())]).T
        m, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        xs = np.linspace(x[mask].min(), x[mask].max(), 200)
        fig2.add_trace(
            go.Scatter(x=10**xs, y=m*xs + b, mode="lines", name="OLS on log10(dwell)"),
            row=1, col=1
        )

    # Amplitude histogram per voltage
    for v in volts:
        dv = data[data["voltage_mV"] == v]
        fig2.add_trace(
            go.Histogram(
                x=dv["amp_norm"], name=f"{int(v)} mV", legendgroup=f"v{v}",
                opacity=0.55, bingroup="amp", nbinsx=int(amp_bins)
            ),
            row=1, col=2
        )
    fig2.update_xaxes(title_text="Normalized jump amplitude", row=1, col=2)
    fig2.update_yaxes(title_text="Count", row=1, col=2)

    # Dwell histogram (log-x) using pre-binned Bars
    for v in volts:
        dv = data[data["voltage_mV"] == v]
        _add_log_binned_hist(
            fig2, dv["dwell_ms"], bins=int(dwell_bins),
            name=f"{int(v)} mV", legendgroup=f"v{v}",
            row=2, col=1, showlegend=False
        )
    fig2.update_xaxes(type="log", title_text="Dwell (ms)", row=2, col=1)
    fig2.update_yaxes(title_text="Count", rangemode="tozero", row=2, col=1)

    # Mean dwell vs voltage (95% CI)
    agg = (data.groupby("voltage_mV")
                .agg(n=("dwell_ms","size"),
                     mean_ms=("dwell_ms","mean"),
                     std_ms=("dwell_ms","std"))
                .reset_index())
    agg["sem_ms"] = agg["std_ms"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95_ms"] = 1.96 * agg["sem_ms"]
    fig2.add_trace(
        go.Scatter(
            x=agg["voltage_mV"], y=agg["mean_ms"],
            error_y=dict(type="data", array=agg["ci95_ms"], visible=True),
            mode="markers+lines", name="Mean ±95% CI"
        ),
        row=2, col=2
    )
    fig2.update_xaxes(title_text="Voltage (mV)", row=2, col=2)
    fig2.update_yaxes(title_text="Dwell time (ms)", row=2, col=2)

    fig2.update_layout(barmode="overlay", height=720, legend_traceorder="grouped",
                       margin=dict(l=40, r=20, t=60, b=40))

    st.plotly_chart(fig2, use_container_width=True, config={"displaylogo": False})

# --------------------------------------------------------------------------------------
# Across-run analysis (by detector)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Across-run analysis (by detector)")

with st.expander("Build figure from selected runs (by detector)", expanded=True):
    default_runs = [sel] if "sel" in locals() else []
    picks_det = st.multiselect(
        "Select runs to include",
        list(view["run_id"].astype(str)),
        default=default_runs, key="bydet_picks"
    )
    if not picks_det:
        st.info("Select one or more runs.")
        st.stop()

    data_det = _collect_events_for_runs_by_detector(view, picks_det)
    if data_det.empty:
        st.info("No usable events found for the selected runs.")
        st.stop()

    # Summary header with voltage and experiment titles
    counts_by_det = data_det.groupby("detector").size().to_dict()
    meta_cols = ["run_id", "title", "detector", "voltage_mV", "events_path"]
    sel_meta = view.copy()
    sel_meta["run_id"] = sel_meta["run_id"].astype(str)
    sel_meta = sel_meta[sel_meta["run_id"].isin(picks_det)][meta_cols].reset_index(drop=True)

    def _fast_count_rows(parquet_path: str | Path) -> int:
        try:
            import pyarrow.parquet as pq  # type: ignore
            return int(pq.ParquetFile(str(parquet_path)).metadata.num_rows)
        except Exception:
            try:
                return int(len(pd.read_parquet(parquet_path)))
            except Exception:
                return 0

    sel_meta["n"] = [
        _fast_count_rows(p) if isinstance(p, (str, Path)) and str(p) else 0
        for p in sel_meta["events_path"].fillna("")
    ]

    vols = sorted({int(v) for v in sel_meta["voltage_mV"].dropna().astype(int).tolist()})
    volt_text = ""
    if len(vols) == 1:
        volt_text = f" ({vols[0]} mV)"
    elif len(vols) > 1:
        volt_text = " (" + ", ".join(f"{v} mV" for v in vols) + ")"

    detector_summary = ", ".join([f"**{k}** (n={v})" for k, v in sorted(counts_by_det.items())])
    st.markdown(f"**Plotting{volt_text}:** {detector_summary}")

    if not sel_meta.empty:
        lines = []
        for _, r in sel_meta.iterrows():
            rid = str(r.get("run_id"))
            title_row = str(r.get("title"))
            det = str(r.get("detector") or "unknown")
            v = r.get("voltage_mV")
            n = int(r.get("n", 0))
            vtxt = f"{int(v)} mV" if pd.notna(v) else "—"
            lines.append(f"- **{rid}** — {title_row} (*{det}*, {vtxt}; n={n})")
        st.markdown("**Runs included:**\n" + "\n".join(lines))

    dets = [d for d in sorted(data_det["detector"].dropna().unique()) if d]

    figD = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Normalized amplitude vs dwell (log-x)",
            "Amplitude histogram (normalized)",
            "Dwell histogram (ms, log-x)",
            "Mean dwell vs detector (ms)"
        )
    )

    c1, c2, _ = st.columns([1,1,3])
    with c1:
        amp_bins_d = st.slider("Amplitude bins (by detector)", 5, 80, 25, 1, key="bydet_ampbins")
    with c2:
        dwell_bins_d = st.slider("Dwell bins (log-x, by detector)", 5, 80, 22, 1, key="bydet_dwells")

    # 1) Scatter
    for d in dets:
        gd = data_det[data_det["detector"] == d]
        figD.add_trace(
            go.Scattergl(
                x=gd["dwell_ms"], y=gd["amp_norm"], mode="markers",
                name=f"{d} (n={len(gd)})", legendgroup=f"det-{d}",
                opacity=0.6, marker={"size": 5}
            ),
            row=1, col=1
        )
    figD.update_xaxes(type="log", title_text="Dwell (ms)", row=1, col=1)
    figD.update_yaxes(title_text="Normalized jump amplitude", row=1, col=1)

    # Optional OLS
    x = np.log10(pd.to_numeric(data_det["dwell_ms"], errors="coerce").to_numpy(dtype=float))
    y = pd.to_numeric(data_det["amp_norm"], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() > 2:
        A = np.vstack([x[mask], np.ones(mask.sum())]).T
        m, b = np.linalg.lstsq(A, y[mask], rcond=None)[0]
        xs = np.linspace(x[mask].min(), x[mask].max(), 200)
        figD.add_trace(
            go.Scatter(x=10**xs, y=m*xs + b, mode="lines", name="OLS on log10(dwell)"),
            row=1, col=1
        )

    # 2) Amplitude histogram per detector
    for d in dets:
        gd = data_det[data_det["detector"] == d]
        figD.add_trace(
            go.Histogram(
                x=gd["amp_norm"], name=d, legendgroup=f"det-{d}",
                opacity=0.55, bingroup="amp", nbinsx=int(amp_bins_d)
            ),
            row=1, col=2
        )
    figD.update_xaxes(title_text="Normalized jump amplitude", row=1, col=2)
    figD.update_yaxes(title_text="Count", row=1, col=2)

    # 3) Dwell histogram per detector (log-x)
    for d in dets:
        _add_log_binned_hist(
            figD, data_det.loc[data_det["detector"] == d, "dwell_ms"],
            bins=int(dwell_bins_d),
            name=d, legendgroup=f"det-{d}",
            row=2, col=1, showlegend=False
        )
    figD.update_xaxes(type="log", title_text="Dwell (ms)", row=2, col=1)
    figD.update_yaxes(title_text="Count", rangemode="tozero", row=2, col=1)

    # 4) Mean dwell vs detector (95% CI)
    agg = (data_det.groupby("detector")
                    .agg(n=("dwell_ms", "size"),
                         mean_ms=("dwell_ms", "mean"),
                         std_ms=("dwell_ms", "std"))
                    .reset_index())
    agg["sem_ms"] = agg["std_ms"] / np.sqrt(agg["n"].clip(lower=1))
    agg["ci95_ms"] = 1.96 * agg["sem_ms"]
    figD.add_trace(
        go.Scatter(
            x=agg["detector"], y=agg["mean_ms"],
            error_y=dict(type="data", array=agg["ci95_ms"], visible=True),
            mode="markers+lines", name="Mean ±95% CI"
        ),
        row=2, col=2
    )
    figD.update_xaxes(title_text="Detector", row=2, col=2)
    figD.update_yaxes(title_text="Dwell time (ms)", row=2, col=2)

    figD.update_layout(
        barmode="overlay", height=720, legend_traceorder="grouped",
        margin=dict(l=40, r=20, t=60, b=40)
    )
    st.plotly_chart(figD, use_container_width=True, config={"displaylogo": False})
