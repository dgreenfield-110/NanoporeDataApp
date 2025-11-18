from __future__ import annotations
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def resolve_cols(df: pd.DataFrame):
    tcol = "Time[s]" if "Time[s]" in df.columns else ("Time (s)" if "Time (s)" in df.columns else ("t" if "t" in df.columns else None))
    icol = "I_filt" if "I_filt" in df.columns else ("current_pA" if "current_pA" in df.columns else None)
    if tcol is None or icol is None:
        raise KeyError("Need time in ['Time[s]','Time (s)','t'] and current in ['I_filt','current_pA'].")
    return tcol, icol

def minmax_downsample(t, y, max_points=50_000):
    t = np.asarray(t, float); y = np.asarray(y, float)
    n = y.size
    if n <= max_points or max_points <= 0: return t, y
    bins = np.linspace(t[0], t[-1], max_points + 1)
    idx = np.searchsorted(t, bins); idx[-1] = n
    t_out = []; y_out = []
    for i in range(max_points):
        a, b = idx[i], idx[i+1]
        if b <= a: continue
        seg = y[a:b]
        jmin = a + int(np.argmin(seg)); jmax = a + int(np.argmax(seg))
        if jmin < jmax:
            t_out.extend([t[jmin], t[jmax]]); y_out.extend([y[jmin], y[jmax]])
        else:
            t_out.extend([t[jmax], t[jmin]]); y_out.extend([y[jmax], y[jmin]])
    return np.asarray(t_out), np.asarray(y_out)

def ensure_event_times(events: pd.DataFrame, t_arr: np.ndarray) -> pd.DataFrame:
    ev = events.copy()
    if {"start_time","end_time"}.issubset(ev.columns):
        return ev
    if {"start_idx","end_idx"}.issubset(ev.columns):
        n = len(t_arr)
        st = np.clip(ev["start_idx"].to_numpy(int), 0, max(0, n-1))
        en = np.clip(ev["end_idx"].to_numpy(int)-1, 0, max(0, n-1))
        ev["start_time"] = t_arr[st]
        ev["end_time"]   = t_arr[en]
        ev["dwell_s"]    = ev["end_time"] - ev["start_time"]
        return ev
    raise KeyError("events must include start_time/end_time or start_idx/end_idx.")

def event_metrics(row, ta, Ia, pre_pad_s=0.010):
    i0 = np.searchsorted(ta, float(row.start_time))
    i1 = np.searchsorted(ta, float(row.end_time), side="right")
    seg = Ia[i0:i1]
    if seg.size == 0:
        return dict(baseline_pA=np.nan, amp_mean_pA=np.nan, amp_min_pA=np.nan, charge_deficit_pA_s=np.nan)
    fs  = 1.0 / np.median(np.diff(ta)) if len(ta) > 1 else np.nan
    pre_n = int(round(max(1, pre_pad_s * (fs if np.isfinite(fs) else 10000.0))))
    local_base = np.nanmedian(Ia[max(0, i0-pre_n):i0]) if i0 > 0 else np.nan
    ev_min = np.nanmin(seg); ev_mean = np.nanmean(seg)
    baseline = local_base if np.isfinite(local_base) else np.nanmedian(Ia)
    amp_mean = baseline - ev_mean; amp_min = baseline - ev_min
    cd = np.trapz(np.clip(baseline - seg, 0, None), ta[i0:i1]) if seg.size >= 2 else np.nan
    return dict(baseline_pA=float(baseline), amp_mean_pA=float(amp_mean), amp_min_pA=float(amp_min),
                charge_deficit_pA_s=float(cd))

def trace_view(work_df: pd.DataFrame, events_df: pd.DataFrame | None,
               baseline_pa: float | None, view: tuple[float,float],
               max_points: int = 50_000, title: str = "Trace preview"):
    tcol, icol = resolve_cols(work_df)
    t_all = work_df[tcol].to_numpy(float)
    I_all = work_df[icol].to_numpy(float)
    v0, v1 = float(view[0]), float(view[1])
    if v1 <= v0: v0, v1 = float(t_all.min()), float(t_all.min()) + 5.0
    m = (t_all >= v0) & (t_all <= v1)
    if not np.any(m):
        pad = 0.5 * (t_all.max() - t_all.min()) / 1000.0
        m = (t_all >= v0 - pad) & (t_all <= v1 + pad)
    t, I = t_all[m], I_all[m]
    td, Id = minmax_downsample(t, I, max_points=max_points)

    y_lo, y_hi = -1.0, 1.0
    if np.isfinite(Id).any():
        y_lo = float(np.nanquantile(Id, 0.01)); y_hi = float(np.nanquantile(Id, 0.99))
        if not (np.isfinite(y_lo) and np.isfinite(y_hi) and y_hi > y_lo):
            y_lo, y_hi = float(np.nanmin(Id)), float(np.nanmax(Id))

    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=td, y=Id, mode="lines", name="trace", line={"width": 1}))

    if events_df is not None and len(events_df):
        ev = events_df[["start_time","end_time"]].dropna().astype(float)
        ev = ev[(ev["end_time"] >= v0) & (ev["start_time"] <= v1)]
        if len(ev):
            y_level = y_lo + 0.02 * (y_hi - y_lo)
            xs, ys = [], []
            for st_, en_ in ev.itertuples(index=False):
                stt, enn = max(st_, v0), min(en_, v1)
                if enn > stt:
                    xs.extend([stt, enn, None]); ys.extend([y_level, y_level, None])
            if xs:
                fig.add_trace(go.Scattergl(x=np.array(xs), y=np.array(ys), mode="lines",
                                           line={"width": 6}, name="events", opacity=0.3, hoverinfo="skip"))

    if baseline_pa is not None and np.isfinite(baseline_pa):
        fig.add_hline(y=float(baseline_pa), line_dash="dash", line_color="green",
                      annotation_text=f"Baseline {baseline_pa:.1f} pA", annotation_position="bottom right")
    fig.update_layout(title=title, xaxis_title="Time (s)", yaxis_title="Current (pA)",
                      dragmode="pan", hovermode="x",
                      xaxis=dict(range=[v0, v1], showspikes=True),
                      yaxis=dict(showspikes=True),
                      uirevision="trace-preview-window",
                      margin=dict(l=50, r=10, t=40, b=40), showlegend=False)
    return fig

def event_window(ta, Ia, ev_row, pad_s, baseline_pa=None, title="Event preview"):
    t0 = float(ev_row.start_time) - pad_s
    t1 = float(ev_row.end_time)   + pad_s
    m  = (ta >= t0) & (ta <= t1)
    tx, Ix = ta[m], Ia[m]
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=tx, y=Ix, mode="lines", line={"width":1}, name="trace"))
    fig.add_vrect(x0=float(ev_row.start_time), x1=float(ev_row.end_time),
                  fillcolor="red", opacity=0.2, line_width=0)
    if baseline_pa is not None and np.isfinite(baseline_pa):
        fig.add_hline(y=float(baseline_pa), line_dash="dash", line_color="green",
                      annotation_text=f"Baseline {baseline_pa:.1f} pA", annotation_position="bottom right")
    fig.update_layout(
        title=title, xaxis_title="Time (s)", yaxis_title="Current (pA)",
        dragmode="pan", hovermode="x", uirevision="event-slideshow",
        margin=dict(l=50, r=10, t=40, b=40), showlegend=False,
    )
    return fig
