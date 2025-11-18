# pages/5_Algo_Agreement_Prototype.py
from __future__ import annotations
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="Algorithm Agreement (Prototype)", page_icon="🧪", layout="wide")
st.title("🧪 Algorithm Agreement (Prototype)")
st.caption(
    "Upload a long-form trace (optional), a **ground truth** events file, and detector outputs "
    "(PELT & Hysteresis). Nothing is saved. All events are normalized to a single time base."
)

# --------------------------------------------------------------------------------------
# Trace loader
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_trace(uploaded, fs_fallback: float) -> pd.DataFrame:
    """
    Accept .npy (1D current, 2D [t, I], or dict with Time[s]/current) or .parquet.
    Returns DataFrame with columns: Time[s], current_pA, I_filt.
    """
    name = uploaded.name.lower()
    if name.endswith(".parquet"):
        df = pd.read_parquet(io.BytesIO(uploaded.getvalue()))
        # standardize time
        if "Time[s]" not in df.columns:
            if "t" in df.columns:
                df = df.rename(columns={"t": "Time[s]"})
            elif "Time (s)" in df.columns:
                df = df.rename(columns={"Time (s)": "Time[s]"})
            else:
                df = df.rename(columns={df.columns[0]: "Time[s]"})
        # standardize current
        cur = None
        for c in ("current_pA", "I_filt", "current", "I", "current_nA"):
            if c in df.columns:
                cur = c
                break
        if cur is None:
            if df.shape[1] == 1:
                df["current_pA"] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
                t = np.arange(len(df), dtype=float) / float(fs_fallback)
                df["Time[s]"] = t
            else:
                raise ValueError("Could not find a current column in the parquet.")
        else:
            df["current_pA"] = pd.to_numeric(df[cur], errors="coerce")
        if "I_filt" not in df:
            df["I_filt"] = df["current_pA"]
        return df[["Time[s]", "current_pA", "I_filt"]].copy()

    # .npy
    obj = np.load(io.BytesIO(uploaded.getvalue()), allow_pickle=True)
    t = None
    I = None
    if isinstance(obj, np.ndarray) and obj.dtype != object:
        if obj.ndim == 1:
            I = obj.astype(float)
            t = np.arange(I.size, dtype=float) / float(fs_fallback)
        elif obj.ndim == 2 and obj.shape[1] >= 2:
            t = obj[:, 0].astype(float)
            I = obj[:, 1].astype(float)
    else:
        try:
            d = obj.item() if hasattr(obj, "item") else dict(obj)
            for k in ("Time[s]", "t", "time", "Time (s)"):
                if k in d:
                    t = np.asarray(d[k], dtype=float)
                    break
            for k in ("current_pA", "I_filt", "current", "I"):
                if k in d:
                    I = np.asarray(d[k], dtype=float)
                    break
        except Exception:
            pass

    if I is None:
        raise ValueError("Unsupported .npy. Use 1D current; 2D [t, I]; or dict with Time[s] & current.")
    if t is None:
        t = np.arange(I.size, dtype=float) / float(fs_fallback)

    df = pd.DataFrame({"Time[s]": t, "current_pA": I})
    df["I_filt"] = df["current_pA"]
    return df

# --------------------------------------------------------------------------------------
# Robust event reader (handles tricky parquet/CSV) -> returns raw DataFrame
# --------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _read_events_file(uploaded) -> pd.DataFrame:
    if uploaded is None:
        return pd.DataFrame()
    raw = uploaded.getvalue()
    errors = []

    # 1) pandas + pyarrow
    try:
        return pd.read_parquet(io.BytesIO(raw), engine="pyarrow")
    except Exception as e:
        errors.append(f"pandas/pyarrow failed: {e}")

    # 2) pyarrow Table with no pandas metadata
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        tbl = pq.read_table(pa.BufferReader(raw), use_pandas_metadata=False)
        return tbl.to_pandas()
    except Exception as e:
        errors.append(f"pyarrow read_table failed: {e}")

    # 3) pyarrow row-group by row-group (skip corrupt groups)
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        pf = pq.ParquetFile(pa.BufferReader(raw))
        pieces = []
        for i in range(pf.num_row_groups):
            try:
                pieces.append(pf.read_row_group(i).to_pandas())
            except Exception as e_rg:
                errors.append(f"row-group {i} failed: {e_rg}")
        if pieces:
            return pd.concat(pieces, ignore_index=True)
        else:
            raise RuntimeError("All row groups failed.")
    except Exception as e:
        errors.append(f"pyarrow row-group fallback failed: {e}")

    # 4) fastparquet
    try:
        import fastparquet  # noqa
        return pd.read_parquet(io.BytesIO(raw), engine="fastparquet")
    except Exception as e:
        errors.append(f"fastparquet failed: {e}")

    # 5) CSV fallback
    try:
        return pd.read_csv(io.BytesIO(raw))
    except Exception as e:
        errors.append(f"csv fallback failed: {e}")

    st.error("Could not read events file.")
    st.caption("Errors:\n" + "\n".join(f"- {m}" for m in errors))
    return pd.DataFrame()

# --------------------------------------------------------------------------------------
# Normalization: map ANY schema to start_time/end_time/dwell_s on a single clock
# --------------------------------------------------------------------------------------
def _normalize_events(
    raw: pd.DataFrame,
    t_arr: np.ndarray | None,
    fs_for_indices: float | None,
    label: str = "events",
    prefer_indices: bool = False,
    fencepost_half_sample: bool = False,
    offset_s: float = 0.0,
) -> pd.DataFrame:
    """
    Produce columns: start_time (s), end_time (s), dwell_s (s).
    Priority (unless prefer_indices=True):
      - (start_time,end_time) or (start_s,end_s) or (start_ms,end_ms)/1000.
      - Else if has indices: (start_idx,end_idx) or (start,end):
            if t_arr is provided -> map via t_arr[idx] (best)
            elif fs_for_indices is provided -> idx / fs_for_indices (±½ sample if fencepost set)
    After conversion, apply a constant offset_s.
    """
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["start_time", "end_time", "dwell_s"])
    df = raw.copy()

    # Gather candidates
    has_seconds = False
    if not prefer_indices:
        if {"start_time", "end_time"}.issubset(df.columns):
            s = pd.to_numeric(df["start_time"], errors="coerce")
            e = pd.to_numeric(df["end_time"], errors="coerce")
            unit = "seconds (start_time/end_time)"
            has_seconds = True
        elif {"start_s", "end_s"}.issubset(df.columns):
            s = pd.to_numeric(df["start_s"], errors="coerce")
            e = pd.to_numeric(df["end_s"], errors="coerce")
            unit = "seconds (start_s/end_s)"
            has_seconds = True
        elif {"start_ms", "end_ms"}.issubset(df.columns):
            s = pd.to_numeric(df["start_ms"], errors="coerce") / 1000.0
            e = pd.to_numeric(df["end_ms"], errors="coerce") / 1000.0
            unit = "milliseconds → seconds"
            has_seconds = True
        else:
            s = e = None
    else:
        s = e = None

    if has_seconds:
        out = pd.DataFrame({"start_time": s, "end_time": e})
    else:
        # Indices variants
        idx_pairs = None
        if {"start_idx", "end_idx"}.issubset(df.columns):
            idx_pairs = (pd.to_numeric(df["start_idx"], errors="coerce"),
                         pd.to_numeric(df["end_idx"], errors="coerce"))
        elif {"start", "end"}.issubset(df.columns):
            idx_pairs = (pd.to_numeric(df["start"], errors="coerce"),
                         pd.to_numeric(df["end"], errors="coerce"))

        if idx_pairs is None:
            st.warning(f"'{label}': no recognizable time columns.")
            return pd.DataFrame(columns=["start_time", "end_time", "dwell_s"])

        s_idx, e_idx = idx_pairs

        # Prefer the trace's time-base if we have it
        if t_arr is not None and len(t_arr) > 2:
            s_i = np.clip(np.rint(s_idx).astype("Int64"), 0, len(t_arr) - 1)
            e_i = np.clip(np.rint(e_idx).astype("Int64"), 0, len(t_arr) - 1)
            s = pd.Series(t_arr[s_i.fillna(0).astype(int)].astype(float), index=df.index)
            e = pd.Series(t_arr[e_i.fillna(0).astype(int)].astype(float), index=df.index)
            unit = "sample indices → trace time"
            out = pd.DataFrame({"start_time": s, "end_time": e})
        elif fs_for_indices and fs_for_indices > 0:
            fs = float(fs_for_indices)
            half = (0.5 / fs) if fencepost_half_sample else 0.0
            s = s_idx / fs - half
            e = e_idx / fs + half
            unit = f"samples @ fs={fs:.3f} Hz → seconds" + (" (±½ sample)" if fencepost_half_sample else "")
            out = pd.DataFrame({"start_time": s, "end_time": e})
        else:
            st.warning(f"'{label}': only indices present but no trace/fs to convert to seconds.")
            return pd.DataFrame(columns=["start_time", "end_time", "dwell_s"])

    # Clean and offset
    out = out.replace([np.inf, -np.inf], np.nan).dropna()
    out = out[out["end_time"] > out["start_time"]].copy()
    if offset_s:
        out["start_time"] += float(offset_s)
        out["end_time"] += float(offset_s)
    out["dwell_s"] = (out["end_time"] - out["start_time"]).clip(lower=0)
    out = out.reset_index(drop=True)[["start_time", "end_time", "dwell_s"]]

    if len(out):
        med_ms = out["dwell_s"].median() * 1e3
        st.caption(
            f"Loaded **{label}** using **{unit}** — n={len(out):,}, "
            f"median dwell ≈ {med_ms:.2f} ms, offset={offset_s*1e3:.2f} ms"
        )

    return out

# --------------------------------------------------------------------------------------
# Matching & metrics
# --------------------------------------------------------------------------------------
def _merge_intervals(intervals):
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

def _total_len(intervals):
    return float(sum(max(0.0, e - s) for s, e in intervals))

def _intersections_len(A, B):
    i = j = 0
    inter = 0.0
    while i < len(A) and j < len(B):
        a1, a2 = A[i]
        b1, b2 = B[j]
        s = max(a1, b1)
        e = min(a2, b2)
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
    a["a_end"] = a["end_time"] + pad_s
    b["b_start"] = b["start_time"] + offsetB_s - pad_s
    b["b_end"] = b["end_time"] + offsetB_s + pad_s
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
        usedA.add(i)
        usedB.add(j)
        out.append((i, j, sA, eA, sB, eB, ov, un, iou))
    return pd.DataFrame(out, columns=["a_idx","b_idx","a_start","a_end","b_start","b_end","overlap_s","union_s","iou"])

def _compute_metrics(gt: pd.DataFrame, det: pd.DataFrame, pad: float, offset: float, iou_min: float):
    matches = _match_greedy_iou(gt, det, pad_s=pad, offsetB_s=offset)
    tp_tbl = matches.loc[matches["iou"] >= iou_min].copy()
    TP = int(len(tp_tbl))
    FN = int(len(gt) - TP)
    FP = int(len(det) - TP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(tp_tbl["iou"].mean()) if TP > 0 else 0.0
    A_union = _merge_intervals(list(zip(gt["start_time"], gt["end_time"])))
    B_union = _merge_intervals(list(zip(det["start_time"], det["end_time"])))
    dice_time = 0.0
    if A_union and B_union:
        inter_len = _intersections_len(A_union, B_union)
        A_len = _total_len(A_union)
        B_len = _total_len(B_union)
        dice_time = 2 * inter_len / (A_len + B_len) if (A_len + B_len) > 0 else 0.0
    return {
        "TP": TP,
        "FP": FP,
        "FN": FN,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_iou": mean_iou,
        "dice_time": dice_time,
        "tp_tbl": tp_tbl,
        "matches": matches,
    }

# Quick drift estimator (diagnostics only)
def _estimate_drift_s_per_s(a: pd.DataFrame, b: pd.DataFrame) -> float:
    if a.empty or b.empty:
        return 0.0
    A = a.sort_values("start_time").reset_index(drop=True)
    B = b.sort_values("start_time").reset_index(drop=True)
    j = 0
    pairs = []
    for _, r in A.iterrows():
        while j + 1 < len(B) and abs(B.loc[j + 1, "start_time"] - r["start_time"]) < abs(
            B.loc[j, "start_time"] - r["start_time"]
        ):
            j += 1
        if abs(B.loc[j, "start_time"] - r["start_time"]) < 0.2:
            pairs.append((r["start_time"], B.loc[j, "start_time"] - r["start_time"]))
    if len(pairs) < 5:
        return 0.0
    t = np.array([p[0] for p in pairs])
    dt = np.array([p[1] for p in pairs])
    X = np.vstack([t, np.ones_like(t)]).T
    alpha, _ = np.linalg.lstsq(X, dt, rcond=None)[0]
    return float(alpha)

# --------------------------------------------------------------------------------------
# Sidebar — uploads & controls
# --------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Uploads")
    trace_file = st.file_uploader("Trace (.npy or .parquet) — optional", type=["npy", "parquet"])
    fs_hz_ui = st.number_input("Sampling rate (Hz) for 1D .npy", min_value=1, value=10_000)

    st.markdown("---")
    gt_file = st.file_uploader("Ground truth events (.parquet/.csv)", type=["parquet", "csv"], key="gt_upl")
    pelt_file = st.file_uploader("PELT events (.parquet/.csv)", type=["parquet", "csv"], key="pelt_upl")
    hyst_file = st.file_uploader("Hysteresis events (.parquet/.csv)", type=["parquet", "csv"], key="hyst_upl")

    st.markdown("---")
    st.markdown("### When a file uses **indices** instead of seconds")
    events_fs = st.number_input("Default events sampling rate (Hz)", min_value=1, value=9765)
    st.caption("Used for index→seconds **only** when no trace timebase is available.")

    st.markdown("### Ground truth alignment help")
    gt_prefer_idx = st.checkbox("Force GT to use *indices* (ignore any start_s/end_s)", value=False)
    gt_fence = st.checkbox("Fencepost correction (±½ sample)", value=True)
    gt_fs_custom = st.number_input("GT fs for index→seconds (Hz)", min_value=1, value=events_fs, step=1)
    gt_offset_ms = st.slider(
        "GT constant offset (ms)", -20.0, 20.0, 0.0, 0.1, help="Applied after conversion. Positive shifts GT to the right."
    )

    st.markdown("---")
    st.markdown("### Matching parameters")
    iou_min = st.slider("Min IoU to count as TP", 0.0, 1.0, 0.2, 0.05)
    pad_s = st.slider("Pad each event (± s)", 0.0, 0.050, 0.005, 0.001)
    off_pelt = st.slider("Time offset for PELT vs GT (s)", -0.5, 0.5, 0.0, 0.001)
    off_hyst = st.slider("Time offset for Hysteresis vs GT (s)", -0.5, 0.5, 0.0, 0.001)
    show_tbl = st.checkbox("Show matched tables", True)

# --------------------------------------------------------------------------------------
# Load inputs
# --------------------------------------------------------------------------------------
trace_df = None
t_arr = None
if trace_file is not None:
    try:
        trace_df = _load_trace(trace_file, float(fs_hz_ui))
        t_arr = pd.to_numeric(trace_df["Time[s]"], errors="coerce").to_numpy(float)
        dur = float(t_arr[-1] - t_arr[0]) if t_arr.size else 0.0
        st.success(f"Trace loaded: {len(trace_df):,} samples; duration {dur:.2f} s")
    except Exception as e:
        st.warning(f"Trace load failed: {e}")

# read raw event tables
gt_raw = _read_events_file(gt_file) if gt_file else pd.DataFrame()
pelt_raw = _read_events_file(pelt_file) if pelt_file else pd.DataFrame()
hyst_raw = _read_events_file(hyst_file) if hyst_file else pd.DataFrame()

# Need at least GT + one detector
if gt_raw.empty:
    st.info("Upload a **ground truth** events file to begin.")
    st.stop()
if pelt_raw.empty and hyst_raw.empty:
    st.info("Upload at least one detector events file (PELT and/or Hysteresis).")
    st.stop()

# normalize all to the SAME clock
gt = _normalize_events(
    gt_raw,
    t_arr=t_arr,
    fs_for_indices=(None if t_arr is not None else float(gt_fs_custom)),
    label="ground truth",
    prefer_indices=bool(gt_prefer_idx),
    fencepost_half_sample=bool(gt_fence),
    offset_s=float(gt_offset_ms) / 1000.0,
)
pelt = _normalize_events(
    pelt_raw, t_arr=t_arr, fs_for_indices=(None if t_arr is not None else float(events_fs)), label="PELT"
)
hyst = _normalize_events(
    hyst_raw, t_arr=t_arr, fs_for_indices=(None if t_arr is not None else float(events_fs)), label="Hysteresis"
)

if t_arr is None:
    st.caption(
        "ℹ️ No trace provided — any index-based events were converted using "
        f"**{events_fs} Hz** (GT used **{gt_fs_custom} Hz** if forced to indices)."
    )

# --------------------------------------------------------------------------------------
# Quick counts
# --------------------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Ground truth", f"{len(gt):,} events")
c2.metric("PELT", f"{len(pelt):,}")
c3.metric("Hysteresis", f"{len(hyst):,}")

# Drift diagnostics (GT vs each detector)
diag_rows = []
if not pelt.empty:
    a = _estimate_drift_s_per_s(gt, pelt)
    diag_rows.append({"pair": "GT vs PELT", "drift (s/s)": f"{a:.3e}"})
if not hyst.empty:
    a = _estimate_drift_s_per_s(gt, hyst)
    diag_rows.append({"pair": "GT vs HYST", "drift (s/s)": f"{a:.3e}"})
if diag_rows:
    st.dataframe(pd.DataFrame(diag_rows), hide_index=True, use_container_width=True)

# --------------------------------------------------------------------------------------
# Agreement: PELT vs GT; Hysteresis vs GT
# --------------------------------------------------------------------------------------
results = {}
if not pelt.empty:
    results["PELT"] = _compute_metrics(gt, pelt, pad_s, off_pelt, iou_min)
if not hyst.empty:
    results["Hysteresis"] = _compute_metrics(gt, hyst, pad_s, off_hyst, iou_min)

# Table of metrics
rows = []
for name, r in results.items():
    rows.append(
        dict(
            detector=name,
            TP=r["TP"],
            FP=r["FP"],
            FN=r["FN"],
            precision=f"{r['precision']:.3f}",
            recall=f"{r['recall']:.3f}",
            f1=f"{r['f1']:.3f}",
            mean_iou=f"{r['mean_iou']:.3f}",
            dice_time=f"{r['dice_time']:.3f}",
        )
    )
st.markdown("### Summary metrics")
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# IoU histograms
fig = go.Figure()
for name, r in results.items():
    tp_tbl = r["tp_tbl"]
    if not tp_tbl.empty:
        fig.add_trace(go.Histogram(x=tp_tbl["iou"], nbinsx=30, name=name, opacity=0.65))
fig.update_layout(
    title="IoU distribution for true matches",
    xaxis_title="IoU",
    yaxis_title="Count",
    barmode="overlay",
    margin=dict(l=50, r=20, t=50, b=50),
)
st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

# Matched tables & downloads
if show_tbl:
    for name, r in results.items():
        st.markdown(f"#### Matched pairs — {name}")
        tp_tbl = r["tp_tbl"]
        if tp_tbl.empty:
            st.info(f"No true matches for {name} at IoU ≥ {iou_min}.")
            continue
        st.dataframe(
            tp_tbl[
                ["a_idx", "b_idx", "a_start", "a_end", "b_start", "b_end", "overlap_s", "union_s", "iou"]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            f"⬇️ Download matched_pairs_{name}.csv",
            data=tp_tbl.to_csv(index=False),
            file_name=f"matched_pairs_{name}.csv",
            mime="text/csv",
            use_container_width=True,
        )

# --------------------------------------------------------------------------------------
# Head-to-head: Hysteresis vs PELT (PELT treated as reference)
# --------------------------------------------------------------------------------------
if not pelt.empty and not hyst.empty:
    # relative offset between detectors (reuse your GT offsets)
    rel_off = float(off_hyst) - float(off_pelt)

    hh = _compute_metrics(pelt, hyst, pad=pad_s, offset=rel_off, iou_min=iou_min)

    st.markdown("### Head-to-head: Hysteresis vs PELT (PELT = reference)")
    st.caption(
        f"Relative offset used: HYST−PELT = {rel_off*1000:.2f} ms; padding ±{pad_s*1000:.1f} ms; IoU ≥ {iou_min:.2f}"
    )

    hh_rows = [
        dict(
            comparison="HYST vs PELT",
            TP=hh["TP"],
            FP=hh["FP"],
            FN=hh["FN"],
            precision=f"{hh['precision']:.3f}",
            recall=f"{hh['recall']:.3f}",
            f1=f"{hh['f1']:.3f}",
            mean_iou=f"{hh['mean_iou']:.3f}",
            dice_time=f"{hh['dice_time']:.3f}",
        )
    ]
    st.dataframe(pd.DataFrame(hh_rows), use_container_width=True, hide_index=True)

    # Optional: head-to-head IoU histogram (true matches only)
    if not hh["tp_tbl"].empty:
        fig_hh = go.Figure()
        fig_hh.add_trace(
            go.Histogram(x=hh["tp_tbl"]["iou"], nbinsx=30, name="HYST vs PELT", opacity=0.75)
        )
        fig_hh.update_layout(
            title="IoU (true matches) — HYST vs PELT",
            xaxis_title="IoU",
            yaxis_title="Count",
            margin=dict(l=50, r=20, t=50, b=50),
        )
        st.plotly_chart(fig_hh, use_container_width=True, config={"displaylogo": False})

# --------------------------------------------------------------------------------------
# Optional: quick trace preview with event overlays (clean legend, no per-rect labels)
# --------------------------------------------------------------------------------------
st.markdown("---")
st.subheader("Quick trace preview (optional)")
if trace_df is None:
    st.info("Upload a trace to preview overlays (not required for metrics).")
else:
    t = pd.to_numeric(trace_df["Time[s]"], errors="coerce").to_numpy(float)
    I = (trace_df["I_filt"] if "I_filt" in trace_df else trace_df["current_pA"]).to_numpy(float)
    tmin = float(np.nanmin(t))
    tmax = float(np.nanmax(t))
    view_len = st.slider("Preview window length (s)", 0.5, 30.0, 5.0, 0.5)
    v0 = st.slider("Window start (s)", tmin, max(tmax - view_len, tmin), tmin)
    v1 = min(v0 + view_len, tmax)

    m = (t >= v0) & (t <= v1)
    fig_p = go.Figure()
    fig_p.add_trace(go.Scattergl(x=t[m], y=I[m], mode="lines", name="trace", line=dict(width=1), showlegend=False))

    # --- legend swatches (clean, non-overlapping) ---
    GT_COLOR = "rgba(46,160,67,0.20)"
    PELT_COLOR = "rgba(31,119,180,0.20)"
    HYST_COLOR = "rgba(214,39,40,0.20)"
    RIM = dict(line_width=1, line_color="rgba(0,0,0,0.08)")

    fig_p.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=GT_COLOR), name="GT")
    )
    fig_p.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=PELT_COLOR), name="PELT")
    )
    fig_p.add_trace(
        go.Scatter(x=[None], y=[None], mode="markers", marker=dict(size=12, color=HYST_COLOR), name="HYST")
    )

    def _add_boxes(fig: go.Figure, df: pd.DataFrame, color: str, v0: float, v1: float) -> None:
        """Overlay shaded boxes; only draw the part falling inside [v0, v1]."""
        if df is None or df.empty:
            return
        dfv = df[(df["end_time"] >= v0) & (df["start_time"] <= v1)]
        for _, r in dfv.iterrows():
            s = float(max(r["start_time"], v0))
            e = float(min(r["end_time"], v1))
            if e <= s:
                continue
            fig.add_vrect(x0=s, x1=e, fillcolor=color, opacity=1.0, layer="below", **RIM)

    _add_boxes(fig_p, gt, GT_COLOR, v0, v1)
    _add_boxes(fig_p, pelt, PELT_COLOR, v0, v1)
    _add_boxes(fig_p, hyst, HYST_COLOR, v0, v1)

    fig_p.update_layout(
        title=f"Trace overlay [{v0:.3f}s → {v1:.3f}s] \n",
        xaxis_title="Time (s)",
        yaxis_title="Current (pA)",
        hovermode="x unified",
        margin=dict(l=50, r=20, t=40, b=40),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=.95, xanchor="center", x=0.5, itemsizing="trace"),
    )
    st.plotly_chart(fig_p, use_container_width=True, config={"displaylogo": False})
