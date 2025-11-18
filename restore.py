# restore.py
# Rebuilds the entire app structure (files + folders).

from pathlib import Path
import textwrap

FILES = {
    ".gitignore": textwrap.dedent("""\
        __pycache__/
        .streamlit/secrets.toml
        .env
        .DS_Store
        data/
        */__pycache__/
        .pytest_cache/
        *.pyc
    """),

    "requirements.txt": textwrap.dedent("""\
        streamlit>=1.36
        numpy>=1.26
        pandas>=2.2
        plotly>=5.22
        duckdb>=1.0
        pyarrow>=17.0
        # Optional (only needed if you ever want ABF import again)
        # pyabf
    """),

    "README.md": textwrap.dedent("""\
        # Nanopore App (Offline Ingest)
        This app ingests **precomputed events** (PELT/Hysteresis) and a trace to enable fast preview,
        slideshow review, annotation export, and cataloging in DuckDB.

        ## Quickstart
        ```bash
        python restore.py
        pip install -r requirements.txt
        streamlit run Home.py
        ```

        ## Layout
        - `Home.py` — landing page with quick links
        - `pages/1_Upload_and_Ingest.py` — upload offline results (trace + events)
        - `pages/2_Experiment_Explorer.py` — browse the catalog; preview traces
        - `pages/3_Event_Reviewer.py` — deep-dive slideshow + annotation export
        - `lib/config.py` — storage paths (RUNS_DIR, DB)
        - `lib/data_io.py` — DuckDB catalog helpers
        - `lib/viz.py` — plotting + event utils

        ## Offline Inputs
        - Trace: `.npy` (1D current or 2D [t, I] or dict with `Time[s]`/`t` + `I_filt`/`current_pA`) or `.parquet`
        - Events: `.parquet` with columns either
          - `start_time`, `end_time` (seconds), or
          - `start_idx`, `end_idx` (sample indices; will be converted to time)
    """),

    ".streamlit/config.toml": textwrap.dedent("""\
        [theme]
        base="light"
        primaryColor="#2B6CB0"
        backgroundColor="#ffffff"
        secondaryBackgroundColor="#F6F9FC"
        textColor="#1A202C"
        font="sans serif"
    """),

    "Home.py": textwrap.dedent("""\
        import streamlit as st
        from lib.config import RUNS_DIR, DB_PATH
        from lib.data_io import init_catalog, count_experiments

        st.set_page_config(page_title="Nanopore App", page_icon="🧬", layout="wide")
        st.title("🧬 Nanopore App — Offline Workflow")

        init_catalog()  # ensure DB exists

        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown(
                "Use **Upload & Ingest** to add results you've computed offline "
                "(trace + events). Then explore and review events in the other pages."
            )
            st.markdown(f"**Runs directory:** `{RUNS_DIR}`  \n**Catalog:** `{DB_PATH}`")

        with c2:
            st.metric("Experiments in catalog", f"{count_experiments():,}")

        st.divider()
        st.page_link("pages/1_Upload_and_Ingest.py", label="📤 Upload & Ingest (offline)", icon="📤")
        st.page_link("pages/2_Experiment_Explorer.py", label="🔎 Experiment Explorer", icon="🔎")
        st.page_link("pages/3_Event_Reviewer.py", label="🖼️ Event Reviewer", icon="🖼️")

        st.caption("Tip: change paths in `lib/config.py`.")
    """),

    "lib/__init__.py": "",

    "lib/config.py": textwrap.dedent("""\
        from pathlib import Path

        ROOT = Path(__file__).resolve().parents[1]
        DATA_DIR = ROOT / "data"
        RUNS_DIR = DATA_DIR / "runs"
        DB_PATH  = DATA_DIR / "catalog.duckdb"

        # Ensure directories exist
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        RUNS_DIR.mkdir(parents=True, exist_ok=True)
    """),

    "lib/data_io.py": textwrap.dedent("""\
        from __future__ import annotations
        from pathlib import Path
        import json
        import duckdb
        import pandas as pd
        from datetime import datetime
        from .config import DB_PATH, RUNS_DIR

        SCHEMA_SQL = '''
        CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT PRIMARY KEY,
            title TEXT,
            created_at TIMESTAMP,
            operator TEXT,
            pore TEXT,
            analyte TEXT,
            concentration_nM DOUBLE,
            denaturant TEXT,
            voltage_mV INTEGER,
            fs_hz DOUBLE,
            detector TEXT,
            cutoff_hz DOUBLE,
            baseline_pa DOUBLE,
            notes TEXT,
            trace_path TEXT,
            events_path TEXT,
            n_events INTEGER,
            extra JSON
        );
        '''

        def _connect():
            con = duckdb.connect(DB_PATH)
            return con

        def init_catalog():
            con = _connect()
            con.execute(SCHEMA_SQL)
            con.close()

        def count_experiments() -> int:
            init_catalog()
            con = _connect()
            try:
                n = con.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
            except Exception:
                n = 0
            finally:
                con.close()
            return int(n)

        def new_experiment_row(meta: dict, trace_path: Path | None, events_path: Path | None, stats: dict | None) -> dict:
            run_id = meta.get("title") or meta.get("run_id") or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            trace_rel  = str(Path(trace_path).resolve()) if trace_path else None
            events_rel = str(Path(events_path).resolve()) if events_path else None
            row = {
                "run_id": run_id,
                "title": meta.get("title", run_id),
                "created_at": datetime.utcnow(),
                "operator": meta.get("operator", ""),
                "pore": meta.get("pore", ""),
                "analyte": meta.get("analyte", ""),
                "concentration_nM": float(meta.get("concentration_nM", 0.0) or 0.0),
                "denaturant": meta.get("denaturant", ""),
                "voltage_mV": int(meta.get("voltage_mV", 0) or 0),
                "fs_hz": float(meta.get("fs_hz", 0.0) or 0.0),
                "detector": meta.get("detector", ""),
                "cutoff_hz": float(meta.get("cutoff_hz", float('nan')) if meta.get("cutoff_hz") is not None else float('nan')),
                "baseline_pa": float(meta.get("baseline_pa", float('nan')) if meta.get("baseline_pa") is not None else float('nan')),
                "notes": meta.get("notes", ""),
                "trace_path": trace_rel,
                "events_path": events_rel,
                "n_events": int(meta.get("n_events", 0)),
                "extra": json.dumps(stats or {}),
            }
            return row

        def insert_experiment(row: dict):
            init_catalog()
            con = _connect()
            # upsert
            cols = ", ".join(row.keys())
            placeholders = ", ".join(["?"] * len(row))
            updates = ", ".join([f"{k}=excluded.{k}" for k in row.keys() if k != "run_id"])
            sql = f"INSERT INTO experiments ({cols}) VALUES ({placeholders}) ON CONFLICT(run_id) DO UPDATE SET {updates}"
            con.execute(sql, list(row.values()))
            con.close()

        def list_experiments() -> pd.DataFrame:
            init_catalog()
            con = _connect()
            df = con.execute("SELECT * FROM experiments ORDER BY created_at DESC").df()
            con.close()
            return df

        def get_experiment(run_id: str) -> dict | None:
            init_catalog()
            con = _connect()
            row = con.execute("SELECT * FROM experiments WHERE run_id = ?", [run_id]).fetchone()
            con.close()
            if row is None:
                return None
            cols = [c[0] for c in row._description]
            return dict(zip(cols, row))
    """),

    "lib/viz.py": textwrap.dedent("""\
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
    """),

    "pages/1_Upload_and_Ingest.py": textwrap.dedent("""\
        from __future__ import annotations
        import io
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import streamlit as st

        from lib.config import RUNS_DIR
        from lib.data_io import new_experiment_row, insert_experiment
        from lib.viz import trace_view, ensure_event_times, event_window

        st.set_page_config(page_title="Upload & Ingest (Offline)", page_icon="📤", layout="wide")
        st.title("📤 Upload & Ingest — Offline results")

        # ---------- Sidebar ----------
        with st.sidebar:
            st.markdown("**Metadata**")
            operator   = st.text_input("Operator", value="")
            pore       = st.text_input("Pore", value="")
            analyte    = st.text_input("Analyte", value="")
            conc       = st.number_input("Concentration (mg/mL)", min_value=0.0, step=1.0, value=0.0)
            denat      = st.text_input("Electrolyte Solution", value="")
            voltage    = st.number_input("Voltage (mV)", step=10, value=60)
            notes      = st.text_area("Notes", value="")

            st.markdown("---")
            fs_hz_ui   = st.number_input("Sampling rate (Hz) — used if trace lacks time", step=1, value=10_000)

            st.markdown("---")
            view_len   = st.slider("View window length (s)", 1.0, 60.0, 8.0, step=1.0)
            max_pts    = st.slider("Max plot points (window)", 10_000, 120_000, 40_000, step=5_000)
            show_preview = st.checkbox("Show preview", value=True)

        c1, c2, c3 = st.columns(3)
        with c1:
            trace_file = st.file_uploader("Trace (.npy preferred, also supports .parquet)", type=["npy","parquet"])
        with c2:
            events_pelt_file = st.file_uploader("Events — PELT (.parquet)", type=["parquet"])
        with c3:
            events_hyst_file = st.file_uploader("Events — Hysteresis (.parquet)", type=["parquet"])

        if trace_file is None:
            st.info("Upload a trace to begin.")
            st.stop()

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

        # Load inputs
        try:
            work_df = _load_trace(trace_file, fs_fallback=float(fs_hz_ui))
            st.success(f"Trace loaded: {len(work_df):,} samples")
        except Exception as e:
            st.exception(e)
            st.stop()

        baseline_pa = float(np.nanmedian(work_df["current_pA"])) if "current_pA" in work_df else float(np.nan)
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
            active_detector = st.radio("Active events to preview/save", available, horizontal=True, index=0)
            events_df = events_pelt if active_detector == "pelt" else events_hyst

        # Ensure event times
        try:
            t_arr = work_df["Time[s]"].to_numpy(float)
            if not events_df.empty:
                events_df = ensure_event_times(events_df, t_arr)
                if "dwell_s" not in events_df.columns:
                    events_df["dwell_s"] = events_df["end_time"] - events_df["start_time"]
        except Exception as e:
            st.warning(f"Event time mapping issue: {e}")
            events_df = pd.DataFrame(columns=["start_time","end_time","dwell_s"])

        # ---------- Preview ----------
        if show_preview:
            try:
                tmin, tmax = float(work_df["Time[s]"].min()), float(work_df["Time[s]"].max())
                key = "trace_view"
                if key not in st.session_state:
                    st.session_state[key] = (tmin, min(tmin + view_len, tmax))
                v0, v1 = st.slider("View range (s)", min_value=tmin, max_value=tmax,
                                   value=st.session_state[key],
                                   step=max((tmax - tmin) / 1000.0, 0.001))
                if abs((v1 - v0) - view_len) > 1e-9:
                    v1 = min(v0 + view_len, tmax)
                st.session_state[key] = (v0, v1)

                fig = trace_view(
                    work_df, events_df if not events_df.empty else None,
                    baseline_pa=baseline_pa, view=(v0, v1), max_points=max_pts,
                    title=f"Preview ({active_detector or 'no events'})"
                )
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            except Exception as e:
                st.warning(f"Preview failed: {e}")

        st.write("**Preview (trace)**", work_df.head())
        if not events_df.empty:
            st.write(f"**Preview (events — {active_detector})**", events_df.head())

        # ---------- Slideshow ----------
        st.markdown("---")
        st.subheader("Event slideshow")

        with st.sidebar:
            st.markdown("---")
            st.markdown("**Slideshow preview**")
            pad_s = st.slider("Context padding (± s)", 0.2, 5.0, 2.0, 0.1)
            min_dwell = st.number_input("Min dwell (ms)", min_value=0.0, value=0.0, step=0.1)
            max_dwell = st.number_input("Max dwell (ms, 0=∞)", min_value=0.0, value=0.0, step=0.1)
            min_amp   = st.number_input("Min mean depth (pA)", value=0.0, step=0.5)
            sort_by   = st.selectbox("Sort events by", ["start_time","dwell_s"], index=0)
            sample_n  = st.slider("Sample (0=all)", 0, 2000, 0, step=50)

        t_arr = work_df["Time[s]"].to_numpy(float)
        I_arr = (work_df["I_filt"] if "I_filt" in work_df else work_df["current_pA"]).to_numpy(float)
        ev_view = events_df.copy()
        if not ev_view.empty:
            if min_dwell > 0: ev_view = ev_view[ev_view["dwell_s"] >= (min_dwell/1000.0)]
            if max_dwell > 0: ev_view = ev_view[ev_view["dwell_s"] <= (max_dwell/1000.0)]
            if "blockade_depth_mean_pA" in ev_view.columns and min_amp > 0:
                ev_view = ev_view[ev_view["blockade_depth_mean_pA"] >= min_amp]
            if sort_by in ev_view.columns:
                ev_view = ev_view.sort_values(sort_by).reset_index(drop=True)
            if sample_n and len(ev_view) > sample_n:
                ev_view = (ev_view.sample(sample_n, random_state=0)
                                   .sort_values(sort_by).reset_index(drop=True))

        n_ev = len(ev_view)
        st.write(f"**Slideshow:** {n_ev} event(s) after filters)")
        if n_ev == 0:
            st.info("No events to show.")
        else:
            if "slideshow_idx" not in st.session_state:
                st.session_state["slideshow_idx"] = 0
            idx = st.session_state["slideshow_idx"]

            c1, c2, c3, c4 = st.columns([1,1,2,2])
            with c1:
                if st.button("⟵ Prev", use_container_width=True):
                    st.session_state["slideshow_idx"] = (idx - 1) % n_ev
                    st.rerun()
            with c2:
                if st.button("Next ⟶", use_container_width=True):
                    st.session_state["slideshow_idx"] = (idx + 1) % n_ev
                    st.rerun()
            with c3:
                st.write("")
                idx = st.number_input("Event #", min_value=0, max_value=max(n_ev-1, 0),
                                      value=idx, step=1, key="idx_num")
                st.session_state["slideshow_idx"] = int(idx)
            with c4:
                show_metrics = st.checkbox("Show metrics", value=True)

            row = ev_view.iloc[int(st.session_state["slideshow_idx"])]
            m = _event_metrics = __import__("lib.viz", fromlist=["event_metrics"]).event_metrics
            metrics = m(row, t_arr, I_arr, pre_pad_s=0.010)
            baseline_for_plot = metrics.get("baseline_pA", baseline_pa)

            fig_event = event_window(
                t_arr, I_arr, row, pad_s=pad_s, baseline_pa=baseline_for_plot,
                title=f"Event {st.session_state['slideshow_idx']}/{n_ev-1} | "
                    f"start={row.start_time:.3f}s  dwell={row.dwell_s:.4f}s"
            )
            st.plotly_chart(fig_event, use_container_width=True, config={"displaylogo": False})

            if show_metrics:
                colA, colB, colC, colD = st.columns(4)
                colA.metric("Dwell (ms)", f"{row.dwell_s*1e3:.2f}")
                colB.metric("Mean depth (pA)", f"{metrics['amp_mean_pA']:.2f}")
                colC.metric("Min depth (pA)", f"{metrics['amp_min_pA']:.2f}")
                colD.metric("Charge deficit (pA·s)", f"{metrics['charge_deficit_pA_s']:.3g}")

            if "labels" not in st.session_state:
                st.session_state["labels"] = {}
            ev_id = int(row.get("event_id", st.session_state["slideshow_idx"]))
            c1, c2, c3 = st.columns([1,1,2])
            if c1.button("✅ Accept", use_container_width=True):
                st.session_state["labels"][ev_id] = "accept"
            if c2.button("🗑️ Reject", use_container_width=True):
                st.session_state["labels"][ev_id] = "reject"
            with c3:
                if st.button("⬇️ Export annotations CSV", use_container_width=True):
                    lab = pd.DataFrame([{"event_id": k, "label": v} for k, v in st.session_state["labels"].items()])
                    st.download_button("Download labels.csv", data=lab.to_csv(index=False),
                                       file_name="event_labels.csv", mime="text/csv")

        # ---------- Persist ----------
        st.markdown("---")
        save_ok = st.button("💾 Save this run to catalog", type="primary")
        if save_ok:
            try:
                run_id = Path(trace_file.name).stem
                run_dir = RUNS_DIR / run_id
                run_dir.mkdir(parents=True, exist_ok=True)

                trace_path  = run_dir / "trace_work.parquet"
                work_df.to_parquet(trace_path)

                active_path = run_dir / "events.parquet"
                if not events_df.empty:
                    events_df.to_parquet(active_path)

                # Stash originals if both provided
                if events_pelt_file is not None:
                    (run_dir / "events_pelt.parquet").write_bytes(events_pelt_file.getvalue())
                if events_hyst_file is not None:
                    (run_dir / "events_hysteresis.parquet").write_bytes(events_hyst_file.getvalue())

                # Minimal stats for catalog
                fs_est = float(1.0/np.median(np.diff(work_df["Time[s]"])) if len(work_df)>1 else fs_hz_ui)
                stats = dict(
                    fs_hz=round(fs_est, 6),
                    baseline_pA=float(baseline_pa) if np.isfinite(baseline_pa) else float("nan"),
                    sigma_open=float(sigma_open) if np.isfinite(sigma_open) else float("nan"),
                    detector=(active_detector or "none"),
                    n_events=int(len(events_df)) if not events_df.empty else 0,
                    source="offline-import",
                )

                meta = dict(
                    title=run_id, operator=operator, pore=pore, analyte=analyte,
                    concentration_nM=conc, denaturant=denat, voltage_mV=int(voltage),
                    fs_hz=float(stats["fs_hz"]),
                    detector=stats["detector"],
                    cutoff_hz=float("nan"),
                    baseline_pa=float(stats.get("baseline_pA", float("nan"))),
                    notes=notes,
                    n_events=int(stats["n_events"]),
                )

                row = new_experiment_row(meta, trace_path, active_path if not events_df.empty else None, stats)
                insert_experiment(row)
                st.success("Ingested and saved ✔")
            except Exception as e:
                st.exception(e)
    """),

    "pages/2_Experiment_Explorer.py": textwrap.dedent("""\
        from __future__ import annotations
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import streamlit as st

        from lib.data_io import list_experiments
        from lib.viz import trace_view

        st.set_page_config(page_title="Experiment Explorer", page_icon="🔎", layout="wide")
        st.title("🔎 Experiment Explorer")

        df = list_experiments()
        if df.empty:
            st.info("Catalog is empty. Ingest some runs in the Upload & Ingest page.")
            st.stop()

        with st.sidebar:
            q = st.text_input("Search title / operator / analyte", value="")
            min_events = st.number_input("Min events", value=0, step=1)
            det_sel = st.multiselect("Detector", sorted([d for d in df["detector"].dropna().unique() if d]), [])

        view = df.copy()
        if q:
            ql = q.lower()
            view = view[view.apply(lambda r: ql in str(r["title"]).lower()
                                            or ql in str(r["operator"]).lower()
                                            or ql in str(r["analyte"]).lower(), axis=1)]
        if det_sel:
            view = view[view["detector"].isin(det_sel)]
        view = view[view["n_events"] >= int(min_events)]
        view = view.sort_values("created_at", ascending=False).reset_index(drop=True)

        st.dataframe(view[["run_id","title","created_at","operator","analyte","detector","n_events"]], use_container_width=True)

        st.markdown("---")
        st.subheader("Quick preview")
        if view.empty:
            st.info("No matching runs.")
            st.stop()

        sel = st.selectbox("Select run", options=list(view["run_id"]), index=0)
        row = view[view["run_id"] == sel].iloc[0]
        trace_path = Path(row["trace_path"])
        events_path = Path(row["events_path"]) if row["events_path"] else None

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception as e:
            st.error(f"Failed to load trace: {e}")
            st.stop()

        ev = None
        if events_path and events_path.exists():
            try:
                ev = pd.read_parquet(events_path)
            except Exception as e:
                st.warning(f"Failed to load events: {e}")

        with st.sidebar:
            view_len = st.slider("View window length (s)", 1.0, 60.0, 8.0, step=1.0)
            max_pts  = st.slider("Max plot points (window)", 10_000, 120_000, 40_000, step=5_000)
        tmin, tmax = float(tdf["Time[s]"].min()), float(tdf["Time[s]"].max())
        key = "explorer_view"
        if key not in st.session_state:
            st.session_state[key] = (tmin, min(tmin + view_len, tmax))
        v0, v1 = st.slider("View range (s)", min_value=tmin, max_value=tmax, value=st.session_state[key],
                           step=max((tmax - tmin) / 1000.0, 0.001))
        if abs((v1 - v0) - view_len) > 1e-9:
            v1 = min(v0 + view_len, tmax)
        st.session_state[key] = (v0, v1)

        base = float(np.nanmedian(tdf.get("current_pA", tdf.get("I_filt")).to_numpy()))
        fig = trace_view(tdf, ev, baseline_pa=base, view=(v0, v1), max_points=max_pts,
                         title=f"{row['title']} — {row['detector'] or 'no events'}")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.page_link("pages/3_Event_Reviewer.py", label="Open in Event Reviewer", icon="🖼️",
                     help="Open the run in the reviewer page for a slideshow/annotation view.",
                     args={"run_id": sel})
    """),

    "pages/3_Event_Reviewer.py": textwrap.dedent("""\
        from __future__ import annotations
        from pathlib import Path
        import numpy as np
        import pandas as pd
        import streamlit as st

        from lib.data_io import get_experiment
        from lib.viz import ensure_event_times, event_window, event_metrics

        st.set_page_config(page_title="Event Reviewer", page_icon="🖼️", layout="wide")
        st.title("🖼️ Event Reviewer")

        qp = st.query_params.to_dict()
        run_id = qp.get("run_id") or st.text_input("Enter run_id to load", value="")
        if not run_id:
            st.info("Provide a run_id (from Explorer) using the link or text box above.")
            st.stop()

        row = get_experiment(run_id)
        if not row:
            st.error(f"Run '{run_id}' not found in catalog.")
            st.stop()

        st.caption(f"**Loaded:** {row['title']}  |  {row['detector']}  |  n_events={row['n_events']}")

        trace_path = Path(row["trace_path"])
        events_path = Path(row["events_path"]) if row["events_path"] else None

        try:
            tdf = pd.read_parquet(trace_path)
        except Exception as e:
            st.error(f"Failed to load trace: {e}")
            st.stop()

        if not events_path or not events_path.exists():
            st.info("This run has no events file.")
            st.stop()

        try:
            ev = pd.read_parquet(events_path)
        except Exception as e:
            st.error(f"Failed to load events: {e}")
            st.stop()

        # Prepare arrays
        t_arr = tdf["Time[s]"].to_numpy(float)
        I_arr = (tdf["I_filt"] if "I_filt" in tdf else tdf["current_pA"]).to_numpy(float)
        ev = ensure_event_times(ev, t_arr).copy()
        if "dwell_s" not in ev.columns:
            ev["dwell_s"] = ev["end_time"] - ev["start_time"]

        with st.sidebar:
            st.markdown("**Filters**")
            pad_s = st.slider("Context padding (± s)", 0.1, 5.0, 2.0, 0.1)
            min_dwell = st.number_input("Min dwell (ms)", min_value=0.0, value=0.0, step=0.1)
            max_dwell = st.number_input("Max dwell (ms, 0=∞)", min_value=0.0, value=0.0, step=0.1)
            min_amp   = st.number_input("Min mean depth (pA)", value=0.0, step=0.5)
            sort_by   = st.selectbox("Sort by", ["start_time","dwell_s"], index=0)
            sample_n  = st.slider("Sample (0=all)", 0, 2000, 0, step=50)

        ev_view = ev.copy()
        if min_dwell > 0: ev_view = ev_view[ev_view["dwell_s"] >= (min_dwell/1000.0)]
        if max_dwell > 0: ev_view = ev_view[ev_view["dwell_s"] <= (max_dwell/1000.0)]
        if "blockade_depth_mean_pA" in ev_view.columns and min_amp > 0:
            ev_view = ev_view[ev_view["blockade_depth_mean_pA"] >= min_amp]
        if sort_by in ev_view.columns:
            ev_view = ev_view.sort_values(sort_by).reset_index(drop=True)
        if sample_n and len(ev_view) > sample_n:
            ev_view = (ev_view.sample(sample_n, random_state=0)
                               .sort_values(sort_by).reset_index(drop=True))

        st.write(f"**Events after filters:** {len(ev_view)}")
        if ev_view.empty:
            st.info("No events after filters.")
            st.stop()

        if "rev_idx" not in st.session_state:
            st.session_state["rev_idx"] = 0
        idx = st.session_state["rev_idx"]

        c1, c2, c3, c4 = st.columns([1,1,2,2])
        with c1:
            if st.button("⟵ Prev", use_container_width=True):
                st.session_state["rev_idx"] = (idx - 1) % len(ev_view); st.rerun()
        with c2:
            if st.button("Next ⟶", use_container_width=True):
                st.session_state["rev_idx"] = (idx + 1) % len(ev_view); st.rerun()
        with c3:
            st.write("")
            idx = st.number_input("Event #", min_value=0, max_value=max(len(ev_view)-1, 0),
                                  value=idx, step=1, key="rev_idx_num")
            st.session_state["rev_idx"] = int(idx)
        with c4:
            show_metrics = st.checkbox("Show metrics", value=True)

        row_ev = ev_view.iloc[int(st.session_state["rev_idx"])]
        metrics = event_metrics(row_ev, t_arr, I_arr, pre_pad_s=0.010)
        baseline_for_plot = metrics.get("baseline_pA", float(np.nanmedian(I_arr)))

        fig_event = event_window(
            t_arr, I_arr, row_ev, pad_s=pad_s, baseline_pa=baseline_for_plot,
            title=f"{row['title']} | Event {st.session_state['rev_idx']}/{len(ev_view)-1} "
                  f"| start={row_ev.start_time:.3f}s dwell={row_ev.dwell_s:.4f}s"
        )
        st.plotly_chart(fig_event, use_container_width=True, config={"displaylogo": False})

        if show_metrics:
            colA, colB, colC, colD = st.columns(4)
            colA.metric("Dwell (ms)", f"{row_ev.dwell_s*1e3:.2f}")
            colB.metric("Mean depth (pA)", f"{metrics['amp_mean_pA']:.2f}")
            colC.metric("Min depth (pA)", f"{metrics['amp_min_pA']:.2f}")
            colD.metric("Charge deficit (pA·s)", f"{metrics['charge_deficit_pA_s']:.3g}")

        # Simple labels (per run viewer)
        if "labels" not in st.session_state: st.session_state["labels"] = {}
        ev_id = int(row_ev.get("event_id", st.session_state["rev_idx"]))
        c1, c2, c3 = st.columns([1,1,2])
        if c1.button("✅ Accept", use_container_width=True):
            st.session_state["labels"][ev_id] = "accept"
        if c2.button("🗑️ Reject", use_container_width=True):
            st.session_state["labels"][ev_id] = "reject"
        with c3:
            if st.button("⬇️ Export annotations CSV", use_container_width=True):
                lab = pd.DataFrame([{"event_id": k, "label": v} for k, v in st.session_state["labels"].items()])
                st.download_button("Download labels.csv", data=lab.to_csv(index=False),
                                   file_name=f"{run_id}_labels.csv", mime="text/csv")
    """),
}

def write_files(files: dict):
    for rel, content in files.items():
        p = Path(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(content)
    print(f"✅ Wrote {len(files)} files.")
    print("Next:")
    print("  pip install -r requirements.txt")
    print("  streamlit run Home.py")

if __name__ == "__main__":
    write_files(FILES)

