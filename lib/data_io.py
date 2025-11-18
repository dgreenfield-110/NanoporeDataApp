# lib/data_io.py
from __future__ import annotations

import os
import re
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# ---------------------------------------------------------------------
# DB basics
# ---------------------------------------------------------------------

DEFAULT_DB = "data/catalog.db"


def get_conn(db_path: str = DEFAULT_DB) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    return conn


def _ensure_column(cursor: sqlite3.Cursor, name: str, decl: str) -> None:
    cursor.execute("PRAGMA table_info(experiments)")
    cols = [r[1] for r in cursor.fetchall()]
    if name not in cols:
        cursor.execute(f"ALTER TABLE experiments ADD COLUMN {name} {decl}")


def init_catalog(db_path: str = DEFAULT_DB) -> None:
    """
    Initialize the 'experiments' table and migrate missing columns.
    Safe to call every run.
    """
    conn = get_conn(db_path)
    cur = conn.cursor()

    # Base table (create once)
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT UNIQUE,
            title TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            operator TEXT,
            analyte TEXT,
            pore TEXT,
            detector TEXT,
            concentration_nM REAL,
            denaturant TEXT,
            voltage_mV INTEGER,
            measurement_voltage_mV INTEGER,
            fs_hz REAL,
            cutoff_hz REAL,
            baseline_pa REAL,
            notes TEXT,
            trace_path TEXT,
            events_path TEXT,
            stats_json TEXT
        )
        """
    )
    conn.commit()

    # Ensure newer columns exist (no-op if already present)
    _ensure_column(cur, "detector", "TEXT")
    _ensure_column(cur, "voltage_mV", "INTEGER")
    _ensure_column(cur, "measurement_voltage_mV", "INTEGER")
    _ensure_column(cur, "baseline_pa", "REAL")
    _ensure_column(cur, "trace_path", "TEXT")
    _ensure_column(cur, "events_path", "TEXT")
    _ensure_column(cur, "stats_json", "TEXT")

    conn.commit()
    conn.close()


def count_experiments(db_path: str = DEFAULT_DB) -> int:
    try:
        conn = get_conn(db_path)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM experiments")
        (n,) = cur.fetchone()
        conn.close()
        return int(n)
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return 0


def reset_catalog(hard: bool = False, db_path: str = DEFAULT_DB) -> None:
    """
    If hard=False: delete all rows from experiments.
    If hard=True:  delete the whole DB file.
    """
    if hard:
        try:
            if Path(db_path).exists():
                os.remove(db_path)
        except Exception:
            pass
        return

    init_catalog(db_path)  # make sure it exists
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM experiments")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------
# Voltage parsing / helpers
# ---------------------------------------------------------------------

# Matches a single token like "60mV" (case-insensitive)
_VOLTTOK_RE = re.compile(r"^(-?\d{1,3})\s*m[vV]$")
# Matches anywhere in a string "... 60mV ..."
#_MV_ANY_RE = re.compile(r"(\d{1,3})\s*m[vV]\b")
# Matches anywhere in a string "... 60mV ..." even if followed by "_" etc.
_MV_ANY_RE = re.compile(r"(?<!\d)(-?\d{1,3})\s*m[vV](?=_|\b|$)")


def _as_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        xi = int(float(x))
        return xi
    except Exception:
        return None


#def _guess_voltage_from_strings(*parts: Any) -> Optional[int]:
# Matches a single token like "60mV" (case-insensitive)
_VOLTTOK_RE = re.compile(r"^(-?\d{1,3})\s*m[vV]$")
# Matches anywhere in a string "... 60mV ..."
_MV_ANY_RE = re.compile(r"(\d{1,3})\s*m[vV]\b")

def _extract_last_mv_token(s: Any) -> Optional[int]:
    """
    Extract the last 'NNmV' token from a *single* string (path/name/etc.).
    Looks first for tokenized matches (split on [_/\\ ]), then falls back
    to an anywhere-in-string search.
    """
    if not s:
        return None
    s = str(s)

    # 1) Tokenized match
    last = None
    for tok in re.split(r"[ _/\\]", s):
        m = _VOLTTOK_RE.match(tok)
        if m:
            last = int(m.group(1))
    if last is not None:
        return last

    # 2) Fallback: anywhere in the string (keep the last occurrence)
    last_any = None
    for m in _MV_ANY_RE.finditer(s):
        last_any = int(m.group(1))
    return last_any


def _guess_voltage_prioritized(
    run_id: Any = None,
    title: Any = None,
    events_path: Any = None,
    trace_path: Any = None,
) -> Optional[int]:
    """
    Prefer voltage tokens that come from events_path (or its filename)
    and run_id/title. Only if none are found, fall back to trace_path.
    """
    for s in (events_path, run_id, title):
        v = _extract_last_mv_token(s)
        if v is not None:
            return v
    return _extract_last_mv_token(trace_path)



def voltage_from_run_id(run_id_or_name: str | None) -> Optional[int]:
    """
    Public helper: parse trailing voltage from a run_id or filename stem.
    """
    if not run_id_or_name:
        return None
    stem = Path(str(run_id_or_name)).stem
    # Prefer the last token like 60mV
    return _guess_voltage_from_strings(stem)


# ---------------------------------------------------------------------
# Event counting from Parquet (fast if pyarrow is available)
# ---------------------------------------------------------------------

def _n_events_from_parquet(parquet_path: str | Path) -> int:
    try:
        p = Path(parquet_path)
        if not p.exists():
            return 0
        # Try fast path via pyarrow, fall back to pandas
        try:
            import pyarrow.parquet as pq  # type: ignore
            pf = pq.ParquetFile(str(p))
            return int(pf.metadata.num_rows)
        except Exception:
            df = pd.read_parquet(p)
            return int(len(df))
    except Exception:
        return 0


# ---------------------------------------------------------------------
# Row building / CRUD
# ---------------------------------------------------------------------
def get_next_run_id(db_path: str = DEFAULT_DB) -> str:
    """
    Return the next numeric run_id as a string: '1', '2', ...
    We consider only rows whose run_id is purely numeric.
    """
    init_catalog(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    # Only look at run_id values that are all digits
    cur.execute("SELECT MAX(CAST(run_id AS INTEGER)) FROM experiments WHERE run_id GLOB '[0-9]*'")
    row = cur.fetchone()
    n = int(row[0]) if row and row[0] is not None else 0
    conn.close()
    return str(n + 1)



def new_experiment_row(meta: Dict[str, Any],
                       trace_path: str | Path,
                       events_path: str | Path,
                       stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """
    Build a row dict ready for insert/update. We try to be very tolerant
    about missing fields and backfill voltage from strings.
    """
    init_catalog()  # ensure DB/schema

    trace_path = str(trace_path) if trace_path is not None else None
    events_path = str(events_path) if events_path is not None else None

    run_id = meta.get("run_id") or meta.get("title") or Path(str(events_path or trace_path or "run")).stem
    run_id = str(run_id)

    # Title: keep whatever caller gave, or mirror run_id
    title = meta.get("title") or run_id

    # Detector normalized values
    detector = meta.get("detector")
    if detector:
        detector = str(detector).lower()

    # Voltage precedence: explicit > measurement_voltage_mV > guess-from-strings
    v_explicit = _as_int(meta.get("voltage_mV"))
    v_meas = _as_int(meta.get("measurement_voltage_mV"))
    v_guess = _guess_voltage_prioritized(run_id=run_id, title=title,
                                         events_path=events_path, trace_path=trace_path)


    voltage_mV = v_explicit if v_explicit is not None else (v_meas if v_meas is not None else v_guess)
    measurement_voltage_mV = v_meas if v_meas is not None else voltage_mV

    row = dict(
        run_id=run_id,
        title=title,
        created_at=meta.get("created_at") or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        operator=meta.get("operator"),
        analyte=meta.get("analyte"),
        pore=meta.get("pore"),
        detector=detector,
        concentration_nM=meta.get("concentration_nM"),
        denaturant=meta.get("denaturant"),
        voltage_mV=voltage_mV,
        measurement_voltage_mV=measurement_voltage_mV,
        fs_hz=meta.get("fs_hz"),
        cutoff_hz=meta.get("cutoff_hz"),
        baseline_pa=meta.get("baseline_pa"),
        notes=meta.get("notes"),
        trace_path=trace_path,
        events_path=events_path,
        stats_json=json.dumps(stats or {}, default=repr),
    )
    return row


def insert_experiment(row: Dict[str, Any], db_path: str = DEFAULT_DB) -> None:
    """
    Upsert by run_id.
    """
    init_catalog(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()

    cols = [
        "run_id", "title", "created_at", "operator", "analyte", "pore", "detector",
        "concentration_nM", "denaturant", "voltage_mV", "measurement_voltage_mV",
        "fs_hz", "cutoff_hz", "baseline_pa", "notes", "trace_path", "events_path", "stats_json"
    ]

    values = [row.get(c) for c in cols]

    # SQLite UPSERT (requires SQLite 3.24+)
    placeholders = ",".join(["?"] * len(cols))
    update_clause = ",".join([f"{c}=excluded.{c}" for c in cols if c != "run_id"])
    cur.execute(
        f"""
        INSERT INTO experiments ({",".join(cols)})
        VALUES ({placeholders})
        ON CONFLICT(run_id) DO UPDATE SET
        {update_clause}
        """,
        values
    )
    conn.commit()
    conn.close()


def list_experiments(db_path: str = DEFAULT_DB, with_counts: bool = True) -> pd.DataFrame:
    """
    Return a DataFrame of experiments. Optionally add an 'events' column by
    counting the rows in the events parquet for each row.
    """
    init_catalog(db_path)

    conn = get_conn(db_path)
    df = pd.read_sql_query("SELECT * FROM experiments ORDER BY created_at DESC, id DESC", conn)
    conn.close()

    # Backfill voltage if missing
    if "voltage_mV" in df.columns:
        # Use measurement if present and voltage is null
        if "measurement_voltage_mV" in df.columns:
            mask = df["voltage_mV"].isna()
            df.loc[mask, "voltage_mV"] = df.loc[mask, "measurement_voltage_mV"]

        # As a last resort, parse from run_id/title/events_path
    def _fix_row(row):
        v = row.get("voltage_mV")
        if pd.isna(v):
            gv = _guess_voltage_prioritized(
                run_id=row.get("run_id"),
                title=row.get("title"),
                events_path=row.get("events_path"),
                trace_path=row.get("trace_path"),
            )
            return gv
        return v
    

        df["voltage_mV"] = df.apply(_fix_row, axis=1)

    # Optional event counts
    if with_counts:
        ev_counts = []
        for p in df.get("events_path", pd.Series([], dtype=object)).fillna(""):
            try:
                ev_counts.append(_n_events_from_parquet(p))
            except Exception:
                ev_counts.append(0)
        df["events"] = pd.Series(ev_counts, index=df.index)

    return df


# ---------------------------------------------------------------------
# Maintenance / migrations
# ---------------------------------------------------------------------

def repair_voltages(db_path: str = DEFAULT_DB) -> int:
    """
    Backfill or correct voltage columns using run_id/title/paths.
    Returns number of rows updated.
    """
    init_catalog(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()

    cur.execute("SELECT id, run_id, title, voltage_mV, measurement_voltage_mV, trace_path, events_path FROM experiments")
    rows = cur.fetchall()

    updated = 0
    for (rid, run_id, title, v_db, v_meas, trace_p, events_p) in rows:
        guess = _guess_voltage_prioritized(run_id=run_id, title=title, events_path=events_p, trace_path=trace_p)
        new_v = v_db
        new_meas = v_meas

        if guess is not None:
            if v_db is None or int(v_db) != int(guess):
                new_v = int(guess)
            if v_meas is None:
                new_meas = int(guess)

        if new_v != v_db or new_meas != v_meas:
            cur.execute(
                "UPDATE experiments SET voltage_mV=?, measurement_voltage_mV=? WHERE id=?",
                (new_v, new_meas, rid),
            )
            updated += 1

    conn.commit()
    conn.close()
    return updated


# Backwards-compatible alias used in older pages
def migrate_voltage_backfill(db_path: str = DEFAULT_DB) -> int:
    return repair_voltages(db_path)


# ---------------------------------------------------------------------
# Convenience helpers (optional)
# ---------------------------------------------------------------------

def get_experiment(run_id: str, db_path: str = DEFAULT_DB) -> Optional[Dict[str, Any]]:
    init_catalog(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("SELECT * FROM experiments WHERE run_id=?", (run_id,))
    r = cur.fetchone()
    if not r:
        conn.close()
        return None
    # map result to dict
    cur.execute("PRAGMA table_info(experiments)")
    cols = [c[1] for c in cur.fetchall()]
    conn.close()
    return dict(zip(cols, r))


def delete_experiment(run_id: str, db_path: str = DEFAULT_DB) -> None:
    init_catalog(db_path)
    conn = get_conn(db_path)
    cur = conn.cursor()
    cur.execute("DELETE FROM experiments WHERE run_id=?", (run_id,))
    conn.commit()
    conn.close()
