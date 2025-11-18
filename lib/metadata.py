# lib/metadata.py
from __future__ import annotations
import re
from typing import Dict, Optional, Any

_VOL_RE = re.compile(r'(\d{2,3})\s*m[v]?\b', re.IGNORECASE)   # 60mV, 120 mv, 150m
_PORE_RE = re.compile(r'\bch\s?(\d{2,4})\b', re.IGNORECASE)   # CH007, ch7, ch 07
_CONC_RE = re.compile(r'(\d+(?:\.\d+)?)\s*mg[_-]?(?:ml|/ml)\b', re.IGNORECASE)  # 0.5mgml, 0.5 mg/ml

def parse_from_name(name: str) -> Dict[str, Any]:
    """
    Best-effort metadata extraction from a filename / run title.
    Returns a dict with keys present only if they could be parsed.
    """
    out: Dict[str, Any] = {}
    s = str(name)
    m = _VOL_RE.search(s)
    if m:
        v = int(m.group(1))
        if 10 <= v <= 300:
            out["voltage_mV"] = v
    m = _PORE_RE.search(s)
    if m:
        out["pore"] = f"CH{m.group(1)}"
    m = _CONC_RE.search(s)
    if m:
        out["concentration_mg_ml"] = float(m.group(1))
    return out
