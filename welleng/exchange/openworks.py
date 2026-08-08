"""Read Landmark OpenWorks OW-IO horizon exports.

OpenWorks (Landmark's G&G platform) exports a horizon as a delimited ASCII
``.dat`` file (an ``ADTFormatDef`` / OW-IO export). A ``#``-comment header
records the export provenance and cartographic system; a short text header names
the survey, horizon and domain (``DEPTH`` / ``TWT``); then the data rows are
comma-delimited::

    inline, crossline, X (easting), Y (northing), Z (depth or time)

This reads that file into a :class:`~welleng.surface.Surface`. (The native
OpenWorks ``.bck`` / DecisionSpace ``.dszip`` backups are proprietary binary and
are not read here — export to OW-IO first.)
"""
from __future__ import annotations

import re
from typing import List, Optional

from ..surface import Surface

__all__ = ["read_ow_horizon"]

_CRS_RE = re.compile(r"cartographic system name[:\s]+(\S+)", re.IGNORECASE)
_DOMAINS = {"DEPTH", "TWT", "TIME"}


def _open(path_or_lines):
    if isinstance(path_or_lines, (list, tuple)):
        return list(path_or_lines)
    with open(path_or_lines, "r", errors="replace") as fh:
        return fh.read().splitlines()


def read_ow_horizon(path, *, name: Optional[str] = None) -> Surface:
    """Read an OpenWorks OW-IO horizon ``.dat`` export into a :class:`Surface`.

    Parameters
    ----------
    path : str or list of str
        Path to the ``.dat`` file (or its lines, for testing).
    name : str, optional
        Override the horizon name (else taken from the file header).

    Returns
    -------
    Surface
    """
    lines = _open(path)

    crs = None
    header_text: List[str] = []
    il, xl, x, y, z = [], [], [], [], []

    for ln in lines:
        s = ln.strip()
        if not s or s == "=":
            continue
        if s.startswith("#"):
            m = _CRS_RE.search(s)
            if m and crs is None:
                crs = m.group(1)
            continue
        # a data row: >= 5 comma-separated numeric fields
        parts = s.split(",")
        if len(parts) >= 5:
            try:
                vals = [float(p) for p in parts[:5]]
            except ValueError:
                header_text.append(s)
                continue
            il.append(vals[0])
            xl.append(vals[1])
            x.append(vals[2])
            y.append(vals[3])
            z.append(vals[4])
        else:
            header_text.append(s)

    if not z:
        raise ValueError(f"no horizon data rows found in {path!r}")

    # domain + name from the short text header
    domain = "DEPTH"
    for h in header_text:
        if h.upper() in _DOMAINS:
            domain = "TWT" if h.upper() in ("TWT", "TIME") else "DEPTH"
            break
    if name is None:
        # the horizon name is the header line that is not the survey/STAT/domain
        # tokens — the one carrying letters and typically the longest.
        candidates = [
            h for h in header_text
            if h.upper() not in _DOMAINS and h.upper() != "STAT"
            and re.search(r"[A-Za-z]", h)
        ]
        name = max(candidates, key=len) if candidates else ""

    return Surface.from_nodes(il, xl, x, y, z, name=name, domain=domain, crs=crs)
