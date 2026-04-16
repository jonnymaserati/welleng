"""Parser for Copsegrove ``ErrorModelDiagnostics`` .txt files.

These are the canonical reference outputs published by ISCWSA / OWSG
for each standard tool model on each ISCWSA standard test well. Each
file is ~16 k lines containing a per-station diagnostic block with:

- Per-error-source HLA sigmas (HighSide / Lateral / Along-Hole) in
  the ``svy`` and ``dep`` rows.
- Per-error-source NEV covariance contributions (NN, EE, VV, NE, NV,
  EV) in the ``NEV Covariance Matrix Terms`` section.
- Partial sums by category and station totals.

For the conformance harness, the per-term per-station ``svy`` row is
the most direct comparison point: it equals
``abs(dpde[axis]) × magnitude`` for the error term at that station,
in the order [depth=A/H, inclination=H/S, azimuth=LAT]. No
propagation between stations is involved at the ``svy`` level, so
the validation is pure formula-evaluation.

Reference precision in the source files: 4 decimal places.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class StationDiagnostic:
    """One per-station block extracted from a Copsegrove .txt file.

    Attributes
    ----------
    md : float
        Measured depth (the value at the start of the H/S/LAT/A/H summary line).
    inc_deg : float
        Inclination in degrees (parsed from the 'Inc:' field on that
        same summary line).
    azi_deg : float
        Azimuth in degrees (parsed from 'Az (True):' on that line).
    sigmas_svy : dict[str, tuple[float, float, float]]
        For each error-term code present in this station's block, the
        (H/S, LAT, A/H) survey-only sigmas before propagation.
    """

    md: float
    inc_deg: float
    azi_deg: float
    sigmas_svy: dict[str, tuple[float, float, float]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Lines like:
#   "DRFR     svy      0.0000      0.0000      0.3500      0.0000  ..."
# Code in col 1, 'svy' or 'dep' in col 2, then 9 numbers (we only use first 3).
_SIG_LINE_RE = re.compile(
    r"^\s*([A-Z][A-Z0-9\-]*)\s+svy\s+"
    r"(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)

# Lines like:
#   "30.00        Inc: 0.00       Az (True): 0.00  TF: 0.00"
_STATION_HDR_RE = re.compile(
    r"^\s*(\d+\.\d+)\s+Inc:\s+(-?\d+\.\d+)\s+Az\s*\(True\):\s+(-?\d+\.\d+)"
)


def parse_copsegrove(path: str | Path) -> list[StationDiagnostic]:
    """Walk a Copsegrove .txt file, yield one StationDiagnostic per station.

    The parser is tolerant: it ignores everything outside the per-station
    diagnostic blocks. It collects every ``svy`` row before each station
    header into the corresponding station's ``sigmas_svy`` dict.
    """
    stations: list[StationDiagnostic] = []
    pending: dict[str, tuple[float, float, float]] = {}
    with open(path) as f:
        for line in f:
            m = _SIG_LINE_RE.match(line)
            if m:
                code = m.group(1)
                # Skip aggregate / category labels that look like codes.
                if code in {"Sensor", "Az", "Magnetic", "Align", "Reading",
                            "Depth", "Totals", "Total"}:
                    continue
                pending[code] = (
                    float(m.group(2)), float(m.group(3)), float(m.group(4)),
                )
                continue
            h = _STATION_HDR_RE.match(line)
            if h:
                stations.append(StationDiagnostic(
                    md=float(h.group(1)),
                    inc_deg=float(h.group(2)),
                    azi_deg=float(h.group(3)),
                    sigmas_svy=dict(pending),
                ))
                pending.clear()
                continue
    return stations


# ---------------------------------------------------------------------------
# Indexed access: per-term per-station sigmas as a (n_stations, 3) array
# ---------------------------------------------------------------------------


def per_term_sigmas(
    stations: list[StationDiagnostic], term: str,
) -> tuple[list[float], list[float], list[float], list[tuple[float, float, float]]]:
    """For one error-term code, gather the per-station (H/S, LAT, A/H)
    sigmas across all stations where the term appears.

    Returns ``(md, inc_deg, azi_deg, sigmas)`` aligned by station —
    sigmas is a list of (H/S, LAT, A/H) tuples per station.

    Stations that don't carry the term are skipped, which is the
    expected behaviour: Copsegrove only emits a row for terms that
    were active at that station.
    """
    md, inc, azi, sig = [], [], [], []
    for s in stations:
        if term in s.sigmas_svy:
            md.append(s.md)
            inc.append(s.inc_deg)
            azi.append(s.azi_deg)
            sig.append(s.sigmas_svy[term])
    return md, inc, azi, sig
