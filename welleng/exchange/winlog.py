"""Reader for WinLog mudlog datapacks (Datalog Technology wellsite-geology exports).

A WinLog project is a directory holding:
  * a ``.wwf`` "Winlog Well File" -- a LAS-like sectioned ASCII header
    (well name, field, block, well type, ...),
  * a set of dBASE ``.dbf`` tables (one per data type: Survey, Fm_tops, Gas,
    Drill_Para, Casing, Lith Desc, ...), some with ``.dbt`` memo side-files,
  * ``.wlg`` / ``.wwf`` layout files (plot presentation) and ``.backup`` twins.

This reader returns the RAW table data (readers-vs-logic: core parses, the
consumer interprets). It is stdlib-only -- a compact dBASE III/IV reader is
inlined rather than adding a dependency, matching the exchange package's
stdlib-only convention (cf. :mod:`welleng.exchange.nlog`).

Presentation files (``.wlg``) and ``.backup`` twins are ignored. Empty tables
(schema, no records) are reported by :meth:`WinlogProject.empty_tables` and
excluded from :meth:`WinlogProject.table_names`.

Example
-------
>>> from welleng.exchange.winlog import open_winlog
>>> proj = open_winlog("/path/to/datapack")
>>> proj.well["WELLNAME"]                       # doctest: +SKIP
>>> proj.table_names()                          # doctest: +SKIP
>>> rows = proj.table("Survey (Composite)")     # doctest: +SKIP
>>> survey = proj.survey()                       # -> welleng Survey  # doctest: +SKIP
"""
from __future__ import annotations

import datetime
import glob
import os
import struct
from dataclasses import dataclass, field
from typing import Any


class WinlogError(Exception):
    """A WinLog datapack could not be read."""


# --------------------------------------------------------------------------- #
# dBASE III / IV (.dbf) reader -- stdlib only.
# --------------------------------------------------------------------------- #
@dataclass
class _Field:
    name: str
    type: str
    length: int
    decimal: int


def _read_memo(dbt_path: str, block: int) -> str:
    """Best-effort dBASE memo (.dbt) read. Returns '' if unreadable/absent."""
    if block <= 0 or not os.path.exists(dbt_path):
        return ""
    try:
        with open(dbt_path, "rb") as fh:
            # dBASE III memos are 512-byte blocks, text terminated by 0x1A 0x1A
            # (or a single 0x1A). dBASE IV prefixes an 8-byte block header.
            fh.seek(block * 512)
            raw = fh.read()
        if raw[:4] == b"\xff\xff\x08\x00":          # dBASE IV block header
            length = struct.unpack("<I", raw[4:8])[0]
            text = raw[8:8 + max(length - 8, 0)]
        else:                                        # dBASE III: read to 0x1A
            end = raw.find(b"\x1a")
            text = raw[:end] if end != -1 else raw
        return text.decode("latin-1", "ignore").rstrip("\x00 \r\n")
    except Exception:
        return ""


def _coerce(value: str, ftype: str) -> Any:
    v = value.strip()
    if ftype in "NF":                       # numeric / float
        if v in ("", "-", ".", "-."):
            return None
        try:
            return int(v) if ftype == "N" and "." not in v else float(v)
        except ValueError:
            return None
    if ftype == "L":                        # logical
        return {"Y": True, "T": True, "N": False, "F": False}.get(v.upper())
    if ftype == "D":                        # date YYYYMMDD
        if len(v) == 8 and v.isdigit():
            try:
                return datetime.date(int(v[:4]), int(v[4:6]), int(v[6:8]))
            except ValueError:
                return v
        return None
    return v                                # C (char) and anything else


def _read_dbf(path: str) -> tuple[list[_Field], list[dict[str, Any]]]:
    """Read a .dbf (+ sibling .dbt memo) -> (fields, list-of-row-dicts)."""
    with open(path, "rb") as fh:
        header = fh.read(32)
        n_records, header_len, record_len = struct.unpack("<xxxxIHH", header[:12])
        fields: list[_Field] = []
        while True:
            fd = fh.read(32)
            if not fd or fd[0] == 0x0D:      # 0x0D terminates the field array
                break
            name = fd[:11].split(b"\x00")[0].decode("latin-1", "ignore")
            ftype = chr(fd[11])
            flen, fdec = fd[16], fd[17]
            fields.append(_Field(name, ftype, flen, fdec))

    dbt = None
    for ext in (".dbt", ".DBT"):
        cand = os.path.splitext(path)[0] + ext
        if os.path.exists(cand):
            dbt = cand
            break

    rows: list[dict[str, Any]] = []
    with open(path, "rb") as fh:
        fh.seek(header_len)
        for _ in range(n_records):
            rec = fh.read(record_len)
            if len(rec) < record_len or rec[:1] == b"\x1a":
                break
            if rec[:1] == b"*":              # deleted record marker
                continue
            off, row = 1, {}                 # first byte is the deletion flag
            for f in fields:
                raw = rec[off:off + f.length].decode("latin-1", "ignore")
                off += f.length
                if f.type == "M":
                    blk = raw.strip()
                    row[f.name] = _read_memo(dbt, int(blk)) if blk.isdigit() else ""
                else:
                    row[f.name] = _coerce(raw, f.type)
            rows.append(row)
    return fields, rows


# --------------------------------------------------------------------------- #
# .wwf header reader (LAS-like sectioned "MNEM.  VALUE" lines).
# --------------------------------------------------------------------------- #
def _read_wwf(path: str) -> dict[str, str]:
    well: dict[str, str] = {}
    with open(path, "r", encoding="latin-1", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\r\n")
            if not line or line.startswith("~") or line.startswith("#"):
                continue
            if "." in line:
                mnem, rest = line.split(".", 1)
                key = mnem.strip().rstrip(".").upper()
                val = rest.strip()
                if key and val:
                    well.setdefault(key, val)
    return well


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def _norm(basename: str) -> str:
    return os.path.splitext(basename)[0]


@dataclass
class WinlogProject:
    """A parsed WinLog datapack. ``well`` is the header; tables are read lazily."""

    path: str
    well: dict[str, str] = field(default_factory=dict)
    _dbf_paths: dict[str, str] = field(default_factory=dict)   # norm-name -> path
    _cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)      # norm-name -> n_records

    def table_names(self) -> list[str]:
        """Names of tables that carry at least one record."""
        return sorted(n for n, c in self._counts.items() if c > 0)

    def empty_tables(self) -> list[str]:
        """Names of tables present but holding no records (schema only)."""
        return sorted(n for n, c in self._counts.items() if c == 0)

    def table(self, name: str) -> list[dict[str, Any]]:
        """Rows of one table (list of dicts). Case-insensitive on ``name``."""
        key = self._resolve(name)
        if key not in self._cache:
            _, rows = _read_dbf(self._dbf_paths[key])
            self._cache[key] = rows
        return self._cache[key]

    def _resolve(self, name: str) -> str:
        if name in self._dbf_paths:
            return name
        low = name.lower()
        for k in self._dbf_paths:
            if k.lower() == low:
                return k
        raise WinlogError(
            f"no table {name!r}; available: {', '.join(sorted(self._dbf_paths))}"
        )

    def survey(self, table: str | None = None, deg: bool = True):
        """Build a :class:`welleng.survey.Survey` from a survey table.

        Uses columns MD, INC, AZI (and TVD/EASTING/NORTHING when present). The
        table defaults to the first name containing 'survey'. Depth/coordinate
        UNITS follow the source pack (see ``well`` / the .wlg header) -- this
        does not convert them.
        """
        if table is None:
            cands = [n for n in self.table_names() if "survey" in n.lower()]
            if not cands:
                raise WinlogError("no survey table found")
            table = cands[0]
        rows = self.table(table)
        try:
            md = [r["MD"] for r in rows]
            inc = [r["INC"] for r in rows]
            azi = [r["AZI"] for r in rows]
        except KeyError as exc:
            raise WinlogError(
                f"survey table {table!r} lacks MD/INC/AZI column {exc}"
            ) from exc
        from welleng.survey import Survey, SurveyHeader   # lazy import
        # Position (N/E/TVD) is computed by welleng minimum curvature from
        # MD/INC/AZI; the pack's own TVD/EASTING/NORTHING columns stay available
        # via table() for a caller who wants to cross-check them.
        return Survey(
            md=md, inc=inc, azi=azi, deg=deg,
            header=SurveyHeader(name=self.well.get("WELLNAME")),
        )


def open_winlog(path: str) -> WinlogProject:
    """Open a WinLog datapack directory (or a .wwf file within one).

    Ignores ``.backup`` twins and ``.wlg`` layout files. Table record counts are
    read from the .dbf headers up front (cheap); row data is read on demand.
    """
    if os.path.isfile(path):
        path = os.path.dirname(path) or "."
    if not os.path.isdir(path):
        raise WinlogError(f"not a directory: {path}")

    proj = WinlogProject(path=path)

    wwfs = [p for p in glob.glob(os.path.join(path, "*"))
            if p.lower().endswith(".wwf") and not p.lower().endswith(".backup")]
    if wwfs:
        proj.well = _read_wwf(wwfs[0])

    for p in glob.glob(os.path.join(path, "*")):
        low = p.lower()
        if not low.endswith(".dbf") or low.endswith(".backup"):
            continue
        name = _norm(os.path.basename(p))
        proj._dbf_paths[name] = p
        try:                                 # count = header record count, no row read
            with open(p, "rb") as fh:
                proj._counts[name] = struct.unpack("<I", fh.read(8)[4:8])[0]
        except Exception:
            proj._counts[name] = -1
    if not proj._dbf_paths:
        raise WinlogError(f"no .dbf tables found in {path}")
    return proj
