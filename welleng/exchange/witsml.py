"""Streaming WITSML 1.4.1.1 (``1series``) reader.

WITSML (Wellsite Information Transfer Standard Markup Language) is the industry
XML format for real-time and historical drilling data. This module reads the
**1.4.1.1** flavour (the ``1series`` schema,
``xmlns="http://www.witsml.org/schemas/1series"``) — the flavour used by the
public Equinor **Volve** realtime export. The 1.3.1.1 and 2.0 ("Energistics")
schemas differ structurally and are **not** handled; the reader guards on the
declared version and raises a clear error on any other flavour.

Two object types are exposed, both streamed from a directory or a ``.zip`` of
per-object XML files:

* **logs** (:class:`LogInfo` / :meth:`WITSMLReader.logs`) — time- or
  depth-indexed channel data (hookload, block position, torque, temperatures,
  ECD/SPP/PWD, …). A ``<log>`` carries a ``<mnemonicList>`` (the column order)
  and comma-joined ``<data>`` rows.
* **tubulars** (:class:`Tubular` / :meth:`WITSMLReader.tubulars`) — the as-run
  string / BHA component tally (OD/ID/length/weight per component), which is
  frequently absent from the EDM and only authoritative in WITSML.

Design notes
------------
* **Streaming index.** A zip of Volve realtime data is ~2.8 GB across ~18k
  member files; parsing every member's data up front is impractical. The reader
  builds a light index by decompressing only a bounded **header prefix**
  (``zipfile.ZipFile.open(...).read(n)``) of each log member — enough to reach
  the ``<mnemonicList>`` — and defers the (large) ``<data>`` parse until
  :meth:`LogInfo.curves` is actually called for that log.
* **Column order.** The authoritative column order for ``<data>`` rows is the
  ``<mnemonicList>`` inside ``<logData>``; the reader falls back to the
  ``<logCurveInfo><mnemonic>`` order only if the list is absent.
* **Null values.** Each log declares ``<nullValue>`` (Volve: ``-999.25``);
  matching values are returned as ``NaN`` in numeric channels.
* **Regex, not DOM.** Like the seed recon scripts, header extraction is
  regex-based: it is robust to the member files being fragments and avoids
  building a DOM for a multi-megabyte member just to read its channel list.
"""

from __future__ import annotations

import os
import re
import zipfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# ---- schema -----------------------------------------------------------------
WITSML_1SERIES_NS = "http://www.witsml.org/schemas/1series"
SUPPORTED_VERSION_PREFIX = "1.4"

# Bytes decompressed per log member to reach the <mnemonicList> for the index.
# Volve's richest headers place the list within ~13 KB; 64 KB is a safe margin.
_HEADER_BYTES = 65536

# ---- header regexes (operate on bytes) --------------------------------------
_RX_VERSION = re.compile(rb'<logs\b[^>]*\bversion="([^"]*)"')
_RX_VERSION_T = re.compile(rb'<tubulars\b[^>]*\bversion="([^"]*)"')
_RX_NAME = re.compile(rb"<name>([^<]*)</name>")
_RX_NAME_WELL = re.compile(rb"<nameWell>([^<]*)</nameWell>")
_RX_NAME_WB = re.compile(rb"<nameWellbore>([^<]*)</nameWellbore>")
_RX_ITYPE = re.compile(rb"<indexType>([^<]*)</indexType>")
_RX_START_T = re.compile(rb"<startDateTimeIndex>([^<]*)</startDateTimeIndex>")
_RX_END_T = re.compile(rb"<endDateTimeIndex>([^<]*)</endDateTimeIndex>")
_RX_START_D = re.compile(rb"<startIndex[^>]*>([^<]*)</startIndex>")
_RX_END_D = re.compile(rb"<endIndex[^>]*>([^<]*)</endIndex>")
_RX_NULL = re.compile(rb"<nullValue>([^<]*)</nullValue>")
_RX_MNEMLIST = re.compile(rb"<mnemonicList>([^<]*)</mnemonicList>")
_RX_MNEM = re.compile(rb"<mnemonic\b[^>]*>([^<]*)</mnemonic>")
_RX_DATA = re.compile(rb"<data>([^<]*)</data>")

# ---- tubular regexes --------------------------------------------------------
_RX_TCOMP = re.compile(rb"<tubularComponent\b.*?</tubularComponent>", re.S)
_RX_SEQ = re.compile(rb"<sequence>([^<]*)</sequence>")
_RX_TYPE = re.compile(rb"<typeTubularComp>([^<]*)</typeTubularComp>")
_RX_OD = re.compile(rb'<od\b[^>]*>([^<]*)</od>')
_RX_ID = re.compile(rb'<id\b[^>]*>([^<]*)</id>')
_RX_LEN = re.compile(rb'<len\b[^>]*>([^<]*)</len>')
_RX_WTPL = re.compile(rb'<wtPerLen\b[^>]*>([^<]*)</wtPerLen>')


def _first(rx: re.Pattern, b: bytes) -> str:
    m = rx.search(b)
    return m.group(1).decode("utf-8", "replace").strip() if m else ""


def _fnum(rx: re.Pattern, b: bytes) -> float:
    s = _first(rx, b)
    try:
        return float(s)
    except ValueError:
        return float("nan")


class WITSMLVersionError(ValueError):
    """Raised when a member declares a WITSML version this reader cannot parse."""


# ---------------------------------------------------------------------------
@dataclass
class LogInfo:
    """Lightweight index entry for one WITSML ``<log>`` (data parsed on demand)."""

    name: str
    well: str
    wellbore: str
    index_type: str          # "date time" | "measured depth" | "length" | ...
    start: str
    end: str
    mnemonics: Tuple[str, ...]
    null_value: float
    member: str = field(repr=False)
    _reader: "WITSMLReader" = field(repr=False, default=None)

    @property
    def is_time_indexed(self) -> bool:
        return self.index_type.lower().startswith("date")

    @property
    def index_mnemonic(self) -> str:
        """The index (column 0) channel, e.g. ``Time`` or ``DEPTH``."""
        return self.mnemonics[0] if self.mnemonics else ""

    def curves(
        self, mnemonics: Optional[Sequence[str]] = None
    ) -> Dict[str, np.ndarray]:
        """Parse this log's ``<data>`` and return ``{mnemonic: array}``.

        The index channel is always included. Numeric channels are ``float``
        arrays with ``null_value`` mapped to ``NaN``; the index of a time log is
        returned as a ``datetime64[ns]`` array, of a depth log as ``float``.
        ``mnemonics=None`` returns every channel.
        """
        return self._reader._read_curves(self, mnemonics)


@dataclass
class TubularComponent:
    sequence: int
    type: str
    od_in: float
    id_in: float
    len_m: float
    wt_kgm: float


@dataclass
class Tubular:
    """A WITSML ``<tubular>`` — an ordered string / BHA component tally."""

    name: str
    well: str
    wellbore: str
    components: List[TubularComponent]
    member: str = field(repr=False)


# ---------------------------------------------------------------------------
class WITSMLReader:
    """Streaming reader for a directory or zip of WITSML 1.4.1.1 files.

    Parameters
    ----------
    path : str
        A ``.zip`` (e.g. the Volve realtime export) or a directory tree of
        WITSML ``.xml`` member files.
    version_check : bool, default True
        Guard on the declared ``version`` of each object; raise
        :class:`WITSMLVersionError` on a non-1.4 flavour. Set ``False`` to read
        version-less fragments best-effort.

    Notes
    -----
    In a zip, log members are identified by ``/log/`` in the path and tubular
    members by ``/tubular/`` (the Volve/SiteCom layout). In a directory tree the
    object type is detected from the member's root element instead.
    """

    def __init__(self, path: str, version_check: bool = True):
        self.path = path
        self.version_check = version_check
        self._zip = zipfile.ZipFile(path) if zipfile.is_zipfile(path) else None
        if self._zip is None and not os.path.isdir(path):
            raise FileNotFoundError(f"not a zip or directory: {path!r}")
        self._members = self._list_members()
        self._log_index: Optional[List[LogInfo]] = None
        self.version: Optional[str] = None

    @classmethod
    def open(cls, path: str, version_check: bool = True) -> "WITSMLReader":
        return cls(path, version_check=version_check)

    # -- member enumeration ---------------------------------------------------
    def _list_members(self) -> List[str]:
        if self._zip is not None:
            return [
                n for n in self._zip.namelist()
                if n.lower().endswith(".xml") and not n.endswith("/")
            ]
        out: List[str] = []
        for root, _, files in os.walk(self.path):
            for f in files:
                if f.lower().endswith(".xml"):
                    out.append(os.path.join(root, f))
        return out

    def _is_log_member(self, member: str, prefix: bytes) -> bool:
        if self._zip is not None:
            return "/log/" in member
        return b"<logs" in prefix or b"<log " in prefix or b"<log>" in prefix

    def _is_tubular_member(self, member: str, prefix: bytes) -> bool:
        if self._zip is not None:
            return "/tubular/" in member
        return b"<tubular" in prefix

    def _read_prefix(self, member: str, nbytes: int = _HEADER_BYTES) -> bytes:
        if self._zip is not None:
            with self._zip.open(member) as fh:
                return fh.read(nbytes)
        with open(member, "rb") as fh:
            return fh.read(nbytes)

    def _read_full(self, member: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(member)
        with open(member, "rb") as fh:
            return fh.read()

    # -- log index ------------------------------------------------------------
    def _well_wellbore(self, member: str, prefix: bytes) -> Tuple[str, str]:
        well = _first(_RX_NAME_WELL, prefix)
        wb = _first(_RX_NAME_WB, prefix)
        if well and wb:
            return well, wb
        # Fall back to the SiteCom path layout .../<well>/<wellbore>/log/...
        if self._zip is not None:
            for tag in ("/log/", "/tubular/"):
                if tag in member:
                    head = member.split(tag)[0].split("/")
                    if len(head) >= 2:
                        return well or head[-2], wb or head[-1]
        return well, wb

    def _index_one_log(self, member: str) -> Optional[LogInfo]:
        prefix = self._read_prefix(member)
        if not self._is_log_member(member, prefix):
            return None
        m = _RX_VERSION.search(prefix)
        if m is not None:
            ver = m.group(1).decode("ascii", "replace").strip()
            if self.version is None:
                self.version = ver
            if self.version_check and not ver.startswith(SUPPORTED_VERSION_PREFIX):
                raise WITSMLVersionError(
                    f"WITSML version {ver!r} in {member!r} is not supported"
                    f" (this reader handles {SUPPORTED_VERSION_PREFIX}x /"
                    f" 1series logs only)"
                )
        itype = _first(_RX_ITYPE, prefix)
        if itype.lower().startswith("date"):
            start, end = _first(_RX_START_T, prefix), _first(_RX_END_T, prefix)
        else:
            start, end = _first(_RX_START_D, prefix), _first(_RX_END_D, prefix)
        mnem_list = _first(_RX_MNEMLIST, prefix)
        if mnem_list:
            mnems = tuple(s.strip() for s in mnem_list.split(",") if s.strip())
        else:
            # Fall back to logCurveInfo order (may require the full member).
            src = prefix if _RX_MNEM.search(prefix) else self._read_full(member)
            mnems = tuple(
                m.decode("utf-8", "replace").strip()
                for m in _RX_MNEM.findall(src)
            )
        null = _first(_RX_NULL, prefix)
        try:
            null_value = float(null) if null else float("nan")
        except ValueError:
            null_value = float("nan")
        well, wb = self._well_wellbore(member, prefix)
        return LogInfo(
            name=_first(_RX_NAME, prefix), well=well, wellbore=wb,
            index_type=itype, start=start, end=end, mnemonics=mnems,
            null_value=null_value, member=member, _reader=self,
        )

    @property
    def logs(self) -> List[LogInfo]:
        """All logs, indexed lazily on first access (header prefixes only)."""
        if self._log_index is None:
            idx: List[LogInfo] = []
            for member in self._members:
                if self._zip is None:
                    # directory: cheap type sniff before indexing
                    pass
                try:
                    li = self._index_one_log(member)
                except WITSMLVersionError:
                    raise
                except Exception:
                    li = None
                if li is not None:
                    idx.append(li)
            self._log_index = idx
        return self._log_index

    @property
    def wells(self) -> List[str]:
        return sorted({li.well for li in self.logs if li.well})

    def find(
        self, mnemonic: str, well: Optional[str] = None
    ) -> List[LogInfo]:
        """Logs carrying ``mnemonic`` (case-sensitive), optionally in one well."""
        out = []
        for li in self.logs:
            if mnemonic in li.mnemonics and (well is None or li.well == well):
                out.append(li)
        return out

    # -- data parse (on demand) ----------------------------------------------
    def _read_curves(
        self, log: LogInfo, mnemonics: Optional[Sequence[str]]
    ) -> Dict[str, np.ndarray]:
        raw = self._read_full(log.member)
        m = _RX_MNEMLIST.search(raw)
        if m:
            cols = [s.strip() for s in
                    m.group(1).decode("utf-8", "replace").split(",") if s.strip()]
        else:
            cols = list(log.mnemonics)
        rows = _RX_DATA.findall(raw)
        ncol = len(cols)
        # Split rows into a (nrow, ncol) string table (ragged rows padded).
        table = [[""] * ncol for _ in range(len(rows))]
        for i, row in enumerate(rows):
            vals = row.decode("utf-8", "replace").split(",")
            for j in range(min(ncol, len(vals))):
                table[i][j] = vals[j].strip()

        want = cols if mnemonics is None else [c for c in cols if c in mnemonics]
        # ensure the index channel is always present
        if cols and cols[0] not in want:
            want = [cols[0]] + want

        null = log.null_value
        out: Dict[str, np.ndarray] = {}
        for c in want:
            j = cols.index(c)
            col = [table[i][j] for i in range(len(rows))]
            if c == cols[0] and log.is_time_indexed:
                # WITSML times are ISO-8601, Volve in UTC ("...Z"). numpy's
                # datetime64 has no tz; strip the trailing "Z" and treat as
                # UTC-naive (avoids the per-parse tz UserWarning).
                clean = [v[:-1] if v.endswith("Z") else v for v in col]
                out[c] = np.array(clean, dtype="datetime64[ns]") if clean \
                    else np.array([], dtype="datetime64[ns]")
            else:
                arr = np.empty(len(col), dtype=float)
                for i, v in enumerate(col):
                    if v == "" :
                        arr[i] = np.nan
                        continue
                    try:
                        x = float(v)
                    except ValueError:
                        arr[i] = np.nan
                        continue
                    arr[i] = np.nan if (not np.isnan(null) and x == null) else x
                out[c] = arr
        return out

    # -- tubulars -------------------------------------------------------------
    def tubulars(self) -> List[Tubular]:
        """All tubular strings (BHA/casing component tallies) in the source."""
        M2IN = 1.0 / 0.0254
        out: List[Tubular] = []
        for member in self._members:
            prefix = self._read_prefix(member)
            if not self._is_tubular_member(member, prefix):
                continue
            if self.version_check:
                mv = _RX_VERSION_T.search(prefix)
                if mv is not None:
                    ver = mv.group(1).decode("ascii", "replace").strip()
                    if not ver.startswith(SUPPORTED_VERSION_PREFIX):
                        raise WITSMLVersionError(
                            f"WITSML tubular version {ver!r} in {member!r}"
                            f" is not supported"
                        )
            raw = self._read_full(member)
            comps: List[TubularComponent] = []
            for c in _RX_TCOMP.findall(raw):
                seq = _first(_RX_SEQ, c)
                comps.append(TubularComponent(
                    sequence=int(seq) if seq.isdigit() else 0,
                    type=_first(_RX_TYPE, c),
                    od_in=_fnum(_RX_OD, c) * M2IN,
                    id_in=_fnum(_RX_ID, c) * M2IN,
                    len_m=_fnum(_RX_LEN, c),
                    wt_kgm=_fnum(_RX_WTPL, c),
                ))
            comps.sort(key=lambda r: r.sequence)
            well, wb = self._well_wellbore(member, prefix)
            out.append(Tubular(
                name=_first(_RX_NAME, prefix), well=well, wellbore=wb,
                components=comps, member=member,
            ))
        return out

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()

    def __enter__(self) -> "WITSMLReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def open_witsml(path: str, version_check: bool = True) -> WITSMLReader:
    """Open a WITSML 1.4.1.1 source (zip or directory). See :class:`WITSMLReader`."""
    return WITSMLReader.open(path, version_check=version_check)
