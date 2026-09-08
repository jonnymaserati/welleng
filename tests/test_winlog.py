"""Hermetic tests for the WinLog datapack reader (welleng.exchange.winlog).

Builds a tiny synthetic WinLog pack (a .wwf header + minimal dBASE III tables)
in a tmp dir -- no external fixture, no proprietary data. The reader was also
validated against a real Datalog/WinLog pack out of band (welleng min-curve TVD
reproduced the pack's own TVD column to < 1 mm).
"""
import struct

import numpy as np
import pytest

from welleng.exchange.winlog import WinlogError, open_winlog


def _write_dbf(path, fields, rows):
    """Minimal dBASE III writer. fields = [(name, type, length, decimal)]."""
    rec_len = 1 + sum(f[2] for f in fields)          # +1 deletion flag
    header_len = 32 + 32 * len(fields) + 1
    hdr = struct.pack("<BBBBIHH20x", 0x03, 125, 1, 1, len(rows), header_len, rec_len)
    fds = b""
    for name, ftype, flen, fdec in fields:
        fds += name.encode("latin-1")[:11].ljust(11, b"\x00")
        fds += ftype.encode("latin-1")
        fds += b"\x00\x00\x00\x00"
        fds += struct.pack("<BB", flen, fdec)
        fds += b"\x00" * 14
    with open(path, "wb") as fh:
        fh.write(hdr + fds + b"\x0d")
        for row in rows:
            fh.write(b"\x20")                        # not-deleted
            for name, ftype, flen, fdec in fields:
                v = row.get(name, "")
                s = ("" if v is None else str(v))
                cell = s.rjust(flen) if ftype in "NF" else s.ljust(flen)
                fh.write(cell.encode("latin-1")[:flen].ljust(flen, b" "))
        fh.write(b"\x1a")


@pytest.fixture
def pack(tmp_path):
    (tmp_path / "WELL.wwf").write_text(
        "~Version\nID.           Winlog Well File\nVERSION.      4.0\n"
        "~Well Data\nWELLNAME.     TEST-1\nFIELD.        Synthetic\n"
        "BLOCK.        X00\nWELL TYPE.    Exploration\n",
        encoding="latin-1",
    )
    # a build well: vertical, then build to ~30 deg
    surv = [
        {"MD": 0.0, "INC": 0.0, "AZI": 45.0, "TVD": 0.0},
        {"MD": 500.0, "INC": 0.0, "AZI": 45.0, "TVD": 500.0},
        {"MD": 800.0, "INC": 30.0, "AZI": 45.0, "TVD": 786.48},
        {"MD": 1100.0, "INC": 30.0, "AZI": 45.0, "TVD": 1046.29},
    ]
    _write_dbf(tmp_path / "Survey (Composite).DBF",
               [("MD", "N", 10, 2), ("INC", "N", 8, 2),
                ("AZI", "N", 8, 2), ("TVD", "N", 10, 2)], surv)
    _write_dbf(tmp_path / "Fm_tops.DBF",
               [("FM_NAME", "C", 20, 0), ("TOP_MD", "N", 10, 2)],
               [{"FM_NAME": "North Sea Grp", "TOP_MD": 120.0},
                {"FM_NAME": "Chalk Grp", "TOP_MD": 950.0}])
    _write_dbf(tmp_path / "RFT.dbf",                 # empty -> schema only
               [("TEST", "C", 8, 0), ("PRES", "N", 10, 2)], [])
    # a .backup twin that must be ignored
    _write_dbf(tmp_path / "Survey (Composite).DBF.backup",
               [("MD", "N", 10, 2)], [{"MD": 999.0}])
    return tmp_path


def test_header(pack):
    p = open_winlog(str(pack))
    assert p.well["WELLNAME"] == "TEST-1"
    assert p.well["FIELD"] == "Synthetic"
    assert p.well["WELL TYPE"] == "Exploration"


def test_table_discovery_and_empty(pack):
    p = open_winlog(str(pack))
    names = p.table_names()
    assert "Survey (Composite)" in names and "Fm_tops" in names
    assert "RFT" in p.empty_tables()          # present but no records
    assert "RFT" not in names
    # the .backup twin is not surfaced as its own table
    assert not any(n.endswith(".backup") for n in names)
    assert "Survey (Composite).DBF" not in names


def test_table_rows_and_types(pack):
    p = open_winlog(str(pack))
    fm = p.table("Fm_tops")
    assert len(fm) == 2
    assert fm[0]["FM_NAME"] == "North Sea Grp"
    assert fm[1]["TOP_MD"] == 950.0            # numeric coerced
    assert isinstance(fm[1]["TOP_MD"], float)


def test_table_case_insensitive(pack):
    p = open_winlog(str(pack))
    assert p.table("fm_tops") == p.table("Fm_tops")


def test_survey_builds_and_matches_pack_tvd(pack):
    p = open_winlog(str(pack))
    s = p.survey()
    assert len(s.md) == 4
    # welleng min-curve TVD reproduces the pack's own TVD column (rounded input)
    src_tvd = np.array([r["TVD"] for r in p.table("Survey (Composite)")], float)
    assert np.max(np.abs(np.asarray(s.tvd) - src_tvd)) < 0.02


def test_missing_table_raises(pack):
    p = open_winlog(str(pack))
    with pytest.raises(WinlogError):
        p.table("does_not_exist")


def test_not_a_pack(tmp_path):
    with pytest.raises(WinlogError):
        open_winlog(str(tmp_path))             # no .dbf tables
