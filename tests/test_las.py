"""Hermetic tests for the LAS reader/renderer (welleng.exchange.las).

Builds a small LAS 2.0 file as text -- no external fixture. lasio and
matplotlib are optional (welleng[las]); the tests skip if absent.
"""
import numpy as np
import pytest

from welleng.exchange.las import LasError, open_las, plot_curves

lasio = pytest.importorskip("lasio")

LAS_TEXT = """~Version Information
VERS.   2.0 : CWLS LOG ASCII STANDARD - VERSION 2.0
WRAP.   NO  : ONE LINE PER DEPTH STEP
~Well Information
STRT.M   100.0000 : START DEPTH
STOP.M   104.0000 : STOP DEPTH
STEP.M     1.0000 : STEP
NULL.     -999.25 : NULL VALUE
WELL.    TEST-1   : WELL
FLD .    Synthetic : FIELD
~Curve Information
DEPT.M            : DEPTH
GR  .GAPI         : GAMMA RAY
RT  .OHMM         : TRUE RESISTIVITY
RHOB.G/C3         : BULK DENSITY
~ASCII
 100.0000   45.2000    2.5000    2.3100
 101.0000   52.7000    4.1000    2.3500
 102.0000   88.1000   20.4000    2.4800
 103.0000   30.5000    1.9000    2.2200
 104.0000  -999.2500   3.3000    2.4000
"""


@pytest.fixture
def las(tmp_path):
    p = tmp_path / "test.las"
    p.write_text(LAS_TEXT)
    return open_las(str(p))


def test_reads_header(las):
    assert las.well["WELL"] == "TEST-1"
    assert las.index_mnemonic == "DEPT"


def test_reads_curves_and_units(las):
    assert las.mnemonics() == ["DEPT", "GR", "RT", "RHOB"]
    assert las.units["GR"] == "GAPI"
    np.testing.assert_allclose(las.depth, [100, 101, 102, 103, 104])
    np.testing.assert_allclose(las.curve("RT")[:2], [2.5, 4.1])


def test_null_becomes_nan(las):
    # the -999.25 NULL in the last GR sample must not survive as a number
    assert np.isnan(las.curve("GR")[-1])


def test_curve_lookup_is_case_insensitive(las):
    np.testing.assert_allclose(las.curve("gr"), las.curve("GR"))


def test_missing_curve_raises(las):
    with pytest.raises(LasError):
        las.curve("NOPE")


def test_open_from_bytes_and_text(tmp_path):
    from_bytes = open_las(LAS_TEXT.encode("latin-1"))
    from_text = open_las(LAS_TEXT)
    assert from_bytes.well["WELL"] == "TEST-1"
    np.testing.assert_allclose(from_text.depth, from_bytes.depth)


def test_describe_reports_completeness(las):
    text = las.describe()
    assert "GR" in text and "GAPI" in text
    assert "80.0%" in text          # 4 of 5 GR samples finite


def test_plot_renders_a_file(las, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "log.png"
    fig = plot_curves(las, out=str(out))
    assert out.exists() and out.stat().st_size > 1000
    # default grouping: gamma track, resistivity track, porosity track
    assert len(fig.axes) == 3


def test_plot_depth_range_and_empty_range(las, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / "sub.png"
    plot_curves(las, depth_range=(101, 103), out=str(out))
    assert out.exists()
    with pytest.raises(LasError):
        plot_curves(las, depth_range=(9000, 9100), out=str(out))


# --- gaps reported by a consumer mining 24 wells ---------------------------- #
WRAPPED = """~Version
VERS. 2.0 : CWLS LOG ASCII STANDARD
WRAP. YES : MULTIPLE LINES PER DEPTH STEP
~Well
STRT.M 100.0 :
STOP.M 102.0 :
STEP.M  1.0 :
NULL. -999.25 :
WELL. WRAPTEST :
~Curve
DMEA.M   : DEPTH
GR  .GAPI : GAMMA
RHOB.G/C3 : DENSITY
~ASCII
   100.0000
     45.1000   2.3100
   101.0000
     46.2000   2.3400
   102.0000
     47.3000   2.3600
"""


def test_wrapped_file_reads(tmp_path):
    """WRAP: YES is common in pre-2000 logs and must not fail.

    Which lasio engine gets picked by default is version dependent -- one
    warns and falls back, another raises part way through the read -- so the
    engine is pinned for wrapped files.
    """
    from welleng.exchange.las import _is_wrapped
    p = tmp_path / "wrapped.las"
    p.write_text(WRAPPED)
    assert _is_wrapped(str(p))
    las = open_las(str(p))
    assert las.mnemonics() == ["DMEA", "GR", "RHOB"]
    np.testing.assert_allclose(las.depth, [100, 101, 102])
    np.testing.assert_allclose(las.curve("RHOB"), [2.31, 2.34, 2.36])


def test_unwrapped_is_not_flagged_as_wrapped(tmp_path):
    from welleng.exchange.las import _is_wrapped
    p = tmp_path / "plain.las"
    p.write_text(LAS_TEXT)
    assert _is_wrapped(str(p)) is False


def test_index_is_detected_not_assumed(tmp_path):
    """Mudlogs in some fields use DMEA; assuming DEPT mis-indexes them all."""
    p = tmp_path / "wrapped.las"
    p.write_text(WRAPPED)
    assert open_las(str(p)).index_mnemonic == "DMEA"


def test_curves_by_unit_finds_density_whatever_it_is_called(tmp_path):
    """Mnemonic matching is a trap: density is RHOB / RHOZ / BDCX by vintage.

    A consumer's name-list search reported "no density curve" on 12 of 14
    wells that had one; matching on the unit found them all.
    """
    p = tmp_path / "odd.las"
    p.write_text(WRAPPED.replace("RHOB.G/C3", "BDCX.G/C3"))
    las = open_las(str(p))
    assert "BDCX" not in _POROSITY_NAMES          # not in any name list
    assert las.curves_by_unit("G/C3") == ["BDCX"]
    assert las.curves_by_unit(" g/c3 ") == ["BDCX"]     # tolerant
    assert las.curves_by_unit("OHMM") == []
    assert las.curves_by_unit()["GAPI"] == ["GR"]


_POROSITY_NAMES = ("NPHI", "PHIN", "TNPH", "RHOB", "DEN", "DPHI", "PEF")
