"""Tests for the WITSML 1.4.1.1 reader (welleng.exchange.witsml)."""
import os
import zipfile

import numpy as np
import pytest

from welleng.exchange.witsml import (
    WITSMLReader,
    WITSMLVersionError,
    open_witsml,
)

HERE = os.path.dirname(__file__)
FIXTURE_DIR = os.path.join(HERE, "test_data", "witsml")


@pytest.fixture
def reader():
    return open_witsml(FIXTURE_DIR)


@pytest.fixture
def zip_reader(tmp_path):
    """Same fixtures packed into a SiteCom-style zip layout."""
    zp = tmp_path / "witsml.zip"
    with zipfile.ZipFile(zp, "w") as zf:
        base = "site/Test-Field-A/A-1"
        zf.write(os.path.join(FIXTURE_DIR, "time_log.xml"), f"{base}/log/1.xml")
        zf.write(os.path.join(FIXTURE_DIR, "depth_log.xml"), f"{base}/log/2.xml")
        zf.write(os.path.join(FIXTURE_DIR, "tubular.xml"), f"{base}/tubular/1.xml")
    return open_witsml(str(zp))


# -- indexing -----------------------------------------------------------------
def test_indexes_both_logs_not_the_tubular(reader):
    names = sorted(li.name for li in reader.logs)
    assert names == ["SurfaceData", "TempGradient"]


def test_log_header_fields(reader):
    surf = next(li for li in reader.logs if li.name == "SurfaceData")
    assert surf.well == "Test-Field-A"
    assert surf.wellbore == "A-1"
    assert surf.is_time_indexed
    assert surf.index_mnemonic == "Time"
    assert surf.mnemonics == ("Time", "HKLD", "MTIN")
    assert surf.null_value == -999.25


def test_wells(reader):
    assert reader.wells == ["Test-Field-A"]


def test_version_detected(reader):
    _ = reader.logs
    assert reader.version == "1.4.1.1"


# -- find ---------------------------------------------------------------------
def test_find_channel(reader):
    hits = reader.find("HKLD")
    assert len(hits) == 1 and hits[0].name == "SurfaceData"
    assert reader.find("ATMP_RT")[0].name == "TempGradient"
    assert reader.find("NOPE") == []


def test_find_well_filter(reader):
    assert reader.find("HKLD", well="Test-Field-A")
    assert reader.find("HKLD", well="Other-Field") == []


# -- curves -------------------------------------------------------------------
def test_time_curves_and_null_to_nan(reader):
    surf = next(li for li in reader.logs if li.name == "SurfaceData")
    c = surf.curves()
    assert c["Time"].dtype == np.dtype("datetime64[ns]")
    assert len(c["Time"]) == 3
    np.testing.assert_allclose(c["HKLD"], [120.5, 121.0, 119.8])
    # -999.25 null must map to NaN
    assert np.isnan(c["MTIN"][1])
    np.testing.assert_allclose(c["MTIN"][[0, 2]], [45.0, 46.2])


def test_depth_curves(reader):
    tg = next(li for li in reader.logs if li.name == "TempGradient")
    c = tg.curves(["ATMP_RT"])
    # index channel always included even when not requested
    assert set(c) == {"DEPTH", "ATMP_RT"}
    np.testing.assert_allclose(c["DEPTH"], [1000, 1100, 1200])
    np.testing.assert_allclose(c["ATMP_RT"], [50.0, 68.5, 87.0])
    assert c["DEPTH"].dtype == float  # depth index is numeric, not datetime


# -- tubulars -----------------------------------------------------------------
def test_tubulars_sorted_bottom_up(reader):
    tubs = reader.tubulars()
    assert len(tubs) == 1
    t = tubs[0]
    assert t.name == "8.5in BHA"
    assert [c.sequence for c in t.components] == [1, 2]  # bit first
    bit, dp = t.components
    assert bit.type == "bit"
    assert dp.od_in == pytest.approx(0.127 / 0.0254)  # 5.0 in
    assert dp.wt_kgm == pytest.approx(29.05)


# -- zip path -----------------------------------------------------------------
def test_zip_layout_equivalent(zip_reader):
    assert sorted(li.name for li in zip_reader.logs) == \
        ["SurfaceData", "TempGradient"]
    surf = zip_reader.find("HKLD")[0]
    # well/wellbore recovered from the SiteCom .../<well>/<wellbore>/log/ path
    assert surf.well == "Test-Field-A"
    assert surf.wellbore == "A-1"
    np.testing.assert_allclose(surf.curves()["HKLD"], [120.5, 121.0, 119.8])
    assert len(zip_reader.tubulars()) == 1


# -- version guard ------------------------------------------------------------
def test_version_guard(tmp_path):
    bad = tmp_path / "v2.xml"
    bad.write_text(
        '<logs xmlns="http://www.energistics.org/schemas/witsml/v2.0"'
        ' version="2.0"><log><name>X</name><indexType>date time</indexType>'
        '<logData><mnemonicList>Time</mnemonicList>'
        '<data>2018-01-01T00:00:00Z</data></logData></log></logs>'
    )
    r = WITSMLReader.open(str(tmp_path))
    with pytest.raises(WITSMLVersionError):
        _ = r.logs
    # opt-out reads best-effort
    r2 = WITSMLReader.open(str(tmp_path), version_check=False)
    assert any(li.name == "X" for li in r2.logs)
