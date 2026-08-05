"""Tests for the opt-in TU_LOAD_* design load-case readers."""
import os

import pytest

from welleng.exchange.edm_stream import open_edm

HERE = os.path.dirname(__file__)
MINI = os.path.join(HERE, "test_data", "edm_geopressure_mini.xml")


@pytest.fixture
def edm():
    return open_edm(MINI, source_units="meters", with_load_cases=True)


def test_off_by_default_and_guarded():
    e = open_edm(MINI, source_units="meters")
    assert not e._load_profiles
    with pytest.raises(RuntimeError):
        e.load_profiles()


def test_load_profiles_grouped_with_default_none(edm):
    # StressCheck filter (default) excludes the WELLPLAN row
    lps = edm.load_profiles()
    assert [lp.profile_name for lp in lps] == ["BrstGasKickProfile"]
    params = lps[0].parameters
    assert params["DesignPipeBurstFactor"] == 1.1     # valued
    assert params["CMN_GasGravity"] is None           # absent -> default, NOT 0


def test_application_filter(edm):
    assert len(edm.load_profiles(application=None)) == 2   # both apps
    assert [lp.profile_name for lp in edm.load_profiles(application="WELLPLAN")] \
        == ["OtherApp"]


def test_custom_load_profile_sorted_with_differential(edm):
    cps = edm.custom_load_profiles()
    assert len(cps) == 1
    pts = cps[0].points
    assert [p.md for p in pts] == [1000, 2000]        # sorted by MD
    assert pts[0].differential_pressure == -1000       # 2000 - 3000 (collapse)
    assert pts[1].differential_pressure == 2000        # 5000 - 3000 (burst)


def test_differential_none_when_either_missing():
    from welleng.exchange.edm_stream import CustomLoadPoint
    assert CustomLoadPoint(100, None, 3000).differential_pressure is None


def test_load_headers_sequence_ordered(edm):
    hs = edm.load_headers()
    assert [h.name for h in hs] == ["L.1.Pressure test", "L.3.Gas kick"]  # by seq
    assert hs[0].load_category == 0 and hs[1].load_type == 3


def test_schema_documents_default_semantics(edm):
    d = edm.schema["TU_LOAD_PROFILE"]["fields"]["parameter_value_num"]
    assert "default" in d
