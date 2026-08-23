"""Tests for the streaming EDM parser (welleng.exchange.edm_stream) plus the
back-compat / mutable-default fixes in welleng.exchange.edm.
"""
import warnings

import numpy as np
import pytest

from welleng.exchange.edm_stream import (
    EDMReader,
    classify_tool,
    ToolKind,
    FEET_TO_METERS,
)

# A small synthetic EDM export exercising the key tables. Mirrors the real
# Volve layout: the CD_* rows are *direct* children of <export>, with a nested
# <TOPLEVEL> holding some non-survey (TU_*) rows -- so both the streaming
# reader (depth-agnostic, matches by tag) and the legacy DOM EDM (iterates the
# root's direct children) see the same CD_* data.
#   project / well / two wellbores (parent + sidetrack) /
#   raw survey header (+ tie) with raw stations /
#   definitive survey headers (ACTUAL + PLAN) with covariance stations /
#   survey program (two tool intervals: gyro then mwd) / two tools.
SYNTHETIC_EDM = """<?xml version="1.0" standalone="no"?>
<export>
<TOPLEVEL>
<TU_TEMP_DERATION_SCHED schedule_name="ignored" temp_deration_sched_id="X1" />
</TOPLEVEL>
<CD_PROJECT project_id="P1" project_name="TEST" />
<CD_SITE site_id="S1" project_id="P1" site_name="TestSite" />
<CD_WELL well_id="W1" site_id="S1" well_common_name="T-1" />
<CD_WELLBORE well_id="W1" wellbore_id="WB0" wellbore_name="T-1 main" />
<CD_WELLBORE well_id="W1" wellbore_id="WB1" wellbore_name="T-1 ST1" parent_wellbore_id="WB0" />
<DP_MAGNETIC well_id="W1" wellbore_id="WB1" magnetic_model_id="M0" sequence_no="0" field_strength="50500" dip_angle="72.1" declination="-1.5" declination_date="{ts '2014-01-01'}" model_name="BGGM2014" />
<DP_MAGNETIC well_id="W1" wellbore_id="WB1" magnetic_model_id="M1" sequence_no="1" field_strength="50550" dip_angle="72.2" declination="-1.6" declination_date="{ts '2015-01-01'}" model_name="BGGM2015" />
<CD_SURVEY_TOOL survey_tool_id="TG" tool_name="Keeper, cont" description="Gyro Tool from SDC" tool_type="0" />
<CD_SURVEY_TOOL survey_tool_id="TM" tool_name="Magnetic, std, non-mag" description="Magnetic Tools (MWD, EMS)" tool_type="0" />
<CD_SURVEY_HEADER well_id="W1" wellbore_id="WB0" survey_header_id="SH0" phase="ACTUAL" survey_name="root raw" md_min="0" md_max="100" />
<CD_SURVEY_HEADER well_id="W1" wellbore_id="WB1" survey_header_id="SH1" phase="ACTUAL" survey_name="st raw" tie_survey_header_id="SH0" survey_tool_id="TM" md_min="100" md_max="200" />
<CD_SURVEY_STATION well_id="W1" wellbore_id="WB0" survey_header_id="SH0" md="0" inclination="0" azimuth="0" tvd="0" offset_north="0" offset_east="0" dogleg_severity="0" sequence_no="0" />
<CD_SURVEY_STATION well_id="W1" wellbore_id="WB0" survey_header_id="SH0" md="100" inclination="5" azimuth="10" tvd="99" offset_north="4" offset_east="1" dogleg_severity="1.5" sequence_no="1" />
<CD_SURVEY_STATION well_id="W1" wellbore_id="WB1" survey_header_id="SH1" md="150" inclination="20" azimuth="30" tvd="145" offset_north="20" offset_east="12" dogleg_severity="2.0" sequence_no="0" />
<CD_DEFINITIVE_SURVEY_HEADER well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" phase="ACTUAL" name="Wellpath" tie_survey_header_id="SH0" />
<CD_DEFINITIVE_SURVEY_HEADER well_id="W1" wellbore_id="WB1" def_survey_header_id="DH2" phase="PLAN" name="Plan" />
<CD_DEFINITIVE_SURVEY_STATION well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" md="50" inclination="2" azimuth="5" tvd="49.9" offset_north="2" offset_east="1" dogleg_severity="0.5" sequence_no="0" covariance_xx="4" covariance_xy="1" covariance_xz="2" covariance_yy="9" covariance_yz="3" covariance_zz="16" ellipse_north="1" ellipse_east="2" ellipse_vertical="3" />
<CD_DEFINITIVE_SURVEY_STATION well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" md="150" inclination="20" azimuth="30" tvd="145" offset_north="20" offset_east="12" dogleg_severity="2.0" sequence_no="1" covariance_xx="8" covariance_xy="0" covariance_xz="0" covariance_yy="18" covariance_yz="0" covariance_zz="32" ellipse_north="1" ellipse_east="2" ellipse_vertical="3" />
<CD_DEFINITIVE_SURVEY_STATION well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" md="190" inclination="35" azimuth="40" tvd="180" offset_north="30" offset_east="22" dogleg_severity="1.0" sequence_no="2" covariance_xx="12" covariance_xy="0" covariance_xz="0" covariance_yy="24" covariance_yz="0" covariance_zz="40" ellipse_north="1" ellipse_east="2" ellipse_vertical="3" />
<CD_DEFINITIVE_SURVEY_STATION well_id="W1" wellbore_id="WB1" def_survey_header_id="DH2" md="0" inclination="0" azimuth="0" tvd="0" offset_north="0" offset_east="0" dogleg_severity="0" sequence_no="0" covariance_xx="1" covariance_xy="0" covariance_xz="0" covariance_yy="1" covariance_yz="0" covariance_zz="1" />
<CD_SURVEY_PROGRAM well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" survey_program_id="PR0" survey_header_id="SH0" survey_tool_id="TG" md_top="0" md_base="100" sequence_no="0" not_in_use="0" />
<CD_SURVEY_PROGRAM well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" survey_program_id="PR1" survey_header_id="SH1" survey_tool_id="TM" md_top="100" md_base="200" sequence_no="1" not_in_use="0" />
<!-- a non-survey table that must be skipped by the streaming index -->
<CD_PORE_PRESSURE well_id="W1" pore_pressure_id="PP1" tvd="1000" pore_pressure="1.2" />
</export>
"""


@pytest.fixture
def edm_file(tmp_path):
    p = tmp_path / "synthetic.xml"
    p.write_text(SYNTHETIC_EDM)
    return str(p)


@pytest.fixture
def reader(edm_file):
    return EDMReader.open(edm_file)


# --------------------------------------------------------------------------
# classify_tool
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name,desc,expected", [
    ("Gyro, cont", "Other Gyro Tools", ToolKind.GYRO),
    ("Keeper, cont", "Gyro Tool from SDC", ToolKind.GYRO),
    ("Wellbore Surveyor, stat", "Gyro Tool from GD", ToolKind.GYRO),
    ("RIGS, cont", "Inertial Tool from BHI", ToolKind.GYRO),
    # gyro-while-drilling: gyro sensor wins over the MWD conveyance
    ("MWD gyro, Gyrodata, GWD70", "Gyrodata's MWD gyro service GWD70",
     ToolKind.GYRO),
    ("Magnetic, std, non-mag", "Magnetic Tools (MWD, EMS)", ToolKind.MWD),
    ("Magn, IFR, mag-corr, dual incl", "Magnetic Tools (MWD, EMS)",
     ToolKind.MWD),
    ("Inclination Only", "Inclination only surveys, vertical wells",
     ToolKind.INCLINATION_ONLY),
    ("Definitive combined", "", ToolKind.DEFINITIVE),
    ("Blind Drilling", "No survey recorded - blind drilling", ToolKind.OTHER),
    ("Dummy Tool", "", ToolKind.OTHER),
    (None, None, ToolKind.OTHER),
    ("", "", ToolKind.OTHER),
])
def test_classify_tool(name, desc, expected):
    assert classify_tool(name, desc) is expected


def test_classify_tool_is_pure():
    # no exceptions on odd input; deterministic
    assert classify_tool("gyro") is ToolKind.GYRO
    assert classify_tool("GYRO") is ToolKind.GYRO  # case-insensitive


# --------------------------------------------------------------------------
# index sweep
# --------------------------------------------------------------------------
def test_index_counts(reader):
    assert len(reader.projects) == 1
    assert len(reader.sites) == 1
    assert len(reader.wells) == 1
    assert len(reader.wellbores) == 2
    assert len(reader.tools) == 2
    # 2 raw headers + 2 definitive headers
    assert len(reader.headers) == 4
    assert set(reader.wellbore_name_to_id) == {"T-1 main", "T-1 ST1"}


def test_index_skips_non_survey_tables(reader):
    # CD_PORE_PRESSURE must not be materialised anywhere in the index
    for attr in ("projects", "sites", "wells"):
        for row in getattr(reader, attr).values():
            assert "pore_pressure" not in row


def test_tool_classification_on_index(reader):
    assert reader.tools["TG"].kind is ToolKind.GYRO
    assert reader.tools["TM"].kind is ToolKind.MWD


def test_station_counts_on_headers(reader):
    hdr = reader.headers["DH1"]
    assert hdr.kind == "definitive"
    assert hdr.n_stations == 3
    assert reader.headers["SH1"].kind == "raw"
    assert reader.headers["SH1"].n_stations == 1


# --------------------------------------------------------------------------
# survey assembly + covariance
# --------------------------------------------------------------------------
def test_definitive_survey_covariance_3x3(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    assert len(s) == 3
    assert s.has_covariance
    st = s.stations[0]  # md=50
    assert st.covariance.shape == (3, 3)
    # native order [[xx,xy,xz],[xy,yy,yz],[xz,yz,zz]]
    expected = np.array([[4, 1, 2], [1, 9, 3], [2, 3, 16]], dtype=float)
    np.testing.assert_array_equal(st.covariance, expected)
    assert np.allclose(st.covariance, st.covariance.T)


def test_definitive_survey_sorted_by_md(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    mds = [st.md for st in s.stations]
    assert mds == sorted(mds)
    assert mds == [50.0, 150.0, 190.0]


def test_raw_survey_has_no_covariance(reader):
    s = reader.survey("T-1 ST1", kind="raw", phase="ACTUAL")
    assert not s.has_covariance
    assert all(st.covariance is None for st in s.stations)


# --------------------------------------------------------------------------
# tool resolution per interval (SURVEY_PROGRAM)
# --------------------------------------------------------------------------
def test_tool_resolution_per_interval(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    # md=50 falls in [0,100] -> gyro; md=150,190 fall in [100,200] -> mwd
    by_md = {st.md: st for st in s.stations}
    assert by_md[50].tool.kind is ToolKind.GYRO
    assert by_md[50].tool.name == "Keeper, cont"
    assert by_md[150].tool.kind is ToolKind.MWD
    assert by_md[190].tool.kind is ToolKind.MWD


def test_raw_survey_single_tool_from_header(reader):
    # SH1 header carries survey_tool_id="TM" -> all raw stations get it
    s = reader.survey("T-1 ST1", kind="raw", phase="ACTUAL")
    assert all(st.tool is not None and st.tool.kind is ToolKind.MWD
               for st in s.stations)


# --------------------------------------------------------------------------
# phase filtering + header selection
# --------------------------------------------------------------------------
def test_phase_filtering(reader):
    actual = reader.survey_headers("T-1 ST1", kind="definitive",
                                   phase="ACTUAL")
    plan = reader.survey_headers("T-1 ST1", kind="definitive", phase="PLAN")
    assert [h.header_id for h in actual] == ["DH1"]
    assert [h.header_id for h in plan] == ["DH2"]


def test_default_header_is_most_stations(reader):
    # no phase filter: DH1 (3 stations) should sort ahead of DH2 (1 station)
    hdrs = reader.survey_headers("T-1 ST1", kind="definitive")
    assert hdrs[0].header_id == "DH1"
    assert hdrs[0].n_stations >= hdrs[-1].n_stations


def test_missing_survey_raises(reader):
    with pytest.raises(LookupError):
        reader.survey("T-1 main", kind="definitive", phase="ACTUAL")


# --------------------------------------------------------------------------
# feet -> meters conversion
# --------------------------------------------------------------------------
def test_to_welleng_feet_to_meters(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    sv = s.to_welleng(units="meters")
    # depths scaled by 0.3048
    assert np.isclose(sv.md[-1], 190.0 * FEET_TO_METERS)
    assert np.isclose(sv.md[0], 50.0 * FEET_TO_METERS)
    assert len(sv.md) == 3


def test_to_welleng_feet_to_feet_no_assert(reader):
    # regression: to_welleng(units="feet") on a feet source used to raise
    # "inconsistent units with header" -- the built SurveyHeader didn't set
    # depth_unit=units, so Survey's assert (unit == header.depth_unit) fired.
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    sv = s.to_welleng(units="feet")   # must NOT raise
    assert sv.unit == "feet" and sv.header.depth_unit == "feet"
    # feet source -> feet target: depths unchanged (no conversion)
    assert np.isclose(sv.md[-1], 190.0) and np.isclose(sv.md[0], 50.0)
    # and it round-trips through interpolate_md (a Node, not None)
    node = sv.interpolate_md(float(sv.md[1]))
    assert node is not None and np.isclose(node.md, sv.md[1])


def test_to_welleng_covariance_reordered_and_scaled(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    sv = s.to_welleng(units="meters")
    assert sv.cov_nev is not None
    assert sv.cov_nev.shape == (3, 3, 3)
    # native (x=E,y=N,z=V) -> NEV: nn=yy, ne=xy, nv=yz, ee=xx, ev=xz, vv=zz
    scale = FEET_TO_METERS ** 2
    expected_nev = np.array(
        [[9, 1, 3], [1, 4, 2], [3, 2, 16]], dtype=float
    ) * scale
    np.testing.assert_allclose(sv.cov_nev[0], expected_nev)


def test_as_arrays_native_units(reader):
    s = reader.survey("T-1 ST1", kind="definitive", phase="ACTUAL")
    arr = s.as_arrays()
    # native feet, unconverted
    np.testing.assert_array_equal(arr["md"], [50.0, 150.0, 190.0])
    assert arr["covariance"].shape == (3, 3, 3)
    assert arr["tool_kind"][0] is ToolKind.GYRO


# --------------------------------------------------------------------------
# sidetrack + tie chains, and no mutable-default leakage across calls
# --------------------------------------------------------------------------
def test_sidetrack_chain(reader):
    chain = reader.sidetrack_chain("T-1 ST1")
    assert [wb.wellbore_id for wb in chain] == ["WB0", "WB1"]


def test_tie_chain(reader):
    chain = reader.tie_chain("SH1")
    assert [h.header_id for h in chain] == ["SH0", "SH1"]


def test_chains_no_mutable_default_leak(reader):
    # repeated calls must return identical results (no accumulation)
    first = [wb.wellbore_id for wb in reader.sidetrack_chain("T-1 ST1")]
    second = [wb.wellbore_id for wb in reader.sidetrack_chain("T-1 ST1")]
    third = [wb.wellbore_id for wb in reader.sidetrack_chain("T-1 ST1")]
    assert first == second == third == ["WB0", "WB1"]

    t1 = [h.header_id for h in reader.tie_chain("SH1")]
    t2 = [h.header_id for h in reader.tie_chain("SH1")]
    assert t1 == t2 == ["SH0", "SH1"]


def test_sidetrack_chain_cycle_guard(reader):
    # induce a self-cycle; must terminate, not recurse forever
    reader.wellbores["WB0"].parent_wellbore_id = "WB0"
    chain = reader.sidetrack_chain("WB0")
    assert [wb.wellbore_id for wb in chain] == ["WB0"]


# --------------------------------------------------------------------------
# back-compat: old DOM EDM still works + mutable-default fixes
# --------------------------------------------------------------------------
def test_old_edm_still_imports_and_parses(edm_file):
    from welleng.exchange.edm import EDM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        edm = EDM(edm_file)
    assert "T-1 main" in edm.wellbore_name_to_id
    assert edm.wellbore_id_to_name["WB1"] == "T-1 ST1"


def test_old_edm_emits_deprecation_warning(edm_file):
    from welleng.exchange.edm import EDM
    with pytest.warns(DeprecationWarning):
        EDM(edm_file)


def test_old_edm_get_parents_no_mutable_default_leak(edm_file):
    from welleng.exchange.edm import EDM
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        edm = EDM(edm_file)
    first = edm.get_parents("WB1")
    second = edm.get_parents("WB1")
    # the pre-fix bug would return ["WB0", "WB0"] on the second call
    assert first == ["WB0"]
    assert second == ["WB0"]


def test_edm_open_returns_streaming_reader(edm_file):
    from welleng.exchange.edm import EDM
    r = EDM.open(edm_file)
    assert isinstance(r, EDMReader)
    assert len(r.wellbores) == 2


# --------------------------------------------------------------------------
# DP_MAGNETIC — per-wellbore geomagnetic reference accessor
# --------------------------------------------------------------------------
def test_magnetics_accessor_returns_operative_row(reader):
    m = reader.magnetics("T-1 ST1")            # WB1, resolved by name
    assert m is not None
    # highest sequence_no wins (the operative reference)
    assert m.sequence_no == 1
    assert m.b_total == 50550.0
    assert m.model == "BGGM2015"
    assert m.dip == pytest.approx(72.2)
    assert m.declination == pytest.approx(-1.6)


def test_magnetics_all_ordered_by_sequence(reader):
    allm = reader.magnetics_all("WB1")
    assert [x.sequence_no for x in allm] == [0, 1]
    assert allm[0].model == "BGGM2014"


def test_magnetics_none_when_absent(reader):
    assert reader.magnetics("WB0") is None     # root wellbore has no DP_MAGNETIC


# --------------------------------------------------------------------------
# tool remarks + CD_SURVEY_PROGRAM tool header-fallback (Volve F-12 gap)
# --------------------------------------------------------------------------
_EDM_TOOL_FALLBACK = """<?xml version="1.0" standalone="no"?>
<export>
<CD_PROJECT project_id="P1" project_name="T" />
<CD_SITE site_id="S1" project_id="P1" site_name="S" />
<CD_WELL well_id="W1" site_id="S1" well_common_name="T-1" />
<CD_WELLBORE well_id="W1" wellbore_id="WB1" wellbore_name="T-1" />
<CD_SURVEY_TOOL survey_tool_id="TM" tool_name="Magnetic, std, mag-corr" \
description="Magnetic" remarks="MWD-std-mag" tool_type="0" correlate="Y" \
is_range_limited="Y" inclination_range_min="5" inclination_range_max="90" />
<CD_SURVEY_HEADER well_id="W1" wellbore_id="WB1" survey_header_id="SH1" \
phase="ACTUAL" survey_name="raw" survey_tool_id="TM" md_min="0" md_max="200" />
<CD_SURVEY_PROGRAM well_id="W1" wellbore_id="WB1" def_survey_header_id="DH1" \
survey_program_id="PR0" survey_header_id="SH1" md_top="0" md_base="200" \
sequence_no="0" not_in_use="0" />
</export>
"""


def _fallback_reader(tmp_path):
    p = tmp_path / "fallback.xml"
    p.write_text(_EDM_TOOL_FALLBACK)
    return EDMReader.open(str(p))


def test_survey_tool_remarks_captured(tmp_path):
    tool = _fallback_reader(tmp_path).tools["TM"]
    assert tool.remarks == "MWD-std-mag"
    assert tool.kind is ToolKind.MWD
    assert tool.correlate is True                    # correlate="Y"
    assert tool.is_range_limited is True
    assert tool.inclination_range_min == 5.0
    assert tool.inclination_range_max == 90.0


def test_survey_program_tool_falls_back_to_header(tmp_path):
    # CD_SURVEY_PROGRAM omits survey_tool_id -> resolve via the linked
    # CD_SURVEY_HEADER's tool (Volve F-12 definitive carried no program tool).
    runs = _fallback_reader(tmp_path).survey_runs("WB1")
    assert len(runs) == 1
    run = runs[0]
    assert run.survey_tool_id == "TM"          # resolved via header fallback
    assert run.tool_name == "Magnetic, std, mag-corr"
    assert run.tool_remarks == "MWD-std-mag"
    assert run.tool_kind == "MWD"


def test_classify_tool_remarks_tiebreak():
    # name + description inconclusive; the remark carries the gyro signal
    assert classify_tool("Tool 7", "no signal", "old gyro run") is ToolKind.GYRO
    assert classify_tool(None, None, None) is ToolKind.OTHER
