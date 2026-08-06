"""Tests for the opt-in geopressure/geometry EDMReader extension."""
import os

import numpy as np
import pytest

from welleng.exchange.edm_stream import EDM_SCHEMA, open_edm

HERE = os.path.dirname(__file__)
MINI = os.path.join(HERE, "test_data", "edm_geopressure_mini.xml")


@pytest.fixture
def edm():
    return open_edm(MINI, source_units="meters", with_geopressure=True)


# -- opt-in guard -------------------------------------------------------------
def test_off_by_default_and_guarded():
    e = open_edm(MINI, source_units="meters")  # with_geopressure defaults False
    assert not e._pp_groups
    with pytest.raises(RuntimeError):
        e.pore_pressure("WB1")


# -- pore pressure (canonical psi, phase filter, sorted, latest) --------------
def test_pore_pressure_actual_sorted_canonical(edm):
    profs = edm.pore_pressure("WB1", phase="ACTUAL")
    assert len(profs) == 1
    p = profs[0]
    assert p.phase == "ACTUAL"
    # sorted by TVD ascending; pressure is psi (canonical)
    np.testing.assert_allclose(p.tvd, [500, 1000])
    np.testing.assert_allclose(p.value, [240, 500])
    np.testing.assert_allclose(p.emw, [9.2, 9.6])  # RKB-view, carried verbatim


def test_pore_pressure_carries_permeable_flag(edm):
    p = edm.pore_pressure("WB1", phase="ACTUAL")[0]
    # fixture PPG1: tvd 500 is_permeable_zone=N, tvd 1000 =Y (sorted by tvd)
    assert p.permeable is not None
    np.testing.assert_array_equal(p.permeable, [False, True])
    # frac/temp carry no permeable flag
    assert edm.frac_gradient("WB1", phase="ACTUAL")[0].permeable is None


def test_well_position_by_id_and_name(edm):
    wp = edm.well_position("W1")
    assert wp.east == 1000 and wp.north == 2000
    assert wp.slot_ew == -3.5 and wp.slot_ns == 12.0
    assert edm.well_position("Test-1").well_id == "W1"  # resolves by common name
    with pytest.raises(KeyError):
        edm.well_position("nope")


def test_pore_pressure_phase_filter(edm):
    assert edm.pore_pressure("WB1", phase="PROTOTYPE")[0].value[0] == 1100
    assert edm.pore_pressure("WB1", phase="PLAN") == []  # none of that phase


def test_pore_pressure_all_phases_and_latest(edm):
    all_ph = edm.pore_pressure("WB1", phase=None, latest=False)
    assert {p.phase for p in all_ph} == {"ACTUAL", "PROTOTYPE"}
    # latest (newest update_date) is the 2014 ACTUAL group
    latest = edm.pore_pressure("WB1", phase=None, latest=True)
    assert len(latest) == 1 and latest[0].phase == "ACTUAL"


def test_frac_and_temperature(edm):
    fg = edm.frac_gradient("WB1", phase="ACTUAL")[0]
    assert fg.value[0] == 800 and fg.kind == "frac"
    tg = edm.temperature("WB1", phase="PROTOTYPE")[0]
    np.testing.assert_allclose(tg.tvd, [0, 2000])
    np.testing.assert_allclose(tg.value, [40, 205])  # degF, raw
    assert tg.emw is None and not tg.is_gradient_form


def test_temperature_gradient_attribute_form(edm):
    # a group with NO child rows, gradient slope on the group row (F-15B case)
    tg = edm.temperature("WB1", phase="PLAN")[0]
    assert tg.is_gradient_form
    assert tg.gradient == 0.015 and tg.reference_tvd == 200
    assert tg.reference_value == 40 and tg.surface_value == 80
    # shallow anchors surface->mudline, and value_at extrapolates below
    np.testing.assert_allclose(tg.tvd, [0, 200])
    np.testing.assert_allclose(tg.value, [80, 40])
    assert tg.value_at(1200) == pytest.approx(40 + 0.015 * (1200 - 200))


# -- geometry (dedup across scenarios) + formations ---------------------------
def test_geometry_dedups_and_sorts(edm):
    geo = edm.geometry("WB1")
    # HG1/HG2 duplicate the 20in section -> collapsed to one; OPEN kept
    assert len(geo) == 2
    assert [h.sect_type_code for h in geo] == ["CAS", "OPEN"]
    cas = geo[0]
    assert cas.od_casing == 20.0 and cas.id_drift == 18.543 and cas.md_shoe == 500
    assert geo[1].od_casing is None  # empty numeric -> None


def test_geometry_scenario_groups_and_selection(edm):
    groups = edm.hole_section_groups("WB1")
    by_id = {g.group_id: g for g in groups}
    assert set(by_id) == {"HG1", "HG2"}
    # HG1 (2 sections) linked to the ACTUAL "As-run" case; HG2 to PLAN "Plan"
    assert by_id["HG1"].n_sections == 2
    assert by_id["HG1"].case_names == ["As-run"] and by_id["HG1"].phases == ["ACTUAL"]
    assert by_id["HG2"].phases == ["PLAN"]
    # selecting a group returns that scenario's sections only, NOT deduped
    g1 = edm.geometry("WB1", group_id="HG1")
    assert {h.sect_type_code for h in g1} == {"CAS", "OPEN"} and len(g1) == 2
    g2 = edm.geometry("WB1", group_id="HG2")
    assert len(g2) == 1 and g2[0].sect_type_code == "CAS"  # the duplicate 20in
    # pooled (no group) still collapses the cross-scenario duplicate
    assert len(edm.geometry("WB1")) == 2


def test_formations_by_md(edm):
    fms = edm.formations("WB1")
    assert [f.name for f in fms] == ["Hod_Fm_Top", "Hugin_Fm_Top"]  # by MD
    assert fms[0].md == 900 and fms[0].tvd == 880


# -- catalogue + datum set ----------------------------------------------------
def test_grade_material_catalogue(edm):
    assert edm.grades["P110"]["grade"] == "P-110"
    assert float(edm.materials["MAT1"]["density"]) == pytest.approx(489.9)


def test_datum_set_not_collapsed(edm):
    ds = edm.datum_set("W1")
    assert len(ds) == 2  # per-rig, dated -- not collapsed to one RT
    assert {d["datum_name"] for d in ds} == {"Rig Alpha RT", "Rig Beta RT"}
    ds_by_name = edm.datum_set("Test-1")  # resolve by well_common_name too
    assert len(ds_by_name) == 2


# -- schema (human-readable names) --------------------------------------------
def test_schema_is_human_readable(edm):
    assert edm.schema is EDM_SCHEMA
    assert edm.schema["CD_PORE_PRESSURE"]["name"] == "Pore pressure"
    assert "canonical" in edm.schema["CD_PORE_PRESSURE"]["description"]
    assert edm.schema["CD_HOLE_SECT"]["fields"]["md_shoe"] == "shoe MD"
