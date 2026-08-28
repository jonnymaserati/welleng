"""Tests for the opt-in CD_ASSEMBLY / CD_ASSEMBLY_COMP reader (the ratings oracle)."""
import os

import pytest

from welleng.exchange.edm_stream import open_edm

HERE = os.path.dirname(__file__)
MINI = os.path.join(HERE, "test_data", "edm_geopressure_mini.xml")


@pytest.fixture
def edm():
    return open_edm(MINI, source_units="meters", with_assemblies=True)


def test_off_by_default_and_guarded():
    e = open_edm(MINI, source_units="meters")
    assert not e._assemblies
    with pytest.raises(RuntimeError):
        e.assemblies()


def test_assemblies_and_components(edm):
    asms = edm.assemblies("WB1")
    assert [a.name for a in asms] == ["9 5/8\" Production", "8 1/2\" BHA"]  # by MD base
    casing = asms[0]
    assert casing.string_type == "Casing"
    assert casing.size == 9.625 and casing.hole_size == 12.25
    assert len(casing.components) == 1


def test_casing_component_carries_stored_ratings_oracle(edm):
    comp = edm.assemblies("WB1")[0].components[0]
    assert comp.sect_type_code == "CAS" and comp.grade == "P-110"
    assert comp.od_body == 9.625 and comp.id_body == 8.535
    # Landmark's stored ratings, verbatim (the oracle)
    assert comp.axial_rating == 1710.06
    assert comp.pipe_pressure_burst == 10900
    assert comp.pressure_collapse == 7950
    # derating input carried in raw, not misread as a result
    assert comp.raw["critical_percent_collapse"] == "100.0"


def test_drillpipe_component_has_no_ratings(edm):
    bha = edm.assemblies("WB1")[1]
    assert bha.string_type == "Drillstring"
    dp = bha.components[0]
    assert dp.sect_type_code == "DP"
    assert dp.axial_rating is None  # unrated
    assert dp.pipe_pressure_burst is None
    assert dp.pressure_collapse is None
    assert dp.min_yield_stress == 135000  # geometry/material still present


def test_all_assemblies_without_wellbore_filter(edm):
    assert len(edm.assemblies()) == 2


def test_schema_documents_the_oracle(edm):
    f = edm.schema["CD_ASSEMBLY_COMP"]["fields"]
    assert f["axial_rating"] == "stored pipe-body yield (klbf)"
    assert "not a result" in f["critical_percent_collapse"]


def test_component_surfaces_rich_geometry_connection_material_fields(edm):
    # the BHA/T&D-relevant fields that used to live only in .raw are now named
    dp = edm.assemblies("WB1")[1].components[0]
    assert dp.description == "Drill Pipe 5.5 in S-135"
    assert dp.length == 9.5
    assert dp.approximate_weight == 24.98          # surfaced RAW (unverified nominal)
    assert dp.od_connection == 7.0 and dp.id_connection == 3.75
    assert dp.length_tool_joint == 0.42
    assert dp.connection_name == "5 1/2 FH"
    assert dp.material == "steel" and dp.material_id == "STL01"
    assert dp.density == 7850.0
    assert dp.youngs_modulus == 30000000.0 and dp.poissons_ratio == 0.3
    assert dp.makeup_torque == 56261.0


def test_component_rich_fields_default_none_when_absent(edm):
    # the casing comp (A1) has none of the new attrs -> clean defaults, no crash
    cas = edm.assemblies("WB1")[0].components[0]
    assert cas.description == "" and cas.connection_name == ""
    assert cas.approximate_weight is None and cas.length_tool_joint is None
    assert cas.youngs_modulus is None
