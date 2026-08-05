"""Tests for the EPP / DLS-override / fluid-rheology EDM readers."""
import os

import pytest

from welleng.exchange.edm_stream import open_edm

HERE = os.path.dirname(__file__)
MINI = os.path.join(HERE, "test_data", "edm_geopressure_mini.xml")


@pytest.fixture
def edm():
    return open_edm(MINI, source_units="meters",
                    with_load_cases=True, with_geopressure=True)


# -- EPP (per-load) -----------------------------------------------------------
def test_epp_is_per_load_with_differing_values(edm):
    epp = edm.epp_parameters()
    assert set(epp) == {"L1", "L2"}
    # same parameter, genuinely different value per load (design's gotcha)
    assert epp["L1"]["EXT_PROF_FluidGradientBelowTOC"] == 8.33
    assert epp["L2"]["EXT_PROF_FluidGradientBelowTOC"] == 10.85


# -- DLS override -------------------------------------------------------------
def test_dls_override_span_handles_inverted_interval(edm):
    ov = edm.dls_overrides("WB1")
    assert len(ov) == 2
    # the inverted row (md_top=3000 > md_base=2000) is ordered by span
    inverted = next(d for d in ov if d.md_top == 3000)
    assert inverted.span == (2000, 3000)
    assert inverted.dogleg_severity == 4.0


def test_dls_override_present_for_wb1(edm):
    # WB1 has overrides; a wellbore with none returns [] (survey DLS governs)
    assert edm.dls_overrides("WB1")


# -- fluid rheology -----------------------------------------------------------
def test_fluid_rheology_skips_incomplete_and_joins_fann(edm):
    fl = edm.fluid_rheology()
    # F1/T2 has no rheology numbers -> skipped; only F1/T1 survives
    assert len(fl) == 1
    p = fl[0]
    assert p.fluid_id == "F1" and p.temperature == 122
    assert p.plastic_viscosity == 16.5 and p.yield_point == 9.6
    # Fann data joined on (fluid_id, temp_id)
    assert sorted(p.fann) == [(300.0, 26.1), (600.0, 42.6)]


def test_fluid_rheology_filter_by_fluid(edm):
    assert edm.fluid_rheology(fluid_id="F1")
    assert edm.fluid_rheology(fluid_id="NOPE") == []


# -- guards -------------------------------------------------------------------
def test_guards():
    e = open_edm(MINI, source_units="meters")  # nothing opted in
    with pytest.raises(RuntimeError):
        e.epp_parameters()
    with pytest.raises(RuntimeError):
        e.dls_overrides("WB1")
    with pytest.raises(RuntimeError):
        e.fluid_rheology()
