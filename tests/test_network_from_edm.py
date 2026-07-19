"""Tests for welleng.hierarchy.network_from_edm (EDM -> WellNetwork adapter).

A small synthetic EDM XML fixture exercises the mapping (entities, parent
linkage, unit conversion, optional survey attachment, OSDU round-trip); the
public Volve export (Equinor, CC BY 4.0) is exercised when available locally.
"""
import math
import os
import pathlib
import warnings

import pytest

from welleng import osdu
from welleng.exchange.edm_stream import EDMReader, FEET_TO_METERS
from welleng.hierarchy import Datum, Site, Well, Wellbore, network_from_edm

ROOT = pathlib.Path(__file__).resolve().parent.parent
VOLVE_XML = os.environ.get("WELLENG_VOLVE_XML", str(ROOT / "data" / "Volve.xml"))

FIXTURE_XML = """<export><TOPLEVEL>
<CD_PROJECT project_id="P1" project_name="TESTFIELD" geo_datum_id="ED50"
    geo_zone_id="UTM-31N" />
<CD_SITE site_id="S1" project_id="P1" site_name="Pad A" convergence="-1.0"
    is_field_center="Y" geo_map_northing="10000.0" geo_map_easting="20000.0" />
<CD_WELL well_id="W1" site_id="S1" well_common_name="T-1" slot_ns="10.0"
    slot_ew="-20.0" slot_radial_error="1.0" wellhead_depth="300.0" />
<CD_DATUM datum_id="D1" well_id="W1" datum_name="Rotary Table"
    datum_elevation="100.0" is_default="Y" />
<CD_DATUM datum_id="D2" well_id="W1" datum_name="Sea Level"
    datum_elevation="0.0" is_default="N" />
<CD_WELLBORE wellbore_id="WB1" well_id="W1" wellbore_name="T-1 main" />
<CD_WELLBORE wellbore_id="WB2" well_id="W1" wellbore_name="T-1 A"
    parent_wellbore_id="WB1" ko_md="1000.0" />
<CD_DEFINITIVE_SURVEY_HEADER def_survey_header_id="DH1" wellbore_id="WB1"
    phase="ACTUAL" name="def WB1" />
<CD_DEFINITIVE_SURVEY_STATION def_survey_header_id="DH1" md="0.0"
    inclination="0.0" azimuth="0.0" tvd="0.0" offset_north="0.0"
    offset_east="0.0" sequence_no="0" />
<CD_DEFINITIVE_SURVEY_STATION def_survey_header_id="DH1" md="500.0"
    inclination="5.0" azimuth="45.0" tvd="499.0" offset_north="15.0"
    offset_east="15.0" sequence_no="1" />
<CD_DEFINITIVE_SURVEY_STATION def_survey_header_id="DH1" md="1000.0"
    inclination="10.0" azimuth="45.0" tvd="990.0" offset_north="60.0"
    offset_east="60.0" sequence_no="2" />
</TOPLEVEL></export>
"""


@pytest.fixture
def reader(tmp_path):
    path = tmp_path / "edm.xml"
    path.write_text(FIXTURE_XML)
    return EDMReader(str(path))


# --------------------------------------------------------------------------- #
# structure-only mapping
# --------------------------------------------------------------------------- #
def test_entity_counts_and_types(reader):
    net = network_from_edm(reader)
    nodes = net._nodes
    assert isinstance(nodes["S1"], Site)
    assert isinstance(nodes["W1"], Well)
    assert isinstance(nodes["WB1"], Wellbore)
    assert isinstance(nodes["WB2"], Wellbore)
    wellbores = [n for n in nodes.values() if isinstance(n, Wellbore)]
    assert len(wellbores) == 2


def test_site_mapping(reader):
    net = network_from_edm(reader)
    site = net.node("S1")
    assert site.name == "Pad A"
    assert site.crs == "EPSG:23031"          # ED50 + UTM-31N only
    assert site.convergence == pytest.approx(math.radians(-1.0))
    assert site.is_field_centre is True
    assert site.location == pytest.approx(
        (10000.0 * FEET_TO_METERS, 20000.0 * FEET_TO_METERS))
    assert site.parent.name == "TESTFIELD"   # CD_PROJECT -> Field


def test_well_and_datum_metres(reader):
    net = network_from_edm(reader)
    well = net.node("W1")
    assert well.name == "T-1"
    assert well.parent is net.node("S1")
    assert well.slot == pytest.approx(
        (10.0 * FEET_TO_METERS, -20.0 * FEET_TO_METERS))
    assert well.slot_radial_error == pytest.approx(1.0 * FEET_TO_METERS)
    assert well.wellhead_depth == pytest.approx(300.0 * FEET_TO_METERS)
    assert isinstance(well.datum, Datum)
    assert well.datum.name == "Rotary Table"     # the default datum wins
    assert well.datum.elevation == pytest.approx(100.0 * FEET_TO_METERS)


def test_parent_linkage_and_kickoff(reader):
    net = network_from_edm(reader)
    assert [n.id for n in net.roots()] == ["WB1"]
    sidetrack = net.node("WB2")
    assert sidetrack.parent is net.node("WB1")
    assert sidetrack.kickoff_md == pytest.approx(1000.0 * FEET_TO_METERS)
    assert net.lowest_common_ancestor("WB2", "WB1") == "WB1"


# --------------------------------------------------------------------------- #
# optional survey attachment
# --------------------------------------------------------------------------- #
def test_structure_only_attaches_no_surveys(reader):
    net = network_from_edm(reader)
    assert net.node("WB1").survey is None
    assert net.node("WB2").survey is None


def test_surveys_true_attaches_and_warns_per_missing(reader):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        net = network_from_edm(reader, surveys=True)
    s = net.node("WB1").survey
    assert s is not None
    assert max(s.md) == pytest.approx(1000.0 * FEET_TO_METERS)  # metres
    assert s.header.azi_reference == "grid"
    # WB2 has no definitive survey: warned + skipped, import not aborted
    assert net.node("WB2").survey is None
    assert any("T-1 A" in str(x.message) for x in w)


# --------------------------------------------------------------------------- #
# OSDU round-trip sanity
# --------------------------------------------------------------------------- #
def test_osdu_export_does_not_crash(reader):
    net = network_from_edm(reader)
    records = [osdu.to_osdu(n) for n in net._nodes.values()]
    assert len(records) == 5                 # Field, Site, Well, 2 Wellbores
    # sidetrack record carries the parent-wellbore edge
    by_id = {r["id"]: r for r in records}
    assert by_id["WB2"]["data"]["KickOffWellbore"] == "WB1"
    # wellbore records re-import into a network without error
    net2 = osdu.network_from_osdu(records)
    assert net2.lowest_common_ancestor("WB2", "WB1") == "WB1"


# --------------------------------------------------------------------------- #
# public Volve export (skipped unless the ~211 MB file is present locally)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not os.path.isfile(VOLVE_XML), reason="Volve.xml not available"
)
def test_volve_structure_only_import():
    net = network_from_edm(EDMReader(VOLVE_XML))
    wellbores = [
        n for n in net._nodes.values() if isinstance(n, Wellbore)
    ]
    assert len(wellbores) >= 50              # the export carries 54
    # empirically-verified parentage: wellbore "F-15D" sidetracks off "F-15"
    (f15d,) = [n for n in wellbores if n.name == "F-15D"]
    assert isinstance(f15d.parent, Wellbore)
    assert f15d.parent.name == "F-15"
    assert f15d.kickoff_md == pytest.approx(4296.587926492 * FEET_TO_METERS)
    # the single Volve site maps with its project-derived CRS
    sites = [n for n in net._nodes.values() if isinstance(n, Site)]
    assert len(sites) == 1
    assert sites[0].name == "Volve F"
    assert sites[0].crs == "EPSG:23031"
