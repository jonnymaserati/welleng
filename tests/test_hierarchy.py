"""Tests for welleng.hierarchy (well hierarchy + wellbore network graph) and
welleng.osdu (versioned, units-aware OSDU import/export)."""
import json
import os
import tempfile
import warnings

import numpy as np
import pytest

from welleng.hierarchy import (
    Well, Wellbore, WellNetwork,
)
from welleng import osdu
from welleng.survey import Survey, SurveyHeader


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _sig(C):
    return float(np.sqrt(np.linalg.eigvalsh(np.asarray(C)).max()))


def _parent_and_sidetracks():
    """A parent well 0->2000 and two sidetracks kicking off at MD 1000."""
    md_p = list(range(0, 2001, 100))
    parent = Survey(
        md=md_p, inc=[min(0.03 * m, 40) for m in md_p], azi=[45.0] * len(md_p),
        header=SurveyHeader(name="P"), error_model="ISCWSA MWD Rev5.11",
    )
    i = md_p.index(1000.0)
    pos, cst = parent.pos_nev[i], parent.cov_nev[i]

    def _side(sign):
        md = list(range(1000, 2501, 100))
        return Survey(
            md=md, inc=[40.0] * len(md),
            azi=[45.0 + sign * 1.2 * (m - 1000) / 100 for m in md],
            header=SurveyHeader(name="S"), error_model="ISCWSA MWD Rev5.11",
            start_nev=pos, start_cov_nev=cst,
        )

    net = WellNetwork()
    p = net.add(Wellbore(id="P", parent=Well(id="W", name="F-11"), survey=parent))
    net.add(Wellbore(id="S1", parent=p, kickoff_md=1000.0, survey=_side(+1)))
    net.add(Wellbore(id="S2", parent=p, kickoff_md=1000.0, survey=_side(-1)))
    return net


# --------------------------------------------------------------------------- #
# graph topology
# --------------------------------------------------------------------------- #
def test_graph_roots_leaves_ancestors():
    net = _parent_and_sidetracks()
    assert [n.id for n in net.roots()] == ["P"]
    assert sorted(n.id for n in net.leaves()) == ["S1", "S2"]
    assert net.ancestors("S1") == ["S1", "P", "W"]


def test_lca_and_shared_divergent():
    net = _parent_and_sidetracks()
    assert net.lowest_common_ancestor("S1", "S2") == "P"
    shared, ba, bb = net.shared_and_divergent("S1", "S2")
    assert shared == ["P", "W"] and ba == ["S1"] and bb == ["S2"]
    # ancestor case: P is an ancestor of S1
    assert net.lowest_common_ancestor("P", "S1") == "P"
    _sh, ba2, bb2 = net.shared_and_divergent("P", "S1")
    assert ba2 == [] and bb2 == ["S1"]


def test_no_shared_ancestry_returns_none_lca():
    net = WellNetwork()
    net.add(Wellbore(id="A", parent=Well(id="WA"),
                     survey=Survey(md=[0, 500], inc=[0, 5], azi=[0, 0],
                                   header=SurveyHeader(name="A"))))
    net.add(Wellbore(id="B", parent=Well(id="WB"),
                     survey=Survey(md=[0, 500], inc=[0, 5], azi=[10, 10],
                                   header=SurveyHeader(name="B"))))
    assert net.lowest_common_ancestor("A", "B") is None


# --------------------------------------------------------------------------- #
# relative covariance (RP method / Williamson A-24)
# --------------------------------------------------------------------------- #
def test_relative_covariance_zero_at_kickoff():
    net = _parent_and_sidetracks()
    # parent vs sidetrack, and sidetrack vs sidetrack, both zero at the kick-off
    C_ps = net.relative_covariance("P", "S1", md_a=1000.0, md_b=1000.0)
    C_ss = net.relative_covariance("S1", "S2", md_a=1000.0, md_b=1000.0)
    assert _sig(C_ps) < 1e-6
    assert _sig(C_ss) < 1e-6


def test_relative_covariance_grows_below_kickoff_and_beats_naive():
    net = _parent_and_sidetracks()
    C_rel = net.relative_covariance("P", "S1")            # at TDs
    s_rel = _sig(C_rel)
    # naive independent sum (no cancellation)
    C_p = net.node("P").survey.cov_nev[-1]
    C_s = net.node("S1").survey.cov_nev[-1]
    s_naive = _sig(np.asarray(C_p) + np.asarray(C_s))
    assert s_rel > 0.0
    assert s_rel < s_naive               # shared trunk cancels -> smaller


def test_relative_covariance_ancestor_matches_manual():
    """P vs its child S1 at TDs = 15.14 m (the manual's worked value)."""
    net = _parent_and_sidetracks()
    assert _sig(net.relative_covariance("P", "S1")) == pytest.approx(15.14, abs=0.1)


def test_relative_covariance_share_mode_monotonic():
    """More shared error sources -> smaller relative uncertainty (RP table);
    still exactly zero at the shared kick-off under any share_mode."""
    net = _parent_and_sidetracks()
    indep = _sig(net.relative_covariance("S1", "S2", share_mode="all_independent"))
    glob = _sig(net.relative_covariance("S1", "S2", share_mode="globals_shared"))
    both = _sig(net.relative_covariance(
        "S1", "S2", share_mode="globals_and_systematic_shared"))
    assert indep > glob > both > 0.0
    z = _sig(net.relative_covariance(
        "S1", "S2", md_a=1000.0, md_b=1000.0, share_mode="globals_shared"))
    assert z < 1e-6


def test_relative_covariance_symmetric():
    net = _parent_and_sidetracks()
    ab = net.relative_covariance("S1", "S2")
    ba = net.relative_covariance("S2", "S1")
    assert np.allclose(np.asarray(ab), np.asarray(ba))


# --------------------------------------------------------------------------- #
# JSON save/load
# --------------------------------------------------------------------------- #
def test_json_round_trip_frame_exact():
    net = _parent_and_sidetracks()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "net.json")
        net.save_json(p)
        net2 = WellNetwork.load_json(p)
    # topology preserved
    assert [n.id for n in net2.roots()] == ["P"]
    assert net2.node("S1").kickoff_md == 1000.0
    # container chain survived (Site + Well are above the wellbores)
    assert "W" in net2._nodes
    # positions round-trip exactly (the geometry invariant)
    assert np.allclose(np.asarray(net.node("P").survey.pos_nev),
                       np.asarray(net2.node("P").survey.pos_nev))
    # azimuth REFERENCE preserved (serialised in its own frame — the default
    # header reference is 'true', not grid, so a naive grid dump would corrupt it)
    assert net2.node("P").survey.header.azi_reference == \
        net.node("P").survey.header.azi_reference
    # non-degenerate azimuths (inc>0 stations) frame-exact
    inc = np.asarray(net.node("P").survey.inc_rad)
    az0 = np.asarray(net.node("P").survey.azi_grid_rad)[inc > 1e-6]
    az1 = np.asarray(net2.node("P").survey.azi_grid_rad)[inc > 1e-6]
    assert np.allclose(az0, az1)


def test_json_serialisable():
    net = _parent_and_sidetracks()
    json.dumps(net.to_dict())            # must not raise (JSON-safe)


# --------------------------------------------------------------------------- #
# OSDU import/export — versioned + units-aware
# --------------------------------------------------------------------------- #
def test_osdu_kind_round_trip():
    k = osdu.build_kind("Wellbore")
    assert k == "osdu:wks:master-data--Wellbore:1.1.0"
    assert osdu.parse_kind(k) == ("master-data", "Wellbore", "1.1.0")


def test_osdu_wellbore_export_carries_parent_and_version():
    wb = Wellbore(id="W1", name="F-11 T2", parent=Well(id="W0", name="F-11"))
    rec = osdu.to_osdu(wb)
    assert rec["kind"].endswith(":1.1.0")
    assert rec["data"]["WellID"] == "W0"
    # a sidetrack points to its parent WELLBORE
    st = Wellbore(id="W2", name="lat", parent=wb)
    assert osdu.to_osdu(st)["data"]["KickOffWellbore"] == "W1"


def test_osdu_version_mismatch_warns():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        osdu.from_osdu({"kind": "osdu:wks:master-data--Wellbore:1.0.0",
                        "id": "X", "data": {"FacilityName": "old"}})
    assert any("schema version 1.0.0" in str(x.message) for x in w)


def test_osdu_units_feet_to_metres():
    well = osdu.from_osdu({
        "kind": "osdu:wks:master-data--Well:1.1.0", "id": "W",
        "data": {"FacilityName": "F", "LengthUnitOfMeasure": "ft",
                 "VerticalMeasurements": [{"VerticalMeasurement": 180.0}]},
    })
    assert well.datum.elevation == pytest.approx(180.0 * 0.3048, abs=1e-6)


def test_osdu_network_from_records_wires_parents():
    recs = [
        {"kind": "osdu:wks:master-data--Wellbore:1.1.0", "id": "P",
         "data": {"FacilityName": "P", "WellID": "W"}},
        {"kind": "osdu:wks:master-data--Wellbore:1.1.0", "id": "S",
         "data": {"FacilityName": "S", "KickOffWellbore": "P"}},
    ]
    net = osdu.network_from_osdu(recs)
    assert net.lowest_common_ancestor("S", "P") == "P"
