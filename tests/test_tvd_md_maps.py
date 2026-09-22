"""TVD <-> MD maps on a survey: frame, reversals, turning points, range.

One question -- "at which MD(s) does this survey reach this TVD, and what TVD
is at this MD?" -- answered by ``Survey`` in its own depth frame and by
``MinCurve`` in the local frame. The two must agree once the frame is shifted.
"""

import warnings

import numpy as np
import pytest

import welleng as we
from welleng.utils import MinCurve

DATUM_TVD = 1000.0


@pytest.fixture
def reversing():
    """Builds to 100 deg, so TVD rises again after the turning point."""
    md = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    inc = np.radians([0.0, 0.0, 30.0, 70.0, 100.0, 100.0])
    azi = np.zeros_like(md)
    mc = MinCurve(md, inc, azi)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = we.survey.Survey.from_min_curve(mc, start_nev=[0.0, 0.0, DATUM_TVD])
    return mc, s


def test_one_turning_point(reversing):
    _, s = reversing
    assert len(s.tvd_turning_points()) == 1


def test_a_tvd_short_of_the_turn_is_crossed_twice_and_each_round_trips(reversing):
    _, s = reversing
    tp = s.tvd_turning_points()[0]
    target = s.interpolate_md(tp).pos_nev[2] - 20.0
    nodes = s.interpolate_tvd(target)
    assert len(nodes) == 2
    assert nodes[0].md < tp < nodes[1].md
    for n in nodes:
        assert s.interpolate_md(n.md).pos_nev[2] == pytest.approx(target, abs=1e-6)


def test_survey_and_mincurve_agree_once_the_frame_is_shifted(reversing):
    mc, s = reversing
    tp = s.tvd_turning_points()[0]
    for target in (DATUM_TVD + 100.0, s.interpolate_md(tp).pos_nev[2] - 5.0):
        via_survey = [n.md for n in s.interpolate_tvd(target)]
        via_mincurve = list(mc.interpolate_tvd(target - DATUM_TVD))
        assert via_survey == pytest.approx(via_mincurve, abs=1e-9)


def test_mincurve_given_a_datum_frame_tvd_is_silently_wrong(reversing):
    """The documented trap: MinCurve has no datum, so it cannot refuse.

    Inside the local range the wrong frame returns a WRONG MD, not nothing;
    outside it, it returns nothing.
    """
    mc, s = reversing
    target = DATUM_TVD + 100.0
    right = [n.md for n in s.interpolate_tvd(target)]
    wrong = list(mc.interpolate_tvd(target))
    assert right == pytest.approx([100.0], abs=1e-6)
    assert len(wrong) == 1 and abs(wrong[0] - right[0]) > 1000.0
    beyond = DATUM_TVD + 1300.0
    assert len(s.interpolate_tvd(beyond)) == 2
    assert len(mc.interpolate_tvd(beyond)) == 0


def test_the_turning_tvd_itself_is_touched_once(reversing):
    _, s = reversing
    tp = s.tvd_turning_points()[0]
    nodes = s.interpolate_tvd(s.interpolate_md(tp).pos_nev[2])
    assert len(nodes) == 1
    assert nodes[0].md == pytest.approx(tp, abs=1e-6)


@pytest.fixture
def offset_start():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return we.survey.Survey(
            md=[100.0, 500.0, 1000.0], inc=[0.0, 10.0, 30.0], azi=[0.0, 0.0, 0.0]
        )


@pytest.mark.parametrize("md", [99.0, 50.0, 0.0])
def test_interpolate_md_above_the_first_station_is_none(offset_start, md):
    assert offset_start.interpolate_md(md) is None


@pytest.mark.parametrize("md", [1000.5, 2000.0])
def test_interpolate_md_beyond_the_last_station_is_none(offset_start, md):
    assert offset_start.interpolate_md(md) is None


@pytest.mark.parametrize("md", [100.0, 1000.0])
def test_interpolate_md_at_either_end_station_is_that_station(offset_start, md):
    node = offset_start.interpolate_md(md)
    assert node is not None
    assert node.md == pytest.approx(md)


# -- Survey.interpolate_md: the MinCurve.interpolate route vs the two-station route --

def _gate_surveys():
    import json

    from welleng.survey import make_survey_header

    yield Survey_(md=[0, 500, 1000, 1500, 2000, 2500], inc=[0, 0, 30, 70, 100, 100],
                  azi=[0] * 6, start_nev=[10.0, 20.0, 1000.0])
    yield Survey_(md=[0, 1000, 3000, 6000], inc=[0, 20, 60, 90], azi=[0, 30, 60, 90],
                  unit="feet", header=we.survey.SurveyHeader(depth_unit="feet"))
    yield Survey_(md=[0, 800, 1600, 2400], inc=[0, 25, 55, 80], azi=[10, 40, 70, 100],
                  header=we.survey.SurveyHeader(azi_reference="true", convergence=1.3))
    with open("tests/test_data/clearance_iscwsa_well_data.json") as f:
        data = json.load(f)
    for d in data["wells"].values():
        yield Survey_(md=d["MD"], inc=d["IncDeg"], azi=d["AziDeg"],
                      n=d["N"], e=d["E"], tvd=d["TVD"],
                      header=make_survey_header(d["header"]),
                      start_xyz=[d["E"][0], d["N"][0], d["TVD"][0]],
                      start_nev=[d["N"][0], d["E"][0], d["TVD"][0]])


def Survey_(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return we.survey.Survey(**kw)


def test_interpolate_md_matches_the_two_station_route():
    """Directions, md, flag and unit identical; positions within 3 ulp relative.

    3 ulp covers the measured maximum of 2.4 ulp over these surveys. The
    two routes differ in summation order (MinCurve.interpolate vs a rebuilt
    two-station Survey), so bit-identity is not the right bar.
    """
    from welleng.survey import _interpolate_survey, get_node

    rng = np.random.default_rng(3)
    checked = 0
    for s in _gate_surveys():
        for q in rng.uniform(s.md[0], s.md[-1], 200):
            i = min(max(int(np.searchsorted(s.md, q, side="left")) - 1, 0),
                    len(s.md) - 2)
            old = get_node(_interpolate_survey(s, q - s.md[i], i), -1, True)
            new = s.interpolate_md(q)
            assert new.vec_nev == old.vec_nev
            assert new.md == old.md
            assert new.interpolated == old.interpolated
            assert new.unit == old.unit
            scale = max(1.0, float(np.max(np.abs(old.pos_nev))))
            dpos = np.max(np.abs(np.array(new.pos_nev) - np.array(old.pos_nev)))
            assert dpos / scale <= 3 * np.finfo(float).eps
            checked += 1
    assert checked == 200 * 15  # 3 synthetic + 12 ISCWSA


def _call_interpolate_survey(s):
    from welleng.survey import _interpolate_survey
    _interpolate_survey(s, 50.0, 1)


@pytest.mark.parametrize("call", [
    lambda s: s.interpolate_md(700.0),
    _call_interpolate_survey,
    lambda s: s.interpolate_mds([250.0, 750.0]),
    lambda s: s.maximum_curvature(),
], ids=["interpolate_md", "_interpolate_survey", "interpolate_mds",
        "maximum_curvature"])
def test_building_a_grid_survey_does_not_touch_the_callers_header(call):
    """Each route builds a grid-referenced survey; the caller's keeps its own."""
    s = Survey_(md=[0, 500, 1000], inc=[0, 10, 30], azi=[0, 20, 40],
                header=we.survey.SurveyHeader(azi_reference="true"))
    call(s)
    assert s.header.azi_reference == "true"
