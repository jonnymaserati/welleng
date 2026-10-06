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


def test_a_survey_does_not_write_into_the_header_it_was_given():
    """The constructor writes its datum into its header; that must be its own."""
    base = Survey_(md=[0, 500, 1000], inc=[0, 10, 30], azi=[0, 20, 40],
                   start_nev=[100.0, 200.0, 300.0])
    Survey_(md=[0, 100], inc=[0, 5], azi=[0, 0], header=base.header,
            start_nev=[9.0, 9.0, 9.0])
    assert list(base.header.start_nev) == [100.0, 200.0, 300.0]


def test_slice_survey_leaves_the_original_header_alone():
    from welleng.survey import slice_survey
    s = Survey_(md=[0, 500, 1000, 1500], inc=[0, 10, 30, 45], azi=[0, 20, 40, 60],
                start_nev=[100.0, 200.0, 300.0])
    sl = slice_survey(s, 1)
    assert list(s.header.start_nev) == [100.0, 200.0, 300.0]
    assert sl.header is not s.header
    assert list(sl.header.start_nev) != [100.0, 200.0, 300.0]


def test_grid_header_copies():
    from welleng.survey import grid_header
    h = we.survey.SurveyHeader(azi_reference="true")
    g = grid_header(h)
    assert (h.azi_reference, g.azi_reference) == ("true", "grid")
    assert g is not h and g.mag_defaults is not h.mag_defaults


# -- interpolate_mds vs interpolate_md: one arc kernel, two entry points --

def _computed_position_surveys():
    """Surveys whose station positions are welleng's own (none supplied)."""
    yield Survey_(md=[0, 500, 1000, 1500, 2000, 2500], inc=[0, 0, 30, 70, 100, 100],
                  azi=[0] * 6, start_nev=[10.0, 20.0, 1000.0])
    yield Survey_(md=[0, 100, 200, 300], inc=[0, 5, 178, 178], azi=[0, 0, 180, 180])
    yield Survey_(md=[0, 1000, 3000, 6000], inc=[0, 20, 60, 90], azi=[0, 30, 60, 90],
                  unit="feet", header=we.survey.SurveyHeader(depth_unit="feet"))
    yield Survey_(md=[0, 800, 1600, 2400], inc=[0, 25, 55, 80], azi=[10, 40, 70, 100],
                  header=we.survey.SurveyHeader(azi_reference="true", convergence=1.3))
    yield Survey_(md=[0, 300, 600, 900], inc=[0] * 4, azi=[0] * 4)


def test_interpolate_mds_agrees_with_interpolate_md():
    """Same arc kernel: directions to 1 ulp; positions to 1e-11 m (measured 1.7e-12)."""
    rng = np.random.default_rng(9)
    checked = 0
    for s in _computed_position_surveys():
        q = np.setdiff1d(np.sort(rng.uniform(s.md[0], s.md[-1], 100)), s.md)
        r = s.interpolate_mds(q)
        mask = np.array(r.interpolated, dtype=bool)
        for j in np.where(mask)[0]:
            node = s.interpolate_md(r.md[j])
            np.testing.assert_allclose(node.vec_nev, r.vec_nev[j], rtol=0,
                                       atol=2 * np.finfo(float).eps)
            np.testing.assert_allclose(node.pos_nev, r.pos_nev[j], rtol=0, atol=1e-11)
            checked += 1
    assert checked > 400




# -- one anchor, one transform: every position field is a view of pos_nev --

WELLHEAD_NEV = [-50.0, -500.0, 0.0]
WELLHEAD_XYZ = [-500.0, -50.0, 0.0]          # the same point, (E, N, TVD)


def _plain(**kw):
    return Survey_(md=[0, 500, 1000, 1500], inc=[0, 10, 30, 45],
                   azi=[0, 20, 40, 60], **kw)


def _views_agree(s):
    np.testing.assert_array_equal(np.column_stack([s.n, s.e, s.tvd]), s.pos_nev)
    np.testing.assert_array_equal(np.column_stack([s.x, s.y, s.z]), s.pos_xyz)
    np.testing.assert_array_equal(s.pos_xyz, s.pos_nev[:, [1, 0, 2]])


@pytest.mark.parametrize("kw", [
    dict(start_nev=WELLHEAD_NEV),
    dict(start_xyz=WELLHEAD_XYZ),
    dict(start_nev=WELLHEAD_NEV, start_xyz=WELLHEAD_XYZ),
], ids=["start_nev", "start_xyz", "both, consistent"])
def test_one_anchor_whichever_way_it_is_given(kw):
    s = _plain(**kw)
    np.testing.assert_array_equal(s.pos_nev[0], WELLHEAD_NEV)
    _views_agree(s)


def test_contradictory_anchors_are_refused():
    with pytest.raises(ValueError, match="SAME point"):
        _plain(start_nev=WELLHEAD_NEV, start_xyz=[1.0, 2.0, 3.0])


def test_supplied_positions_are_the_positions_and_the_residual_is_reported():
    ref = _plain(start_nev=WELLHEAD_NEV)
    rounded = np.round(ref.pos_nev, 2)
    s = _plain(n=rounded[:, 0], e=rounded[:, 1], tvd=rounded[:, 2])
    np.testing.assert_array_equal(s.pos_nev, rounded)
    _views_agree(s)
    assert 0 < s.supplied_residual <= 0.005 + 1e-9


def test_a_computed_survey_reports_no_residual():
    assert _plain(start_nev=WELLHEAD_NEV).supplied_residual is None


def test_interpolate_mds_carries_supplied_positions():
    """Same rule as interpolate_md: stations keep theirs; between, arc
    displacement from the bracketing station."""
    ref = _plain(start_nev=WELLHEAD_NEV)
    rounded = np.round(ref.pos_nev, 2)
    s = _plain(n=rounded[:, 0], e=rounded[:, 1], tvd=rounded[:, 2])
    q = np.array([250.0, 750.0, 1234.5])
    r = s.interpolate_mds(q)
    on_station = ~np.array(r.interpolated, dtype=bool)
    np.testing.assert_array_equal(r.pos_nev[on_station], rounded)
    for j in np.where(~on_station)[0]:
        np.testing.assert_allclose(r.pos_nev[j], s.interpolate_md(r.md[j]).pos_nev,
                                   rtol=0, atol=1e-9)


def test_grid_scale_factor_applies_to_every_view():
    s = _plain(start_nev=WELLHEAD_NEV,
               header=we.survey.SurveyHeader(grid_scale_factor=1.001))
    _views_agree(s)
    local_n = s.poss[-1, 1]
    assert s.n[-1] - WELLHEAD_NEV[0] == pytest.approx(1.001 * local_n, rel=1e-12)


def test_supplied_xyz_is_the_same_as_supplied_nev():
    ref = _plain(start_nev=WELLHEAD_NEV)
    x, y, z = ref.pos_xyz.T
    s = _plain(x=x, y=y, z=z)
    np.testing.assert_array_equal(s.pos_nev, ref.pos_nev)
    with pytest.raises(ValueError, match="two frames"):
        _plain(x=x, y=y, z=z, n=ref.n + 1.0, e=ref.e, tvd=ref.tvd)
