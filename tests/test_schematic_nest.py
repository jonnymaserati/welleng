"""Nest view: the depth map, the connector, and the rendered figure.

What a reader takes at face value has to be true in every depth mode:
  * every tick label is the TRUE depth at its height, and none sits inside a
    pause; nothing that carries a depth is drawn inside one;
  * a lane node's height is the map's height for its MD;
  * the pause takes the parent's packers off the connector bends (known
    positive: on a plain MD axis a hanger sits on the bend);
  * no label overlaps another, no text sits between the axis and the drawing,
    and no break line is drawn over the well.
"""
import math

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")

from matplotlib.text import Text  # noqa: E402

from welleng.schematic import (  # noqa: E402
    Casing,
    CementPlug,
    CompletionItem,
    HoleSection,
    SurveyRef,
    Well,
    Wellbore,
    WellSchematic,
)
from welleng.schematic.nest import (  # noqa: E402
    DEPTH_MODES,
    DepthMap,
    Void,
    lane_change,
    paused_extent,
    render_nest,
)

V1 = Void("pilot", 1838.0, 120.0, "S1")
V2 = Void("s1", 2208.0, 80.0, "S2")


# --------------------------------------------------------------------------
# DepthMap
# --------------------------------------------------------------------------
def test_plain_md_is_the_identity():
    m = DepthMap("md")
    assert [m.y("any", d) for d in (0.0, 1838.0, 3445.0)] == [0.0, 1838.0, 3445.0]
    assert not m.paused and m.spans() == []


def test_paused_shifts_by_every_void_above_and_only_those():
    m = DepthMap("md-paused", voids=[V2, V1])          # order given must not matter
    assert m.y("s1", 1000.0) == 1000.0
    assert m.y("s1", 1838.0, "lo") == 1838.0           # top of the S1 void
    assert m.y("s1", 1838.0, "hi") == 1838.0 + 120.0   # its bottom: the same MD
    assert m.y("s1", 2000.0) == 2000.0 + 120.0
    assert m.y("s2", 2208.0, "hi") == 2208.0 + 200.0
    assert m.y("s2", 3000.0) == 3000.0 + 200.0
    # voids are global: the pilot below the S1 branch is drawn below it too
    assert m.y("pilot", 2100.0) == 2100.0 + 120.0


def test_inverse_is_exact_outside_voids_and_none_inside():
    m = DepthMap("md-paused", voids=[V1, V2])
    for d in (0.0, 500.0, 1837.9, 1838.0, 1900.0, 2207.5, 2208.0, 2600.0, 3445.0):
        for side in ("lo", "hi"):
            assert m.domain_at(m.y_of_domain(d, side)) == pytest.approx(d)
    for y0, y1, _ in m.spans():
        for f in (0.1, 0.5, 0.9):
            assert m.domain_at(y0 + f * (y1 - y0)) is None


def test_ticks_are_true_depth_and_never_inside_a_void():
    m = DepthMap("md-paused", voids=[V1, V2])
    ticks = m.ticks(100.0, 0.0, 3445.0)
    assert [d for _, d in ticks] == [100.0 * k for k in range(35)]
    for y, d in ticks:
        assert m.domain_at(y) == pytest.approx(d)
        assert not any(y0 + 1e-9 < y < y1 - 1e-9 for y0, y1, _ in m.spans())


def test_a_curve_breaks_at_every_void_and_puts_no_point_inside_one():
    m = DepthMap("md-paused", voids=[V1, V2])
    md = [1800.0, 1838.0, 1850.0, 2200.0, 2250.0, 2300.0]
    ys, vs = m.trace("s2", md, [1, 2, 3, 4, 5, 6])
    spans = m.spans()
    for y in ys:
        if not math.isnan(y):
            assert not any(y0 + 1e-9 < y < y1 - 1e-9 for y0, y1, _ in spans)
    fin = [(a, b) for a, b in zip(ys[:-1], ys[1:])
           if not (math.isnan(a) or math.isnan(b))]
    for a, b in fin:
        assert not any(min(a, b) < y0 + 1e-9 and max(a, b) > y1 - 1e-9
                       for y0, y1, _ in spans)
    # the sample AT the branch point is drawn on both edges of its void
    i = ys.index(1838.0)
    assert math.isnan(ys[i + 1]) and ys[i + 2] == 1958.0 and vs[i] == vs[i + 2] == 2
    # known positive: 2200 -> 2250 straddles the S2 void
    j = ys.index(2200.0 + 120.0)
    assert math.isnan(ys[j + 1]) and ys[j + 2] == 2250.0 + 200.0


def test_tvd_mode_uses_the_callers_function_and_voids_key_on_tvd():
    tvd = {"pilot": lambda md: 0.8 * md, "s1": lambda md: 0.5 * md}
    m = DepthMap("tvd", tvd=lambda b, md: tvd[b](md),
                 voids=[Void("pilot", 1000.0, 50.0)])
    assert m.domain("s1", 1200.0) == 600.0
    assert m.y("s1", 1200.0) == 600.0             # 600 TVD is above the void at 800
    assert m.y("pilot", 1200.0) == 960.0 + 50.0
    assert m.spans()[0][:2] == (800.0, 850.0)


@pytest.mark.parametrize("kw, msg", [
    (dict(mode="depth"), "not one of"),
    (dict(mode="tvd"), "tvd\\(bore, md\\)"),
    (dict(mode="md", voids=[V1]), "consumes MD"),
    (dict(mode="md-paused", voids=[Void("a", 10.0, 0.0)]), "positive height"),
    (dict(mode="md-paused", voids=[Void("a", 10.0, 5.0), Void("b", 10.0, 5.0)]),
     "same depth"),
])
def test_depth_map_refusals_name_what_is_wrong(kw, msg):
    mode = kw.pop("mode")
    with pytest.raises(ValueError, match=msg):
        DepthMap(mode, **kw)


def test_paused_extent_sympy_derivation():
    import sympy as sp
    E, E0, P, R = sp.symbols("E E0 P R", positive=True)
    (sol,) = sp.solve(sp.Eq(E, E0 + 2 * R * E / P), E)
    assert sp.simplify(sol - E0 * P / (P - 2 * R)) == 0
    closed = E0 * P / (P - 2 * R)
    # residual of the defining equation
    assert sp.simplify(closed - (E0 + 2 * R * closed / P)) == 0


def test_paused_extent_matches_the_fixed_point_iteration():
    e0, page, radii = 3578.0, 3421.6, [0.61, 0.43]
    e = e0
    for _ in range(200):
        e = e0 + 2.0 * sum(radii) * e / page
    assert paused_extent(e0, page, radii) == pytest.approx(e, rel=1e-12)
    with pytest.raises(ValueError, match="fill"):
        paused_extent(e0, 1.0, [0.5])


# --------------------------------------------------------------------------
# lane_change
# --------------------------------------------------------------------------
def test_lane_change_depth_does_not_depend_on_the_offset():
    for off in (2.0, 3.5, 7.0):
        lc = lane_change(1.0, 100.0, off, radius=1.0, depth_per_x=50.0)
        assert lc.points[0] == pytest.approx((1.0, 100.0))
        assert lc.points[-1] == pytest.approx((1.0 + off, 200.0))  # 2 R k
        assert lc.depth == pytest.approx(100.0)
        assert lc.straight == pytest.approx(off - 2.0)


def test_lane_change_turns_are_circles_in_the_paper_frame():
    R, k = 0.8, 30.0
    lc = lane_change(0.0, 0.0, 4.0, radius=R, depth_per_x=k)
    for x, z in lc.points[:45]:                      # turn 1, centre (R, 0)
        assert math.hypot(x - R, z / k) == pytest.approx(R)


def test_lane_change_walls_run_parallel_at_half_the_width():
    lc = lane_change(0.0, 0.0, 3.0, radius=0.6, depth_per_x=1.0)
    a, b = lc.walls(0.3)
    for (xa, za), (xb, zb) in zip(a, b):
        assert math.hypot(xa - xb, za - zb) == pytest.approx(0.3)


def test_lane_change_refuses_an_offset_below_two_radii():
    with pytest.raises(ValueError, match="two turn radii"):
        lane_change(0.0, 0.0, 1.9, radius=1.0)


# --------------------------------------------------------------------------
# the rendered view, on a synthetic three-bore well
# --------------------------------------------------------------------------
def _family():
    # pilot vertical; ST1 builds east from 1800 m at 3 deg/30 m; ST2 follows
    # ST1 to 2200 m (inc 40) and turns south-east below it
    pilot = Wellbore(
        id="X-1",
        survey=SurveyRef(md=[0.0, 3400.0], inc=[0.0, 0.0], azi=[0.0, 0.0]),
        hole_sections=[HoleSection(bit_in=26.0, top_md=90.0, base_md=500.0),
                       HoleSection(bit_in=17.5, top_md=500.0, base_md=1500.0),
                       HoleSection(bit_in=12.25, top_md=1500.0, base_md=3000.0)],
        casings=[Casing(name="20in", od_in=20.0, nominal_weight_ppf=133.0,
                        top_md=0.0, shoe_md=495.0, toc_md=0.0),
                 Casing(name="13-3/8in", od_in=13.375, nominal_weight_ppf=68.0,
                        top_md=0.0, shoe_md=1495.0, toc_md=900.0)],
        cement_plugs=[CementPlug(name="plug", top_md=1750.0, base_md=2100.0)])
    s1 = Wellbore(
        id="X-1-ST1", parent_id="X-1", kickoff_md=1800.0,
        survey=SurveyRef(md=[0.0, 1800.0, 2400.0, 3400.0],
                         inc=[0.0, 0.0, 60.0, 60.0],
                         azi=[90.0, 90.0, 90.0, 90.0]),
        hole_sections=[HoleSection(bit_in=12.25, top_md=1800.0, base_md=2700.0),
                       HoleSection(bit_in=8.5, top_md=2700.0, base_md=3300.0)],
        casings=[Casing(name="9-5/8in", od_in=9.625, nominal_weight_ppf=47.0,
                        top_md=0.0, shoe_md=2695.0, toc_md=1400.0),
                 Casing(name="7in ST1 liner", kind="liner", od_in=7.0,
                        nominal_weight_ppf=29.0, top_md=2218.0, shoe_md=3250.0)],
        completion=[CompletionItem(type="packer", name="liner hanger",
                                   od_in=8.681, md=2215.0)])
    s2 = Wellbore(
        id="X-1-ST2", parent_id="X-1-ST1", kickoff_md=2210.0, window_md=2200.0,
        survey=SurveyRef(md=[0.0, 1800.0, 2200.0, 2800.0, 3400.0],
                         inc=[0.0, 0.0, 40.0, 80.0, 80.0],
                         azi=[90.0, 90.0, 90.0, 120.0, 120.0]),
        hole_sections=[HoleSection(bit_in=8.5, top_md=2200.0, base_md=3200.0)],
        casings=[Casing(name="7in", kind="liner", od_in=7.0,
                        nominal_weight_ppf=29.0, top_md=2216.0, shoe_md=3150.0,
                        toc_md=2400.0)],
        completion=[CompletionItem(type="packer", name="production packer",
                                   md=2600.0, inner_string="4-1/2in tubing"),
                    CompletionItem(type="tubing", name="4-1/2in tubing",
                                   od_in=4.5, top_md=0.0, base_md=2610.0)])
    return WellSchematic(
        well=Well(name="X-1", depth_reference="RKB", datum_elevation_m=30.0),
        wellbores=[pilot, s1, s2])


@pytest.fixture(scope="module", params=DEPTH_MODES)
def nest(request):
    import matplotlib.pyplot as plt
    view = render_nest(_family(), mode=request.param)
    yield view
    plt.close(view.fig)


def test_nothing_with_a_depth_is_drawn_inside_a_void(nest):
    dmap = nest.depth_map
    for n, P in nest.paths.items():
        for _x, y, _tx, _ty, md in P:
            for (y0, y1, _v), dv in zip(dmap.spans(), dmap.depths):
                if y0 + 1e-6 < y < y1 - 1e-6:
                    assert dmap.domain(n, md) == pytest.approx(dv, abs=1e-6)


def test_a_lane_node_sits_at_the_maps_height_for_its_md(nest):
    dmap = nest.depth_map
    conn_nodes = {id(q) for c in nest.connectors.values() for q in c}
    checked = 0
    for n, P in nest.paths.items():
        for q in P:
            if id(q) in conn_nodes or dmap.domain_at(q[1]) is None:
                continue
            assert dmap.domain_at(q[1]) == pytest.approx(dmap.domain(n, q[4]),
                                                         abs=1e-6)
            checked += 1
    assert checked >= 10


def test_the_window_is_the_branch_point(nest):
    assert nest.branch_md == {"X-1-ST1": 1800.0, "X-1-ST2": 2200.0}
    assert any("milled window 2200" in c for c in nest.caveats)


def test_the_pause_takes_the_packers_off_the_bends(nest):
    dmap, kop, span, par = (nest.depth_map, nest.branch_md, nest.connector_md,
                            nest.parent)
    assert len(nest.packer_boxes) == 2
    if nest.mode == "md":
        # KNOWN POSITIVE: on plain MD the ST2 connector spans MD of ST1, and
        # ST1's hanger (2215 m) falls on it -- what the pause removes
        c = "X-1-ST2"
        y0, y1 = dmap.y(par[c], kop[c]), dmap.y(par[c], kop[c] + span[c])
        assert any(b0 < y1 and b1 > y0 for _n, _md, b0, b1 in nest.packer_boxes)
    elif nest.mode == "md-paused":
        for _n, _md, b0, b1 in nest.packer_boxes:
            for y0, y1, _v in dmap.spans():
                assert not (b0 < y1 - 1e-6 and b1 > y0 + 1e-6)


def test_a_packer_with_no_od_is_drawn_to_its_host_and_says_so(nest):
    assert any("production packer at 2600 m: OD not recorded -- drawn to its "
               "host casing ID 6.184in [DERIVED]" in c for c in nest.caveats)
    assert not any("inner string" in c and "not in the model" in c
                   for c in nest.caveats)


def test_no_two_labels_overlap(nest):
    fig, ax = nest.fig, nest.ax
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = (list(ax.texts) + [ax.yaxis.label, ax.title]
             + list(ax.get_yticklabels()))
    boxes = [(t.get_text().replace("\n", " ")[:50],
              Text.get_window_extent(t, renderer=r))
             for t in texts if t.get_visible() and t.get_text().strip()]
    bad = []
    for i, (ta, a) in enumerate(boxes):
        for tb, b in boxes[i + 1:]:
            if min(a.x1, b.x1) - max(a.x0, b.x0) > 2.0 and \
                    min(a.y1, b.y1) - max(a.y0, b.y0) > 2.0:
                bad.append((ta, tb))
    assert not bad, bad


def test_no_break_line_is_drawn_over_the_well(nest):
    if not nest.depth_map.spans():
        assert nest.mode == "md"
        return
    assert nest.break_lines, "voids but no break lines -- would pass vacuously"
    for ln in nest.break_lines:
        x, y = ln.get_data()
        y0 = float(y[0])
        for a, b in nest.occupied(y0, tol=0.0675 * nest.dpx, pad=0.0):
            assert b < min(x) or a > max(x), (y0, min(x), max(x), a, b)


def test_no_text_in_the_gap_between_the_axis_and_the_drawing(nest):
    fig, ax = nest.fig, nest.ax
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    xs = [ax.transData.transform((x, 0.0))[0] for ln in nest.well_artists
          if hasattr(ln, "get_xdata") for x in ln.get_xdata()]
    left = min(xs)
    inside = [t.get_text()[:40] for t in ax.texts
              if t.get_visible() and t.get_text().strip()
              and t.get_window_extent(renderer=r).x0 < left - 2]
    assert not inside


def test_minor_ticks_are_true_depth_and_never_in_a_pause(nest):
    dmap, ax = nest.depth_map, nest.ax
    step = 250.0 if nest.mode == "tvd" else 500.0
    minor = list(ax.get_yticks(minor=True))
    assert len(minor) > 20
    for y in minor:
        d = dmap.domain_at(y)
        assert d is not None
        assert abs(d / (step / 10.0) - round(d / (step / 10.0))) < 1e-6
        assert not any(y0 + 1e-6 < y < y1 - 1e-6 for y0, y1, _ in dmap.spans())


def test_tvd_mode_draws_tvdss_from_minimum_curvature():
    import matplotlib.pyplot as plt
    from welleng.schematic import DepthResolver
    sch = _family()
    view = render_nest(sch, mode="tvd")
    try:
        # below its branch point ST1's depth is its own survey's TVDSS
        r = DepthResolver(sch.wellbores[1].survey)
        assert view.depth_map.domain("X-1-ST1", 2500.0) == pytest.approx(
            r.tvd_at(2500.0) - 30.0, abs=1e-9)
        # above it, the parent's
        rp = DepthResolver(sch.wellbores[0].survey)
        assert view.depth_map.domain("X-1-ST1", 1000.0) == pytest.approx(
            rp.tvd_at(1000.0) - 30.0, abs=1e-9)
    finally:
        plt.close(view.fig)


def test_tvd_mode_refuses_without_a_datum():
    sch = _family()
    sch.well.datum_elevation_m = None
    with pytest.raises(ValueError, match="datum"):
        render_nest(sch, mode="tvd")


def test_a_child_listed_before_its_parent_is_refused():
    sch = _family()
    sch.wellbores = [sch.wellbores[1], sch.wellbores[0], sch.wellbores[2]]
    with pytest.raises(ValueError, match="listed before"):
        render_nest(sch, mode="md")


# --------------------------------------------------------------------------
# model and resolver support for the view
# --------------------------------------------------------------------------
def test_tubing_still_needs_an_od():
    with pytest.raises(ValueError, match="tubing needs od_in"):
        CompletionItem(type="tubing", top_md=0.0, base_md=100.0)


def test_a_window_needs_a_parent_and_is_the_branch_point():
    sv = SurveyRef(md=[0.0, 1000.0], inc=[0.0, 0.0], azi=[0.0, 0.0])
    with pytest.raises(ValueError, match="window_md is set but parent_id"):
        Wellbore(id="a", survey=sv, window_md=500.0)
    b = Wellbore(id="b", parent_id="a", kickoff_md=515.6, survey=sv)
    assert b.branch_md == 515.6
    b = Wellbore(id="b", parent_id="a", kickoff_md=515.6, window_md=500.0,
                 survey=sv)
    assert b.branch_md == 500.0


def _column_with_packer(od_in, casing=True):
    from welleng.schematic import build_column
    sv = SurveyRef(md=[0.0, 2000.0], inc=[0.0, 0.0], azi=[0.0, 0.0])
    b = Wellbore(
        id="w", survey=sv,
        hole_sections=[HoleSection(bit_in=12.25, top_md=0.0, base_md=2000.0)],
        casings=([Casing(name="9-5/8in", od_in=9.625, nominal_weight_ppf=47.0,
                         top_md=0.0, shoe_md=1990.0)] if casing else []),
        completion=[CompletionItem(type="tubing", name="tbg", od_in=4.5,
                                   top_md=0.0, base_md=1500.0),
                    CompletionItem(type="packer", name="pkr", od_in=od_in,
                                   md=1400.0)])
    return build_column(WellSchematic(well=Well(name="w"), wellbores=[b]))


def _packer_xs(dwg):
    from welleng.schematic import Polygon
    from welleng.schematic.column import L_COMPLETION
    xs = sorted({round(x, 9) for e in dwg.entities
                 if isinstance(e, Polygon) and e.layer == L_COMPLETION
                 for x, _y in e.points})
    assert xs, "no packer drawn -- the comparison would pass vacuously"
    return xs


def test_column_draws_a_packer_with_no_od_to_its_host_casing():
    # the host's ID bounds the seal either way, so an unrecorded OD draws the
    # same packer as one recorded at (or beyond) the host ID
    host_id = Casing(name="c", od_in=9.625, nominal_weight_ppf=47.0,
                     top_md=0.0, shoe_md=1.0).id_in
    assert _packer_xs(_column_with_packer(None)) == \
        _packer_xs(_column_with_packer(host_id))


def test_column_refuses_a_packer_with_no_od_and_no_host():
    with pytest.raises(ValueError, match="no OD recorded and no casing"):
        _column_with_packer(None, casing=False)


def test_resolver_attitude_is_the_min_curve_attitude():
    import numpy as np
    from welleng.schematic import DepthResolver
    from welleng.survey import Survey
    sv = SurveyRef(md=[0.0, 1000.0, 2000.0, 3000.0], inc=[0.0, 5.0, 60.0, 95.0],
                   azi=[0.0, 350.0, 20.0, 45.0])
    r = DepthResolver(sv)
    q = np.array([500.0, 1500.0, 2750.0])
    inc, azi = r.inc_azi_at(q)
    ref = Survey(md=sv.md, inc=sv.inc, azi=sv.azi)
    i_ref, a_ref = ref.inc_azi_at(q)
    assert np.array_equal(inc, np.degrees(i_ref))
    assert np.array_equal(azi, np.degrees(a_ref))
    assert 0.0 <= azi.min() and azi.max() < 360.0


def test_resolver_returns_every_tvd_crossing():
    from welleng.schematic import DepthResolver
    sv = SurveyRef(md=[0.0, 1000.0, 2000.0, 3000.0], inc=[0.0, 0.0, 60.0, 95.0],
                   azi=[0.0, 0.0, 90.0, 90.0])
    r = DepthResolver(sv)
    # known positive: past 90 deg the lateral climbs back through this TVD
    target = r.tvd_at(2800.0)
    mds = r.md_at_tvd(target)
    assert len(mds) == 2 and mds[0] == pytest.approx(2800.0, abs=1e-6)
    for m in mds:
        assert r.tvd_at(m) == pytest.approx(target, abs=1e-6)
    assert r.md_at_tvd(1e6) == []


def test_resolver_refuses_a_tvd_column_that_disagrees_with_its_angles():
    from welleng.schematic import DepthResolver
    md, inc, azi = [0.0, 1000.0, 2000.0], [0.0, 30.0, 60.0], [0.0, 10.0, 20.0]
    col = list(DepthResolver(SurveyRef(md=md, inc=inc, azi=azi))._stations.tvd)
    ok = DepthResolver(SurveyRef(md=md, inc=inc, azi=azi, tvd=col))
    assert ok.station_tvd_gap_m == pytest.approx(0.0, abs=1e-9)
    # compared relative to the first station: a datum offset is not a gap
    shifted = DepthResolver(SurveyRef(md=md, inc=inc, azi=azi,
                                      tvd=[t + 25.0 for t in col]))
    assert shifted.station_tvd_gap_m == pytest.approx(0.0, abs=1e-9)
    bad = col[:]
    bad[2] += 1.5
    with pytest.raises(ValueError, match="disagrees with minimum curvature"):
        DepthResolver(SurveyRef(md=md, inc=inc, azi=azi, tvd=bad))
    with pytest.raises(ValueError, match="same length"):
        SurveyRef(md=md, inc=inc, azi=azi, tvd=col[:2])


# --------------------------------------------------------------------------
# no path turns back across a pause
# --------------------------------------------------------------------------
def _backtracks(view):
    """Consecutive path nodes going from a pause's bottom edge to its top:
    a wall drawn back up through the pause."""
    hits = []
    for n, P in view.paths.items():
        for a, b in zip(P[:-1], P[1:]):
            for y0, y1, _v in view.depth_map.spans():
                if abs(a[1] - y1) < 1e-9 and abs(b[1] - y0) < 1e-9:
                    hits.append((n, a[4], b[4]))
    return hits


def _building_parent():
    """ST1 leaves the pilot at 1838 m while the pilot is building: the TVD
    inverse there returns the branch MD to within an ulp, not exactly."""
    sch = _family()
    pilot, s1, _s2 = sch.wellbores
    pilot.survey = SurveyRef(md=[0.0, 1500.0, 3400.0], inc=[0.0, 0.0, 30.0],
                             azi=[0.0, 0.0, 0.0])
    s1.kickoff_md = 1838.0
    s1.survey = SurveyRef(
        md=[0.0, 1500.0, 1838.0, 2400.0, 3400.0],
        inc=[0.0, 0.0, 30.0 * 338.0 / 1900.0, 60.0, 60.0],
        azi=[0.0, 0.0, 0.0, 90.0, 90.0])
    return sch


def _child_survey_off_parent(inc_at_branch):
    """ST2's own survey puts the shared branch point off ST1's (inc 40.1 at
    2200 m: 15.5 cm shallower; 39.9: 15.5 cm deeper)."""
    sch = _family()
    sch.wellbores[2].survey = SurveyRef(
        md=[0.0, 1800.0, 2200.0, 2800.0, 3400.0],
        inc=[0.0, 0.0, inc_at_branch, 80.0, 80.0],
        azi=[0.0, 0.0, 90.0, 120.0, 120.0])
    return sch


@pytest.mark.parametrize("build", [
    _family, _building_parent,
    lambda: _child_survey_off_parent(40.1),
    lambda: _child_survey_off_parent(39.9),
], ids=["base", "building-parent", "child-shallower", "child-deeper"])
@pytest.mark.parametrize("mode", DEPTH_MODES)
def test_a_path_never_turns_back_across_a_pause(build, mode):
    import matplotlib.pyplot as plt
    view = render_nest(build(), mode=mode)
    try:
        assert _backtracks(view) == []
    finally:
        plt.close(view.fig)


def test_a_child_survey_off_its_parent_is_tied_and_said():
    import matplotlib.pyplot as plt
    view = render_nest(_child_survey_off_parent(40.1), mode="tvd")
    try:
        said = "ST2: its survey puts the branch point 15.5 cm shallower than ST1"
        assert any(said in c for c in view.caveats), view.caveats
        # tied: no step in ST2's depth at its branch point
        d = view.depth_map
        assert d.domain("X-1-ST2", 2200.0 + 1e-6) == pytest.approx(
            d.domain("X-1-ST1", 2200.0), abs=1e-4)
    finally:
        plt.close(view.fig)


def test_a_packer_not_drawn_is_not_also_said_to_be_drawn():
    import matplotlib.pyplot as plt
    sch = _family()
    s2 = sch.wellbores[2]
    s2.completion = [c for c in s2.completion if c.type == "packer"]
    s2.completion[0].inner_string = None          # nothing to seal on
    view = render_nest(sch, mode="md")
    try:
        mine = [c for c in view.caveats if "production packer" in c]
        assert len(mine) == 1 and "not drawn" in mine[0], mine
    finally:
        plt.close(view.fig)
