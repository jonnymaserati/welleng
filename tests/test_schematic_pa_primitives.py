"""P&A after-state primitives on the schematic model: mechanical plugs,
milled lengths of a string, and annular cement beyond the primary job."""
import copy

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from welleng.schematic import (  # noqa: E402
    Casing, CementInterval, MechanicalPlug, MilledInterval, WellSchematic,
    build_column, build_section,
)
from welleng.schematic.plumbing import render_plumbing  # noqa: E402

BASE = {
    "well": {"name": "PA-1"},
    "survey": {"md": [0, 400, 1000, 2600], "inc": [0, 0, 30, 30],
               "azi": [45, 45, 45, 45]},
    "hole_sections": [
        {"bit_in": 26, "top_md": 0, "base_md": 400},
        {"bit_in": 17.5, "top_md": 400, "base_md": 1200},
        {"bit_in": 12.25, "top_md": 1200, "base_md": 2600},
    ],
    "casings": [
        {"name": "20in", "od_in": 20, "id_in": 18.7, "top_md": 0,
         "shoe_md": 400, "toc_md": 0},
        {"name": "13-3/8", "od_in": 13.375, "id_in": 12.4, "top_md": 0,
         "shoe_md": 1200, "toc_md": 800},
        {"name": "9-5/8", "od_in": 9.625, "id_in": 8.68, "top_md": 0,
         "shoe_md": 2600, "toc_md": 1400},
    ],
}


def _casing(**kw):
    base = dict(name="9-5/8", od_in=9.625, id_in=8.68, top_md=0.0,
                shoe_md=2600.0, toc_md=1400.0)
    base.update(kw)
    return Casing(**base)


# --- the model ------------------------------------------------------------
def test_cement_intervals_merge_primary_and_annular():
    c = _casing(annular_cement=[
        {"top_md": 1100, "base_md": 1250, "origin": "perf-wash-cement"},
        {"top_md": 1350, "base_md": 1450, "origin": "squeeze"},   # overlaps primary
    ])
    assert c.cement_intervals() == [(1100, 1250), (1350, 2600)]


def test_cement_intervals_empty_when_nothing_recorded():
    assert _casing(toc_md=None).cement_intervals() == []
    only = _casing(toc_md=None, annular_cement=[{"top_md": 500, "base_md": 600}])
    assert only.cement_intervals() == [(500, 600)]


@pytest.mark.parametrize("interval", [(-10.0, 100.0), (2500.0, 2700.0)])
def test_annular_cement_must_be_behind_the_string(interval):
    top, base = interval
    with pytest.raises(ValueError, match="not behind the string"):
        _casing(annular_cement=[{"top_md": top, "base_md": base}])


def test_interval_order_refused():
    with pytest.raises(ValueError):
        CementInterval(top_md=100, base_md=100)
    with pytest.raises(ValueError):
        MilledInterval(top_md=200, base_md=150)


def test_milled_removes_steel_and_keeps_one_string():
    c = _casing(milled=[{"top_md": 1500, "base_md": 1550}])
    assert c.steel_intervals() == [(0.0, 1500), (1550, 2600.0)]
    assert c.milled_at(1525) and not c.milled_at(1400)
    assert c.shoe_remains
    # a combination string milled across its crossover keeps each diameter
    combo = Casing(name="combo", od_in=9.625, id_in=8.68, top_md=0.0,
                   shoe_md=2000.0, sections=[
                       {"od_in": 10.75, "id_in": 9.95, "top_md": 0.0,
                        "base_md": 800.0},
                       {"od_in": 9.625, "id_in": 8.68, "top_md": 800.0,
                        "base_md": 2000.0}],
                   milled=[{"top_md": 750, "base_md": 900}])
    assert combo.steel_profile() == [(0.0, 750, 10.75, 9.95),
                                     (900, 2000.0, 9.625, 8.68)]


def test_milled_shoe_is_gone():
    c = _casing(milled=[{"top_md": 2550, "base_md": 2600}])
    assert not c.shoe_remains


def test_milled_intervals_refused_outside_or_overlapping():
    with pytest.raises(ValueError, match="outside the string"):
        _casing(milled=[{"top_md": 2590, "base_md": 2700}])
    with pytest.raises(ValueError, match="overlap"):
        _casing(milled=[{"top_md": 1500, "base_md": 1550},
                        {"top_md": 1540, "base_md": 1600}])


def test_mechanical_plug_component_type_must_match_kind():
    MechanicalPlug(md=2340, kind="bridge_plug", component_type="WBEQP.BRP")
    MechanicalPlug(md=2340, kind="cement_retainer", component_type="WBEQP.RET")
    with pytest.raises(ValueError, match="is not a bridge_plug"):
        MechanicalPlug(md=2340, kind="bridge_plug", component_type="WBEQP.RET")


def test_mechanical_plug_refused_inside_tubing():
    data = copy.deepcopy(BASE)
    data["completion"] = [{"type": "tubing", "od_in": 4.5, "top_md": 0,
                           "base_md": 2300}]
    data["mechanical_plugs"] = [{"name": "BP", "md": 2000}]
    with pytest.raises(ValueError, match="inside a tubing run"):
        WellSchematic.from_dict(data)
    # cut and pulled above the plug: allowed
    data["completion"][0]["base_md"] = 1000
    WellSchematic.from_dict(data)


# --- drawn by core --------------------------------------------------------
def _after_state():
    data = copy.deepcopy(BASE)
    data["completion"] = [{"type": "tubing", "od_in": 4.5, "top_md": 0,
                           "base_md": 1000}]
    data["mechanical_plugs"] = [{"name": "Bridge plug", "md": 2340,
                                 "kind": "bridge_plug"}]
    data["cement_plugs"] = [{"name": "Plug 1", "top_md": 2190,
                             "base_md": 2340}]
    data["casings"][2]["milled"] = [{"top_md": 1500, "base_md": 1550}]
    data["casings"][2]["annular_cement"] = [
        {"top_md": 1100, "base_md": 1150, "origin": "perf-wash-cement"}]
    return WellSchematic.from_dict(data)


def _layers(dwg, layer):
    return [e for e in dwg.entities if getattr(e, "layer", None) == layer]


def test_column_draws_the_after_state():
    before = build_column(WellSchematic.from_dict(copy.deepcopy(BASE)))
    after = build_column(_after_state())
    # the 9-5/8 wall is drawn in two pieces each side (window) -> +2 polygons
    assert len(_layers(after, "CASING")) == len(_layers(before, "CASING")) + 2
    # the perf-wash-cement interval adds cement behind the 9-5/8
    assert len(_layers(after, "CEMENT")) > len(_layers(before, "CEMENT"))
    # the bridge plug is on the plug layer with the cement plug
    assert len(_layers(after, "PLUG")) == 2


def test_section_draws_the_after_state():
    before = build_section(WellSchematic.from_dict(copy.deepcopy(BASE)))
    after = build_section(_after_state())
    assert len(_layers(after, "CASING")) == len(_layers(before, "CASING")) + 1
    # +1 perf-wash-cement; +1 because the primary cement is split at the
    # milled window, whose cement removal is not recorded (so not drawn)
    assert len(_layers(after, "CEMENT")) == len(_layers(before, "CEMENT")) + 2
    assert len(_layers(after, "PLUG")) == 2


def _shoe_triangles(ax):
    return [p for p in ax.patches
            if len(p.get_xy()) == 4 and tuple(p.get_facecolor()[:3]) == (0, 0, 0)]


def test_plumbing_draws_the_after_state_and_no_shoe_on_a_screen():
    fig, ax = plt.subplots()
    render_plumbing(_after_state(), ax=ax)
    texts = [t.get_text() for t in ax.texts]
    assert "Bridge plug" in texts
    plt.close(fig)
    # a screen has no shoe: the plumbing view must not draw one
    data = copy.deepcopy(BASE)
    data["casings"].append({"name": "Screen", "od_in": 5.5, "id_in": 4.9,
                            "top_md": 2400, "shoe_md": 2590, "kind": "screen"})
    fig, ax = plt.subplots()
    render_plumbing(WellSchematic.from_dict(data), ax=ax)
    n_with_screen = len(_shoe_triangles(ax))
    plt.close(fig)
    fig, ax = plt.subplots()
    render_plumbing(WellSchematic.from_dict(copy.deepcopy(BASE)), ax=ax)
    assert n_with_screen == len(_shoe_triangles(ax)) == 2 * 3   # 3 casings
    plt.close(fig)


def test_milled_cement_drawn_only_when_recorded_as_remaining():
    kept = _casing(milled=[{"top_md": 1500, "base_md": 1550,
                            "cement_removed": False}])
    assert kept.cement_intervals() == [(1400.0, 2600.0)]
    for removed in (True, None):         # removed, or NOT RECORDED
        c = _casing(milled=[{"top_md": 1500, "base_md": 1550,
                             "cement_removed": removed}])
        assert c.cement_intervals() == [(1400.0, 1500), (1550, 2600.0)]


def test_cement_records_keep_origin_unmerged():
    c = _casing(annular_cement=[
        {"top_md": 1350, "base_md": 1450, "origin": "squeeze"}])
    assert c.cement_records() == [(1350, 1450, "squeeze"),
                                  (1400.0, 2600.0, "primary")]


def test_cut_and_pull():
    c = _casing(toc_md=1400.0, cut_md=1200.0)
    assert c.steel_intervals() == [(1200.0, 2600.0)]
    assert not c.steel_at(1100) and c.steel_at(1300)
    with pytest.raises(ValueError, match="could not have been recovered"):
        _casing(toc_md=1000.0, cut_md=1200.0)        # cut below the TOC
    with pytest.raises(ValueError, match="not behind the string"):
        _casing(cut_md=1200.0, annular_cement=[{"top_md": 1100,
                                                "base_md": 1300}])
    with pytest.raises(ValueError, match="not on the string"):
        _casing(cut_md=2700.0)


def test_cut_stub_draws_no_hanger():
    """A cut stub stands on its own cement: no hanger. Moving top_md down to
    the cut (the workaround before cut_md) drew one that does not exist."""
    def hangers(**nine_five_eighths):
        data = copy.deepcopy(BASE)
        data["casings"][2].update(nine_five_eighths)
        dwg = build_column(WellSchematic.from_dict(data))
        return [e for e in dwg.entities
                if getattr(e, "layer", None) == "HANGER"]
    assert len(hangers(cut_md=1100.0)) == 0
    assert len(hangers(top_md=1100.0)) > 0       # the workaround: a false hanger
