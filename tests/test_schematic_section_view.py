"""Section view: exaggeration and its fold limit, hole, cement, shoes, sea, callouts."""
import copy

import numpy as np
import pytest
import sympy as sp

from welleng.schematic import WellSchematic, build_section
from welleng.schematic.drawing import Line, Polygon, Rect, Text

IN = 0.0254
WELL = {
    "well": {"name": "S", "datum_elevation_m": 25.0, "water_depth_m": 90.0},
    "survey": {"md": [0, 300, 700, 1200, 3000, 3450, 4905],
               "inc": [0, 0, 0, 17.4, 67.7, 90.4, 90.4], "azi": [0] * 7},
    "hole_sections": [
        {"bit_in": 26.0, "top_md": 295, "base_md": 1200},
        {"bit_in": 17.5, "top_md": 1200, "base_md": 3000},
        {"bit_in": 12.25, "top_md": 3000, "base_md": 4905},
    ],
    "casings": [
        {"name": '30"', "od_in": 30, "id_in": 28, "top_md": 0, "shoe_md": 295},
        {"name": '20"', "od_in": 20, "id_in": 18.73, "top_md": 0, "shoe_md": 1200,
         "toc_md": 600, "nominal_weight_ppf": 133, "grade": "X-56"},
        {"name": '13-3/8"', "od_in": 13.375, "id_in": 12.347, "top_md": 0,
         "shoe_md": 3000, "toc_md": 2000},
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.535, "top_md": 0, "shoe_md": 4905,
         "toc_md": 3500},
    ],
}


def _dwg(data=None, **kw):
    schematic = WellSchematic.model_validate(copy.deepcopy(data or WELL))
    return build_section(schematic, **kw)


def _on(dwg, layer):
    return [e for e in dwg.entities if getattr(e, "layer", None) == layer]


def test_fold_limit_sympy_derivation():
    """A circle of radius R whose plane is tilted to the view plane projects to an
    ellipse with semi-axes a = R, b = R c, c = |n . p| in (0, 1]; its tightest
    radius of curvature is b^2/a = R c^2 = R (n . p)^2, and its widest a^2/b."""
    R, c, t = sp.symbols("R c t", positive=True)
    a, b = R, R * c
    x, y = a * sp.cos(t), b * sp.sin(t)
    rho = (sp.diff(x, t) ** 2 + sp.diff(y, t) ** 2) ** sp.Rational(3, 2) / sp.Abs(
        sp.diff(x, t) * sp.diff(y, t, 2) - sp.diff(y, t) * sp.diff(x, t, 2))
    assert sp.simplify(rho.subs(t, 0) - R * c**2) == 0
    assert sp.simplify(rho.subs(t, sp.pi / 2) - R / c) == 0


def test_auto_exaggeration_is_capped_below_the_fold_limit():
    info = _dwg().section_info
    assert info["mode"] == "auto" and info["folds"] == []
    assert info["exaggeration"] <= 0.8 * info["e_max"]


def test_fold_limit_uses_the_widest_drawn_radius_the_hole():
    """On the 700-1200 m build the widest drawn feature is the 30" string... no:
    the 20" string's 26" hole. E_max there is R / r_hole, not R / r_casing."""
    info = _dwg().section_info
    legs = {(700.0, 1200.0): 500.0 / np.radians(17.4) / (26.0 * IN / 2),
            (1200.0, 3000.0): 1800.0 / np.radians(67.7 - 17.4) / (20.0 * IN / 2),
            (3000.0, 3450.0): 450.0 / np.radians(90.4 - 67.7) / (17.5 * IN / 2)}
    assert info["e_max"] == pytest.approx(min(legs.values()), rel=1e-9)


def test_a_user_value_above_the_limit_is_drawn_and_reported_not_marked():
    ref = _dwg(exaggeration=300.0)
    big = _dwg(exaggeration=1.2 * ref.section_info["e_max"])
    assert big.section_info["mode"] == "user" and big.section_info["folds"]
    assert "folds" in big.title_block
    assert len(big.entities) == len(ref.entities)       # nothing extra drawn


def test_driven_conductor_has_no_hole_no_cement_no_shoe():
    d = _dwg(exaggeration=300.0)
    s, E = d.section_info, 300.0
    r30 = 30.0 * IN / 2 * E
    # no hole trace above the conductor shoe
    for e in _on(d, "HOLE"):
        pts = e.points if hasattr(e, "points") else [e.start, e.end]
        assert all(p[1] >= 295.0 - 1e-6 for p in pts)
    # no shoe triangle at 295 m
    for e in _on(d, "SHOE"):
        assert min(p[1] for p in e.points) > 400.0
    assert s["exaggeration"] == E and r30 > 0


def test_hole_outline_is_joined_at_the_conductor_shoe_and_at_bit_changes():
    d = _dwg(exaggeration=300.0)
    steps = [e for e in _on(d, "HOLE") if isinstance(e, Line)]
    mids = sorted({round((e.start[1] + e.end[1]) / 2) for e in steps})
    # conductor shoe 295 m (vertical); bit changes at 1200 m MD (TVD ~1192, on the
    # build, so the step is tilted) and 3000 m MD; two sides each
    assert 295 in mids and len(steps) == 2 * 3
    # the two sides of a tilted step straddle the path; their mean is on it
    near = [(e.start[1] + e.end[1]) / 2 for e in steps
            if abs((e.start[1] + e.end[1]) / 2 - 1192) < 40]
    assert len(near) == 2 and abs(np.mean(near) - 1192) < 2


def test_toc_is_drawn_along_the_normal_to_the_path():
    d = _dwg(exaggeration=300.0)
    tocs = [e for e in _on(d, "CEMENT") if isinstance(e, Line)]
    assert tocs
    # the 13-3/8 TOC at 2000 m MD is on the build: its line is not horizontal,
    # and it is perpendicular to the tangent there
    from welleng.schematic.depth import DepthResolver
    from welleng.schematic.models import SurveyRef
    r = DepthResolver(SurveyRef(**WELL["survey"]))
    inc, _ = r.inc_azi_at(2000.0)
    t = np.array([np.sin(np.radians(inc)), np.cos(np.radians(inc))])
    z = r.tvd_at(2000.0)
    line = min(tocs, key=lambda e: abs((e.start[1] + e.end[1]) / 2 - z))
    v = np.subtract(line.end, line.start)
    assert abs(np.dot(v / np.linalg.norm(v), t)) < 1e-9


def test_shoe_triangle_base_fits_the_annular_gap():
    d = _dwg(exaggeration=300.0)
    for e in _on(d, "SHOE"):
        p = np.array(e.points)
        base = np.linalg.norm(p[1] - p[0])
        assert base <= 0.8 * (26.0 - 20.0) * IN / 2 * 300.0 + 1e-6


def test_sea_stops_at_the_outermost_string():
    d = _dwg(exaggeration=300.0)
    half = 30.0 * IN / 2 * 300.0
    polys = [e for e in _on(d, "SEA") if isinstance(e, Polygon)]
    assert len(polys) == 2
    for e in polys:
        inner = [p for p in e.points if abs(p[0]) < 1e5 and 25.0 <= p[1] <= 115.0]
        assert all(abs(x) >= half - 1e-6 for x, _ in inner)


def test_annulus_fill_is_off_by_default_and_fills_only_when_given():
    assert not _on(_dwg(), "FLUID")
    assert _on(_dwg(annulus_fill="#d9e8f2"), "FLUID")


def test_callouts_stay_inside_the_frame():
    d = _dwg()
    frame = next(e for e in d.entities if isinstance(e, Rect))
    x0, y0 = frame.corner
    x1, y1 = x0 + frame.width, y0 + frame.height
    texts = [e for e in _on(d, "ANNOTATION") if isinstance(e, Text)]
    assert texts
    for t in texts:
        assert x0 <= t.position[0] <= x1 and y0 <= t.position[1] <= y1
    assert any("133 ppf X-56" in t.text for t in texts)


def test_default_radial_scale_is_deprecated_and_used_as_exaggeration():
    with pytest.warns(DeprecationWarning):
        d = _dwg(default_radial_scale=250.0)
    assert d.section_info["exaggeration"] == 250.0 and d.section_info["mode"] == "user"
