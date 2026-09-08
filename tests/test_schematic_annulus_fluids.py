"""Annulus fluids in the well schematic, and the un-roughened hole wall."""
import numpy as np
import pytest

from welleng.schematic import AnnulusFluid, WellSchematic, build_column
from welleng.schematic.column import L_CEMENT, L_FLUID, _fluid_fill

BASE = {
    "well": {"name": "T"},
    "survey": {"md": [0, 2000], "inc": [0, 0], "azi": [0, 0]},
    "hole_sections": [
        {"bit_in": 17.5, "top_md": 0, "base_md": 800, "radial_scale": 40},
        {"bit_in": 12.25, "top_md": 800, "base_md": 2000, "radial_scale": 32},
    ],
    "casings": [
        {"name": '13-3/8"', "od_in": 13.375, "id_in": 12.4,
         "top_md": 0, "shoe_md": 800, "toc_md": 500},
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 2000, "toc_md": 1200},
    ],
}


def _with(fluids):
    d = {**BASE, "annulus_fluids": fluids}
    return WellSchematic.model_validate(d)


def test_flat_form_hoists_annulus_fluids():
    s = _with([{"name": "WBM", "inside_od_in": 13.375,
                "top_md": 0, "base_md": 500, "density_sg": 1.35}])
    assert len(s.primary.annulus_fluids) == 1
    assert s.primary.annulus_fluids[0].density_sg == 1.35


def test_density_is_optional_and_not_inferred():
    """An unstated density stays None -- it must not be guessed from the name."""
    f = AnnulusFluid(name="Seawater", inside_od_in=20, top_md=0, base_md=400)
    assert f.density_sg is None


def test_fluid_is_drawn_on_its_own_layer():
    s = _with([{"name": "WBM", "inside_od_in": 13.375, "top_md": 0, "base_md": 500}])
    dwg = build_column(s, mode="MD")
    layers = [getattr(e, "layer", None) for e in dwg.entities]
    assert L_FLUID in layers


def test_cement_is_drawn_after_fluid_so_it_paints_over():
    """A cemented interval must read as cement even where a fluid overlaps."""
    s = _with([{"name": "WBM", "inside_od_in": 13.375, "top_md": 0, "base_md": 800}])
    dwg = build_column(s, mode="MD")
    layers = [getattr(e, "layer", None) for e in dwg.entities]
    assert layers.index(L_FLUID) < layers.index(L_CEMENT)


def test_fluid_naming_a_missing_annulus_is_skipped_not_fatal():
    s = _with([{"name": "Ghost", "inside_od_in": 5.5, "top_md": 0, "base_md": 400}])
    dwg = build_column(s, mode="MD")
    assert L_FLUID not in [getattr(e, "layer", None) for e in dwg.entities]


def test_inverted_or_zero_interval_is_skipped():
    s = _with([{"name": "WBM", "inside_od_in": 13.375, "top_md": 500, "base_md": 500}])
    dwg = build_column(s, mode="MD")
    assert L_FLUID not in [getattr(e, "layer", None) for e in dwg.entities]


@pytest.mark.parametrize("name,expected", [
    ("Seawater", "#cfe3f2"),
    ("WBM", "#cfd8c0"),
    ("Packer fluid (CaCl2)", "#e2d6ef"),
    ("Something unheard of", "#e6eef3"),
])
def test_fill_falls_back_on_the_name(name, expected):
    assert _fluid_fill(AnnulusFluid(name=name, inside_od_in=9.625,
                                    top_md=0, base_md=1)) == expected


def test_explicit_colour_beats_the_name():
    f = AnnulusFluid(name="Seawater", inside_od_in=9.625, top_md=0, base_md=1,
                     colour="#123456")
    assert _fluid_fill(f) == "#123456"


def test_label_reports_density_only_when_known():
    s = _with([
        {"name": "WBM", "inside_od_in": 13.375, "top_md": 0, "base_md": 500,
         "density_sg": 1.35},
        {"name": "Seawater", "inside_od_in": 9.625, "top_md": 0, "base_md": 500},
    ])
    dwg = build_column(s, mode="MD")
    texts = [e.text for e in dwg.entities if hasattr(e, "text")]
    assert any("WBM (1.35 sg)" == t for t in texts)
    assert any("Seawater" == t for t in texts)     # no "(None sg)"
    assert not any("None" in t for t in texts)


# --- the open hole is drawn at gauge, not roughened ------------------------- #
def test_hole_wall_is_not_roughened():
    """Regression: the drilled wall is a clean line at the gauge radius.

    A drawn wobble implies washout detail the schematic does not hold; a real
    caliper would be plotted as data instead.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import welleng.schematic.plumbing as pl

    assert not hasattr(pl, "_bump"), "hole-wall roughness reintroduced"

    # geometric check: every plotted wall point sits on the gauge radius
    s = _with([])
    res = pl.DepthResolver(s.primary.survey)
    cl = pl.Centreline(res, vs_azi=0.0, mode="MD")
    fig, ax = plt.subplots()
    pl._draw_hole(ax, cl, s.primary, 100.0, 700.0)   # inside the 17.5in section
    r_expected = cl.rdraw(17.5 / 2.0)
    walls = [ln for ln in ax.lines][:2]              # the two side walls
    assert walls, "no hole wall drawn"
    for ln in walls:
        # vertical hole: |x| is exactly the gauge radius, no outward noise
        assert np.allclose(np.abs(ln.get_xdata()), r_expected, atol=1e-9), \
            "hole wall deviates from gauge radius"
    plt.close(fig)


def test_plumbing_still_renders():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from welleng.schematic.plumbing import render_plumbing
    s = _with([{"name": "WBM", "inside_od_in": 13.375, "top_md": 0, "base_md": 500}])
    fig, ax = plt.subplots()
    render_plumbing(s, ax=ax)
    assert len(ax.lines) or len(ax.patches)
    plt.close(fig)


# --- casing shoe geometry --------------------------------------------------- #
def test_shoe_symbol_sits_outside_the_wall():
    """The wedge must not straddle the casing: no point on the inboard side.

    The old symbol was a symmetric triangle centred on the OD, which read as
    sitting across the wall rather than belonging to the string.
    """
    from welleng.schematic.symbols import casing_shoe
    pts = casing_shoe().entities[0].points
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    assert min(xs) == 0.0, "shoe crosses to the inboard side of the wall"
    assert max(xs) > 0.0, "shoe has no outward flare"
    # apex up-hole, base at the shoe depth
    assert min(ys) < 0.0 and max(ys) == 0.0


def test_shoe_is_mirrored_on_each_side():
    """Both sides flare OUTWARD, which needs a sign-flipped x scale."""
    from welleng.schematic.symbols import CASING_SHOE
    s = _with([])
    dwg = build_column(s, mode="MD")
    refs = [e for e in dwg.entities
            if getattr(e, "name", None) == CASING_SHOE]
    assert refs, "no shoes placed"
    assert any(r.sx < 0 for r in refs) and any(r.sx > 0 for r in refs)
    # a shoe on the left is placed at negative x, and vice versa
    for r in refs:
        assert r.position[0] * r.sx > 0, "shoe flares inboard"


def test_shoe_width_tracks_wall_not_od():
    """A 30in conductor and a 7in liner must both read sensibly.

    Sizing the wedge from the OD makes it grotesque on big pipe, because the
    OD is multiplied by the radial exaggeration.
    """
    from welleng.schematic.symbols import CASING_SHOE
    s = _with([])
    dwg = build_column(s, mode="MD")
    widths = {}
    for e in dwg.entities:
        if getattr(e, "name", None) == CASING_SHOE:
            widths[round(abs(e.position[0]), 3)] = abs(e.sx)
    assert len(widths) >= 2
    ratio = max(widths.values()) / min(widths.values())
    assert ratio < 6.0, f"shoe width varies too wildly across strings ({ratio:.1f}x)"


def test_shoe_anchor_coincides_with_the_drawn_wall():
    """Regression: no gap between a shoe and the casing it belongs to.

    _wall() scales each segment by the radial factor at the segment MIDPOINT,
    while the shoe depth sits exactly ON a section boundary and resolves to the
    next section's factor. Deriving the shoe anchor independently therefore put
    it at a different x from the wall. The anchor must come FROM the drawn wall.
    """
    from welleng.schematic.symbols import CASING_SHOE
    d = {**BASE, "hole_sections": [
        # a section boundary exactly at the 13-3/8in shoe, with a scale step
        {"bit_in": 17.5, "top_md": 0, "base_md": 800, "radial_scale": 40},
        {"bit_in": 12.25, "top_md": 800, "base_md": 2000, "radial_scale": 32},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    wall_pts = {(round(x, 6), round(y, 3))
                for e in dwg.entities if hasattr(e, "points")
                for (x, y) in e.points}
    shoes = [e for e in dwg.entities if getattr(e, "name", None) == CASING_SHOE]
    assert shoes
    for sh in shoes:
        key = (round(sh.position[0], 6), round(sh.position[1], 3))
        assert key in wall_pts, f"shoe at {key} is not on a drawn wall vertex"


# --- packer seals the annulus, nothing else --------------------------------- #
def test_packer_stays_between_tubing_and_casing_id():
    """Regression: a packer must not cross the casing wall or fill the bore.

    It was placed CENTRED on its sealing radius with a half-width of the same
    size, so it reached from half that radius to 1.5x it -- through the casing
    wall and into the cement -- and stood tens of metres tall.
    """
    from welleng.schematic.drawing import Polygon as DwgPolygon
    from welleng.schematic.column import L_COMPLETION
    d = {**BASE,
         "casings": [{"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
                      "top_md": 0, "shoe_md": 2000, "toc_md": 1200}],
         "completion": [
             {"type": "tubing", "od_in": 4.5, "top_md": 0, "base_md": 1800},
             {"type": "packer", "name": "Packer", "od_in": 8.68, "md": 1700},
         ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    # L_COMPLETION also carries the TUBING polylines, which have .points too --
    # select the packer's filled bands only.
    bands = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_COMPLETION
             and isinstance(e, DwgPolygon)]
    assert bands, "no packer band drawn"
    scale = 32.0                       # the deep section's radial exaggeration
    r_tbg, r_id, r_od = 4.5 / 2, 8.68 / 2, 9.625 / 2
    for b in bands:
        xs = [abs(x) / scale for (x, _y) in b.points]
        ys = [y for (_x, y) in b.points]
        assert max(xs) <= r_id + 1e-6, "packer crosses the casing ID"
        assert max(xs) < r_od, "packer reaches the casing OD"
        assert min(xs) >= r_tbg - 1e-6, "packer intrudes into the bore"
        assert (max(ys) - min(ys)) < 60.0, "packer is implausibly tall"


def test_packer_is_clamped_to_the_string_it_sets_in():
    """A packer seals against the string it is SET IN, not whatever od_in says.

    Regression: a packer carrying the 9-5/8in casing ID as its od_in, but set
    at a depth where a 7in LINER is the innermost string, was drawn out past
    the liner's wall and into the cement.
    """
    from welleng.schematic.drawing import Polygon as DwgPolygon
    from welleng.schematic.column import L_COMPLETION
    d = {**BASE,
         "casings": [
             {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
              "top_md": 0, "shoe_md": 1400, "toc_md": 1000},
             {"name": '7" Liner', "od_in": 7.0, "id_in": 6.18,
              "top_md": 1300, "shoe_md": 2000, "toc_md": 1300},
         ],
         "completion": [
             {"type": "tubing", "od_in": 4.5, "top_md": 0, "base_md": 1900},
             # od_in is the CASING ID, but at 1700 m the liner is the host
             {"type": "packer", "name": "Packer", "od_in": 8.68, "md": 1700},
         ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    bands = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_COMPLETION
             and isinstance(e, DwgPolygon)]
    assert bands, "no packer band drawn"
    scale = 32.0
    r_liner_id, r_liner_od = 6.18 / 2, 7.0 / 2
    for b in bands:
        xs = [abs(x) / scale for (x, _y) in b.points]
        assert max(xs) <= r_liner_id + 1e-6, (
            f"packer reaches {max(xs):.2f}in, past the liner ID {r_liner_id:.2f}in"
        )
        assert max(xs) < r_liner_od, "packer crosses the liner wall"
