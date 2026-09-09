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


# --- annulus outer boundary is depth-aware ---------------------------------- #
def test_annulus_never_overshoots_into_rock():
    """Regression: cement and fluid must stop at the ACTUAL annulus boundary.

    The outer boundary was taken as the next-outer casing ID regardless of
    depth. Below that casing's shoe the real boundary is the drilled HOLE, so
    the fill was drawn an inch into rock -- visible as the open-hole wall line
    running through the fluid colour.
    """
    from welleng.schematic.column import L_CEMENT, L_FLUID
    from welleng.schematic.drawing import Hatch as DwgHatch
    from welleng.schematic.drawing import Polygon as DwgPolygon
    d = {"well": {"name": "T"},
         "survey": {"md": [0, 500], "inc": [0, 0], "azi": [0, 0]},
         "hole_sections": [
             {"bit_in": 36.0, "top_md": 0, "base_md": 80, "radial_scale": 50},
             {"bit_in": 26.0, "top_md": 80, "base_md": 500, "radial_scale": 50},
         ],
         "casings": [
             # conductor ends at 80 m; below that the 26in HOLE bounds the annulus
             {"name": '30"', "od_in": 30.0, "id_in": 28.0,
              "top_md": 0, "shoe_md": 80, "toc_md": 0},
             {"name": '20"', "od_in": 20.0, "id_in": 18.7,
              "top_md": 0, "shoe_md": 500, "toc_md": 0},
         ],
         "annulus_fluids": [
             {"name": "Seawater", "inside_od_in": 20, "top_md": 0, "base_md": 500},
         ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    scale = 50.0
    checked = 0
    for e in dwg.entities:
        if getattr(e, "layer", None) not in (L_FLUID, L_CEMENT):
            continue
        if not isinstance(e, (DwgPolygon, DwgHatch)):
            continue
        pts = e.boundary if isinstance(e, DwgHatch) else e.points
        xs = [abs(x) / scale for (x, _y) in pts]
        ys = [y for (_x, y) in pts]
        if min(ys) < 79.0:
            continue                      # the 0-80 m segment, bounded by the 30in ID
        checked += 1
        assert max(xs) <= 13.0 + 1e-6, (
            f"annulus reaches r={max(xs):.2f}in below the conductor shoe, "
            "past the 26in hole wall at 13.00in"
        )
    assert checked, "no sub-conductor annulus segment was drawn"


# --- annotation does not sit on the drawing --------------------------------- #
def test_labels_live_in_a_gutter_clear_of_the_geometry():
    """Regression: labels overlapped the schematic and each other.

    Every string starts at surface, so casing names anchored at the top all
    landed on depth 0; and an annulus band is often narrower than its own fluid
    name. Labels now go in a side gutter with leaders.
    """
    from welleng.schematic.column import L_ANNOTATION, L_CASING, L_FLUID
    from welleng.schematic.drawing import Polygon as DwgPolygon
    from welleng.schematic.drawing import Text as DwgText
    s = _with([{"name": "WBM", "inside_od_in": 13.375, "top_md": 0,
                "base_md": 500, "density_sg": 1.35}])
    dwg = build_column(s, mode="MD")

    # widest drawn geometry (casing steel / fluid bands)
    geom_x = max(
        abs(x)
        for e in dwg.entities
        if getattr(e, "layer", None) in (L_CASING, L_FLUID)
        and isinstance(e, DwgPolygon)
        for (x, _y) in e.points
    )
    texts = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_ANNOTATION
             and isinstance(e, DwgText)
             and not e.text.replace(".", "").isdigit()]     # skip depth ruler
    assert texts, "no annotation drawn"
    for t in texts:
        assert abs(t.position[0]) >= geom_x, (
            f"label {t.text!r} at x={t.position[0]:.1f} sits on the geometry "
            f"(which reaches {geom_x:.1f})"
        )

    # and no two labels on the same side share a depth
    for side in (1, -1):
        ys = sorted(t.position[1] for t in texts
                    if (t.position[0] >= 0) == (side > 0))
        for a, b in zip(ys, ys[1:]):
            assert b - a > 1e-6, "two labels placed at the same depth"


# --- a plug cannot be set through a live completion -------------------------- #
def test_plug_overlapping_tubing_is_refused():
    """You cannot set a cement plug with the tubing still in the hole.

    Left unchecked the schematic drew a plug with tubing running through it --
    an impossible well that reads as a real one.
    """
    d = {**BASE,
         "cement_plugs": [{"name": "Barrier plug", "top_md": 900, "base_md": 1100}],
         "completion": [{"type": "tubing", "od_in": 4.5,
                         "top_md": 0, "base_md": 1500}]}
    with pytest.raises(Exception) as exc:
        WellSchematic.model_validate(d)
    assert "still in the hole" in str(exc.value)


def test_plug_below_the_tubing_shoe_is_fine():
    d = {**BASE,
         "cement_plugs": [{"name": "Reservoir plug", "top_md": 1600, "base_md": 1900}],
         "completion": [{"type": "tubing", "od_in": 4.5,
                         "top_md": 0, "base_md": 1500}]}
    s = WellSchematic.model_validate(d)
    assert len(s.primary.cement_plugs) == 1


def test_cut_and_pull_is_expressible():
    """Shortening the run to the cut depth clears the overlap -- the real
    operation the check must not forbid."""
    d = {**BASE,
         "cement_plugs": [{"name": "Plug", "top_md": 800, "base_md": 1000}],
         "completion": [{"type": "tubing", "od_in": 4.5,
                         "top_md": 0, "base_md": 700}]}   # cut at 700 m
    s = WellSchematic.model_validate(d)
    assert s.primary.completion[0].base_md == 700


# --- formation (rock) outside the hole wall --------------------------------- #
ROCK_DATA = {
    **BASE,
    "formations": [
        {"name": "Shale", "top_md": 0, "color": "#c9e6a8"},
        {"name": "Seal", "top_md": 1200, "color": "#5e35b1", "seal": True},
        {"name": "", "top_md": 2000, "color": "#ffffff"},
    ],
}


def test_rock_is_drawn_behind_the_well():
    """Rock must paint before casing/cement, or it covers the wellbore."""
    from welleng.schematic.column import L_CASING, L_ROCK
    dwg = build_column(WellSchematic.model_validate(ROCK_DATA), mode="MD")
    layers = [getattr(e, "layer", None) for e in dwg.entities]
    assert L_ROCK in layers
    assert layers.index(L_ROCK) < layers.index(L_CASING)


def test_rock_colour_is_muted_not_raw():
    """A raw seal/reservoir colour out-reads the wellbore itself."""
    from welleng.schematic.column import L_ROCK, _mute
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = build_column(WellSchematic.model_validate(ROCK_DATA), mode="MD")
    fills = {e.style.fill for e in dwg.entities
             if getattr(e, "layer", None) == L_ROCK and isinstance(e, DwgPolygon)}
    assert "#5e35b1" not in fills, "raw formation colour used for background"
    assert _mute("#5e35b1") in fills


def test_rock_stops_short_of_the_label_gutter():
    from welleng.schematic.column import L_ANNOTATION, L_ROCK
    from welleng.schematic.drawing import Polygon as DwgPolygon
    from welleng.schematic.drawing import Text as DwgText
    dwg = build_column(WellSchematic.model_validate(ROCK_DATA), mode="MD")
    rock_x = max(abs(x) for e in dwg.entities
                 if getattr(e, "layer", None) == L_ROCK
                 and isinstance(e, DwgPolygon) for (x, _y) in e.points)
    labels = [e for e in dwg.entities
              if getattr(e, "layer", None) == L_ANNOTATION
              and isinstance(e, DwgText)
              and not e.text.replace(".", "").isdigit()]
    assert labels
    for t in labels:
        assert abs(t.position[0]) >= rock_x, (
            f"label {t.text!r} sits on the rock band"
        )


def test_unidentified_formation_is_left_blank():
    """An interval with no colour must look unidentified, not be given one."""
    from welleng.schematic.column import L_ROCK
    from welleng.schematic.drawing import Polygon as DwgPolygon
    d = {**BASE, "formations": [
        {"name": "Unknown", "top_md": 0, "color": "#ffffff"},
        {"name": "", "top_md": 2000, "color": "#ffffff"},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    rock = [e for e in dwg.entities
            if getattr(e, "layer", None) == L_ROCK and isinstance(e, DwgPolygon)]
    assert not rock, "an uncoloured formation was filled anyway"


# --- liner hangers ---------------------------------------------------------- #
LINER_DATA = {
    **BASE,
    "survey": {"md": [0, 2600], "inc": [0, 0], "azi": [0, 0]},
    "hole_sections": [
        {"bit_in": 17.5, "top_md": 0, "base_md": 800, "radial_scale": 40},
        {"bit_in": 12.25, "top_md": 800, "base_md": 2000, "radial_scale": 32},
        {"bit_in": 8.5, "top_md": 2000, "base_md": 2600, "radial_scale": 30},
    ],
    "casings": [
        *BASE["casings"],
        {"name": '7" Liner', "od_in": 7.0, "id_in": 6.18,
         "top_md": 1900, "shoe_md": 2600, "toc_md": 1900},
    ],
}


def _liner_column(data=None):
    return build_column(
        WellSchematic.model_validate(data or LINER_DATA), mode="MD"
    )


def test_hanger_is_drawn_only_for_a_hung_string():
    """A string run from surface has nothing to hang from -- no hanger."""
    from welleng.schematic.column import L_HANGER
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = _liner_column()
    boxes = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_HANGER
             and isinstance(e, DwgPolygon)]
    # one liner, two sides -- not one per casing
    assert len(boxes) == 2, f"{len(boxes)} hanger boxes for one liner"


def test_hanger_sits_in_the_annulus_never_in_the_bore():
    """The hanger carries the liner on the string above: it belongs between
    the liner OD and the host ID, not inside the liner."""
    from welleng.schematic.column import L_HANGER
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = _liner_column()
    s = 32.0                                     # radial scale at 1900 m MD
    r_liner_od, r_host_id = 7.0 / 2.0, 8.68 / 2.0
    for e in dwg.entities:
        if getattr(e, "layer", None) != L_HANGER or not isinstance(e, DwgPolygon):
            continue
        xs = [abs(x) for x, _y in e.points]
        assert min(xs) == pytest.approx(r_liner_od * s, rel=1e-6)
        assert max(xs) == pytest.approx(r_host_id * s, rel=1e-6)


def test_hanger_is_mirrored_on_each_side():
    from welleng.schematic.column import L_HANGER
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = _liner_column()
    signs = set()
    for e in dwg.entities:
        if getattr(e, "layer", None) == L_HANGER and isinstance(e, DwgPolygon):
            signs.add(np.sign(sum(x for x, _y in e.points)))
    assert signs == {-1.0, 1.0}


def test_hanger_top_is_at_the_liner_top():
    from welleng.schematic.column import L_HANGER
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = _liner_column()
    boxes = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_HANGER
             and isinstance(e, DwgPolygon)]
    for e in boxes:
        assert min(y for _x, y in e.points) == pytest.approx(1900.0)


def test_hanger_height_scales_with_the_annulus_not_with_depth():
    """Sized off the gap it fills, so it reads square at any depth scale --
    the failure mode every other symbol here had was a fixed metre height."""
    from welleng.schematic.column import HANGER_ASPECT, L_HANGER
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = _liner_column()
    width_x = (8.68 - 7.0) / 2.0 * 32.0
    for e in dwg.entities:
        if getattr(e, "layer", None) == L_HANGER and isinstance(e, DwgPolygon):
            ys = [y for _x, y in e.points]
            assert max(ys) - min(ys) == pytest.approx(
                width_x * HANGER_ASPECT, rel=1e-6
            )


def test_liner_with_no_host_string_is_skipped_not_fatal():
    """Bad data -- a liner hung deeper than every outer shoe -- must not raise."""
    from welleng.schematic.column import L_HANGER
    d = {**LINER_DATA, "casings": [
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 800, "toc_md": 600},
        {"name": '7" Liner', "od_in": 7.0, "id_in": 6.18,
         "top_md": 1900, "shoe_md": 2600, "toc_md": 1900},
    ]}
    dwg = _liner_column(d)
    assert not [e for e in dwg.entities
                if getattr(e, "layer", None) == L_HANGER]


# --- perforations ----------------------------------------------------------- #
def _perf_column(perfs, data=None):
    d = {**(data or LINER_DATA), "perforations": perfs}
    return build_column(WellSchematic.model_validate(d), mode="MD")


def test_flat_form_hoists_perforations():
    s = WellSchematic.model_validate({
        **LINER_DATA,
        "perforations": [{"top_md": 2300, "base_md": 2400}],
    })
    assert len(s.wellbores[0].perforations) == 1
    assert s.wellbores[0].perforations[0].base_md == 2400


def test_perforations_cross_the_shot_wall_into_the_annulus():
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2300, "base_md": 2400,
                         "casing_od_in": 7.0}])
    ticks = [e for e in dwg.entities if getattr(e, "layer", None) == L_PERF]
    assert ticks
    s = 30.0                                    # radial scale below 2000 m
    r_in, r_hole = 6.18 / 2.0, 8.5 / 2.0
    for e in ticks:
        x0, x1 = abs(e.start[0]), abs(e.end[0])
        assert min(x0, x1) == pytest.approx(r_in * s, rel=1e-6)
        assert max(x0, x1) == pytest.approx(r_hole * s, rel=1e-6)


def test_perforations_never_reach_into_the_bore():
    """A tick inside the casing ID would read as a hole in the tubing."""
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2300, "base_md": 2400}])
    r_in = 6.18 / 2.0 * 30.0
    for e in dwg.entities:
        if getattr(e, "layer", None) == L_PERF:
            assert min(abs(e.start[0]), abs(e.end[0])) >= r_in - 1e-9


def test_perforations_are_mirrored_on_each_side():
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2300, "base_md": 2400}])
    signs = {np.sign(e.start[0]) for e in dwg.entities
             if getattr(e, "layer", None) == L_PERF}
    assert signs == {-1.0, 1.0}


def test_perforations_default_to_the_innermost_string_there():
    """With no ``casing_od_in`` the shots go through the string actually in the
    way -- the liner, not the 9-5/8in behind it."""
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2300, "base_md": 2400}])
    inner = min(abs(e.start[0]) for e in dwg.entities
                if getattr(e, "layer", None) == L_PERF)
    assert inner == pytest.approx(6.18 / 2.0 * 30.0, rel=1e-6)


def test_perforations_can_name_an_outer_string():
    """Shot through the 9-5/8in above the liner top: the marks must start at
    THAT wall, otherwise the named string is ignored."""
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 1500, "base_md": 1600,
                         "casing_od_in": 9.625}])
    inner = min(abs(e.start[0]) for e in dwg.entities
                if getattr(e, "layer", None) == L_PERF)
    assert inner == pytest.approx(8.68 / 2.0 * 32.0, rel=1e-6)


def test_perforations_naming_an_absent_string_are_skipped():
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2300, "base_md": 2400,
                         "casing_od_in": 13.375}])
    assert not [e for e in dwg.entities
                if getattr(e, "layer", None) == L_PERF]


@pytest.mark.parametrize("top,base", [(2400, 2400), (2400, 2300)])
def test_inverted_or_zero_perforated_interval_is_skipped(top, base):
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": top, "base_md": base}])
    assert not [e for e in dwg.entities
                if getattr(e, "layer", None) == L_PERF]


def test_a_long_perforated_zone_stays_legible():
    """Tick count is capped: one mark per shot would be a solid black band,
    which reads as missing casing rather than as perforations."""
    from welleng.schematic.column import L_PERF
    dwg = _perf_column([{"top_md": 2000, "base_md": 2600,
                         "shots_per_m": 39}])
    depths = {round(e.start[1], 6) for e in dwg.entities
              if getattr(e, "layer", None) == L_PERF}
    assert 2 <= len(depths) <= 41, f"{len(depths)} tick depths"


def test_perforations_paint_over_cement_and_plugs():
    """Order matters: cement in the perforated annulus is drawn first, so the
    marks must come after it or they vanish under the fill."""
    from welleng.schematic.column import L_CEMENT, L_PERF, L_PLUG
    d = {**LINER_DATA,
         "cement_plugs": [{"name": "P", "top_md": 2300, "base_md": 2500}],
         "perforations": [{"top_md": 2300, "base_md": 2400}]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    layers = [getattr(e, "layer", None) for e in dwg.entities]
    assert layers.index(L_PERF) > layers.index(L_CEMENT)
    assert layers.index(L_PERF) > layers.index(L_PLUG)


# --- callouts vs the depth ruler -------------------------------------------- #
def test_callouts_clear_the_depth_ruler_numbers():
    """The left gutter is shared with the depth numbers. De-colliding the
    callouts only among themselves put "Reservoir plug" on top of "2500"."""
    from welleng.schematic.column import L_ANNOTATION, _ruler_depths
    from welleng.schematic.drawing import Text as DwgText
    d = {**LINER_DATA, "cement_plugs": [
        {"name": "Reservoir plug", "top_md": 2350, "base_md": 2600},
        {"name": "Barrier plug", "top_md": 2100, "base_md": 2260},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    ruler = set(_ruler_depths(2600.0))
    texts = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_ANNOTATION
             and isinstance(e, DwgText)]
    numbers = {round(t.position[1], 6) for t in texts
               if t.text.replace(".", "").isdigit()}
    assert numbers & {round(r, 6) for r in ruler}, "ruler numbers not found"
    min_gap = 2600.0 * 0.022
    for t in texts:
        if t.text.replace(".", "").isdigit() or t.position[0] >= 0:
            continue
        for r in ruler:
            assert abs(t.position[1] - r) >= min_gap - 1e-9, (
                f"callout {t.text!r} at {t.position[1]:.0f} sits on ruler {r:.0f}"
            )


def test_a_free_slot_search_moves_up_when_that_is_nearer():
    """One-way nudging can only push a label further into what it hit."""
    from welleng.schematic.column import _free_slot
    y = _free_slot(2475.0, [2500.0], 57.0, 2600.0)
    assert y < 2475.0, f"pushed away from its own depth: {y}"
    assert abs(y - 2500.0) >= 57.0


# --- toc_md: an unstated TOC is not "cemented to surface" ------------------- #
def _no_toc_data():
    """BASE with the toc_md field simply omitted on both strings."""
    return {**BASE, "casings": [
        {"name": '13-3/8"', "od_in": 13.375, "id_in": 12.4,
         "top_md": 0, "shoe_md": 800},
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 2000},
    ]}


def test_unstated_toc_draws_no_cement():
    """A schematic is read as a BARRIER drawing. Defaulting toc_md to 0.0 made
    an omitted TOC render as the maximum possible cement -- a false statement
    about a barrier, on exactly the strings least likely to have one."""
    dwg = build_column(WellSchematic.model_validate(_no_toc_data()), mode="MD")
    assert not [e for e in dwg.entities
                if getattr(e, "layer", None) == L_CEMENT], (
        "an unstated TOC was drawn as cement"
    )


def test_unstated_toc_is_none_not_zero():
    s = WellSchematic.model_validate(_no_toc_data())
    assert s.wellbores[0].casings[0].toc_md is None


def test_explicit_zero_toc_still_means_cemented_to_surface():
    """The caller that means it must still be believed."""
    d = {**BASE, "casings": [
        {"name": '13-3/8"', "od_in": 13.375, "id_in": 12.4,
         "top_md": 0, "shoe_md": 800, "toc_md": 0.0},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    cement = [e for e in dwg.entities
              if getattr(e, "layer", None) == L_CEMENT]
    assert cement
    tops = [min(y for _x, y in e.boundary) for e in cement]
    assert min(tops) == pytest.approx(0.0)


def test_toc_is_per_string_not_all_or_nothing():
    d = {**BASE, "casings": [
        {"name": "driven conductor", "od_in": 13.375, "id_in": 12.4,
         "top_md": 0, "shoe_md": 800},                      # no cement, ever
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 2000, "toc_md": 1200},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    cement = [e for e in dwg.entities
              if getattr(e, "layer", None) == L_CEMENT]
    assert cement
    assert min(min(y for _x, y in e.boundary) for e in cement) \
        == pytest.approx(1200.0)


# --- tapered / combination string ------------------------------------------- #
COMBO = {
    **BASE,
    "survey": {"md": [0, 2100], "inc": [0, 0], "azi": [0, 0]},
    "hole_sections": [
        {"bit_in": 17.5, "top_md": 0, "base_md": 800, "radial_scale": 40},
        {"bit_in": 12.25, "top_md": 800, "base_md": 2100, "radial_scale": 32},
    ],
    "casings": [
        {"name": '13-3/8"', "od_in": 13.375, "id_in": 12.4,
         "top_md": 0, "shoe_md": 800, "toc_md": 500},
        # ONE string on ONE hanger: 10-3/4in to 191.6 m, then 9-5/8in to shoe
        {"name": "production", "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 1997.6, "toc_md": 1400,
         "sections": [
             {"od_in": 10.75, "id_in": 9.76, "top_md": 0, "base_md": 191.6},
             {"od_in": 9.625, "id_in": 8.68, "top_md": 191.6,
              "base_md": 1997.6},
         ]},
    ],
}


def test_combination_string_has_one_shoe_not_one_per_diameter():
    """A shoe at the crossover asserts the string ENDS there and the annulus
    opens below it. Both false -- and a fabricated shoe is a false statement
    about well architecture, where a missing diameter step is cosmetic."""
    from welleng.schematic.column import L_SHOE
    from welleng.schematic.drawing import SymbolRef
    dwg = build_column(WellSchematic.model_validate(COMBO), mode="MD")
    shoes = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_SHOE
             and isinstance(e, SymbolRef)]
    depths = sorted({round(e.position[1], 3) for e in shoes})
    assert depths == [800.0, 1997.6], f"shoe depths {depths}"


def test_combination_string_draws_steel_at_both_diameters():
    from welleng.schematic.column import L_CASING
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = build_column(WellSchematic.model_validate(COMBO), mode="MD")
    widths = set()
    for e in dwg.entities:
        if getattr(e, "layer", None) == L_CASING and isinstance(e, DwgPolygon):
            xs = [abs(x) for x, _y in e.points]
            widths.add(round(max(xs) / 40.0 * 2.0, 3))   # -> OD in inches
    assert 10.75 in widths and 9.625 in widths, sorted(widths)


def test_od_at_and_id_at_follow_the_profile():
    s = WellSchematic.model_validate(COMBO)
    c = next(c for c in s.wellbores[0].casings if c.name == "production")
    assert c.od_at(100.0) == 10.75
    assert c.id_at(100.0) == 9.76
    assert c.od_at(1000.0) == 9.625
    assert c.crossovers() == [191.6]
    assert c.has_od(10.75) and c.has_od(9.625)
    assert not c.has_od(7.0)


def test_annulus_wall_steps_at_the_crossover():
    """The annulus is keyed on the OD of the string forming the inner wall,
    and that OD CHANGES at the crossover."""
    from welleng.schematic.column import L_FLUID
    from welleng.schematic.drawing import Polygon as DwgPolygon
    d = {**COMBO, "annulus_fluids": [
        {"name": "WBM", "inside_od_in": 10.75, "top_md": 0, "base_md": 400,
         "density_sg": 1.35},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    inner = set()
    for e in dwg.entities:
        if getattr(e, "layer", None) == L_FLUID and isinstance(e, DwgPolygon):
            xs = [abs(x) for x, _y in e.points]
            inner.add(round(min(xs) / 40.0 * 2.0, 3))      # -> OD in inches
    assert inner == {10.75, 9.625}, sorted(inner)


def test_a_fluid_naming_either_diameter_finds_the_string():
    from welleng.schematic.column import L_FLUID
    for od in (10.75, 9.625):
        d = {**COMBO, "annulus_fluids": [
            {"name": "WBM", "inside_od_in": od, "top_md": 0, "base_md": 400},
        ]}
        dwg = build_column(WellSchematic.model_validate(d), mode="MD")
        assert [e for e in dwg.entities
                if getattr(e, "layer", None) == L_FLUID], f"od {od} not found"


@pytest.mark.parametrize("secs,msg", [
    ([{"od_in": 10.75, "id_in": 9.76, "top_md": 0, "base_md": 190.0},
      {"od_in": 9.625, "id_in": 8.68, "top_md": 191.6, "base_md": 1997.6}],
     "not contiguous"),
    ([{"od_in": 10.75, "id_in": 9.76, "top_md": 0, "base_md": 300.0},
      {"od_in": 9.625, "id_in": 8.68, "top_md": 191.6, "base_md": 1997.6}],
     "not contiguous"),
    ([{"od_in": 10.75, "id_in": 9.76, "top_md": 0, "base_md": 191.6},
      {"od_in": 9.625, "id_in": 8.68, "top_md": 191.6, "base_md": 1900.0}],
     "shoe_md"),
    ([{"od_in": 10.75, "id_in": 9.76, "top_md": 50.0, "base_md": 191.6},
      {"od_in": 9.625, "id_in": 8.68, "top_md": 191.6, "base_md": 1997.6}],
     "top_md"),
    ([{"od_in": 10.75, "id_in": 9.76, "top_md": 0, "base_md": 191.6},
      {"od_in": 7.0, "id_in": 6.18, "top_md": 191.6, "base_md": 1997.6}],
     "section at the shoe"),
])
def test_bad_combination_sections_are_refused(secs, msg):
    """A gap draws the string as two pieces with open annulus between them; an
    overlap draws two walls at one depth. Same fabricated-geometry class the
    field exists to remove, so it is refused rather than rendered."""
    d = {**COMBO, "casings": [
        {"name": "production", "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 1997.6, "sections": secs},
    ]}
    with pytest.raises(Exception, match=msg):
        WellSchematic.model_validate(d)


# --- consecutive tubing runs are one string --------------------------------- #
TAPERED_TUBING = {
    **BASE,
    "completion": [
        {"type": "tubing", "od_in": 5.5, "top_md": 0, "base_md": 1683.81},
        {"type": "tubing", "od_in": 4.5, "top_md": 1683.81, "base_md": 1764.19},
        {"type": "tubing", "od_in": 3.5, "top_md": 1764.19, "base_md": 1878.45},
    ],
}


def test_consecutive_tubing_runs_are_joined():
    """Drawn as independent polylines the runs render as floating pairs of
    lines with visible gaps, read as a discontinuity. The string is continuous."""
    from welleng.schematic.column import L_COMPLETION
    from welleng.schematic.drawing import Polyline as DwgPolyline
    dwg = build_column(WellSchematic.model_validate(TAPERED_TUBING), mode="MD")
    steps = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_COMPLETION
             and isinstance(e, DwgPolyline)
             and len(e.points) == 2
             and abs(e.points[0][1] - e.points[1][1]) < 1e-9]
    depths = sorted({round(e.points[0][1], 2) for e in steps})
    assert depths == [1683.81, 1764.19], f"joins at {depths}"


def test_the_join_spans_the_two_radii():
    from welleng.schematic.column import L_COMPLETION
    from welleng.schematic.drawing import Polyline as DwgPolyline
    dwg = build_column(WellSchematic.model_validate(TAPERED_TUBING), mode="MD")
    s = 32.0                                     # radial scale below 800 m
    for e in dwg.entities:
        if (getattr(e, "layer", None) == L_COMPLETION
                and isinstance(e, DwgPolyline) and len(e.points) == 2
                and abs(e.points[0][1] - e.points[1][1]) < 1e-9
                and round(e.points[0][1], 2) == 1683.81):
            xs = sorted(abs(x) for x, _y in e.points)
            assert xs == pytest.approx([4.5 / 2 * s, 5.5 / 2 * s])
            return
    pytest.fail("no join at the 5-1/2in -> 4-1/2in crossover")


def test_a_real_gap_between_tubing_runs_is_not_closed():
    """Only runs whose depths MEET are one string. Closing a genuine gap would
    draw a connection that is not there -- the same defect, inverted."""
    from welleng.schematic.column import L_COMPLETION
    from welleng.schematic.drawing import Polyline as DwgPolyline
    d = {**BASE, "completion": [
        {"type": "tubing", "od_in": 5.5, "top_md": 0, "base_md": 1000.0},
        {"type": "tubing", "od_in": 4.5, "top_md": 1200.0, "base_md": 1500.0},
    ]}
    dwg = build_column(WellSchematic.model_validate(d), mode="MD")
    steps = [e for e in dwg.entities
             if getattr(e, "layer", None) == L_COMPLETION
             and isinstance(e, DwgPolyline) and len(e.points) == 2
             and abs(e.points[0][1] - e.points[1][1]) < 1e-9]
    assert not steps, "a real gap in the completion was closed"


# --- a shoe is a barrier symbol: only a cased string gets one --------------- #
def _kinds_data(kind):
    return {**BASE, "casings": [
        {"name": '9-5/8"', "od_in": 9.625, "id_in": 8.68,
         "top_md": 0, "shoe_md": 1400, "toc_md": 1000},
        {"name": "5-1/2in screens", "od_in": 5.5, "id_in": 4.89,
         "top_md": 1350, "shoe_md": 2000, "kind": kind},
    ]}


@pytest.mark.parametrize("kind,n_shoes", [
    ("casing", 2), ("liner", 2), ("screen", 1), ("tubular", 1),
])
def test_only_a_cased_string_draws_a_shoe(kind, n_shoes):
    """Sand screens hung on a packer, and junk left in hole, are tubulars in
    the hole with no shoe. Drawing one asserts a barrier that is not there."""
    from welleng.schematic.column import L_SHOE
    from welleng.schematic.drawing import SymbolRef
    dwg = build_column(WellSchematic.model_validate(_kinds_data(kind)),
                       mode="MD")
    depths = {round(e.position[1], 1) for e in dwg.entities
              if getattr(e, "layer", None) == L_SHOE
              and isinstance(e, SymbolRef)}
    assert len(depths) == n_shoes, f"{kind}: shoes at {sorted(depths)}"
    if kind in ("screen", "tubular"):
        assert 2000.0 not in depths


def test_a_shoeless_string_is_still_labelled():
    """No shoe must not mean no name -- the string is on the drawing."""
    from welleng.schematic.column import L_ANNOTATION
    from welleng.schematic.drawing import Text as DwgText
    dwg = build_column(WellSchematic.model_validate(_kinds_data("screen")),
                       mode="MD")
    texts = {e.text for e in dwg.entities
             if getattr(e, "layer", None) == L_ANNOTATION
             and isinstance(e, DwgText)}
    assert "5-1/2in screens" in texts


def test_a_shoeless_string_still_draws_its_steel():
    from welleng.schematic.column import L_CASING
    from welleng.schematic.drawing import Polygon as DwgPolygon
    dwg = build_column(WellSchematic.model_validate(_kinds_data("screen")),
                       mode="MD")
    deep = [e for e in dwg.entities
            if getattr(e, "layer", None) == L_CASING
            and isinstance(e, DwgPolygon)
            and max(y for _x, y in e.points) > 1900]
    assert deep, "the screen joint itself vanished"


def test_kind_defaults_to_casing_so_existing_data_is_unchanged():
    s = WellSchematic.model_validate(BASE)
    assert all(c.kind == "casing" and c.has_shoe for c in s.wellbores[0].casings)
