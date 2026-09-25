"""Tests for the lithology column renderer (welleng.lithology).

Hermetic: no network, and the FGDC pattern assets are synthesised in a tmp dir,
since they are a separate CC0 download that a checkout will not have.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from welleng.lithology import (
    pattern_from_name,
    FgdcPatterns,
    LithologyError,
    TILES_PER_AXIS,
    group_of,
    intervals_from_nlog,
    intervals_from_tops,
    nl_groups,
    nl_groups_by_code,
    pattern_names,
    plot_lithology,
)

PIL = pytest.importorskip("PIL")
pytest.importorskip("matplotlib")


@pytest.fixture
def assets(tmp_path):
    """A tiny pattern asset dir: transparent background, one opaque line."""
    from PIL import Image
    d = tmp_path / "png"
    d.mkdir()
    for code in (607, 620, 623, 626, 658, 668):
        a = np.zeros((16, 16, 4), dtype=np.uint8)
        a[8, :, 3] = 255                      # one opaque row
        Image.fromarray(a, mode="RGBA").save(d / f"{code}.png")
    return str(d)


# --- reference data --------------------------------------------------------- #
def test_reference_table_is_complete():
    groups = nl_groups()
    assert len(groups) >= 10
    for g in groups:
        assert g.colour.startswith("#") and len(g.colour) == 7
    by_code = nl_groups_by_code()
    for code in ("NU", "NL", "CK", "KN", "ZE", "RO"):
        assert code in by_code, f"{code} missing from the NL table"


def test_verified_pattern_codes_are_locked():
    """The two corrections must not silently regress.

    The prototype had Chalk on 627 (Limestone) and salt on 642 (Dolostone);
    verified against the FGDC chart, chalk is 626 and salt is 668.
    """
    names = pattern_names()
    assert names[626] == "Chalk"
    assert names[668] == "Salt"
    assert names[627] == "Limestone"          # NOT chalk
    assert names[642] == "Dolostone or dolomite"   # NOT salt
    by_code = nl_groups_by_code()
    assert by_code["CK"].pattern == 626
    assert by_code["ZE"].pattern == 668


# --- RGD roll-up ------------------------------------------------------------ #
@pytest.mark.parametrize("unit,expected", [
    ("NU", "NU"),        # already group rank
    ("NLLF", "NL"),      # formation -> group
    ("CKGR", "CK"),
    ("CKTXM", "CK"),     # member -> group
    ("ZEZ1S", "ZE"),
    ("ROSL", "RO"),
])
def test_group_of_rolls_up(unit, expected):
    assert group_of(unit) == expected


def test_group_of_unknown_is_none():
    assert group_of("NOT_A_CODE") is None


# --- interval construction -------------------------------------------------- #
def _column(rows):
    return SimpleNamespace(intervals=[
        SimpleNamespace(top_md=t, bottom_md=b, unit_id=u) for t, b, u in rows
    ])


_ROWS = [(0, 358.3, "NU"), (358.3, 843.4, "NLLF"), (843.4, 1466.8, "CKTXM"),
         (1466.8, 1815.2, "KNGLU"), (1815.2, 2032.5, "ZEZ1S")]


def test_intervals_from_nlog_binds_by_code():
    ivs = intervals_from_nlog(_column(_ROWS), label="group")
    assert [i.name for i in ivs] == [
        "Upper North Sea Group", "Lower North Sea Group", "Chalk Group",
        "Rijnland Group", "Zechstein Group",
    ]
    assert ivs[2].pattern == 626 and ivs[4].pattern == 668


def test_the_default_resolves_at_formation_rank():
    """Group rank labels a cap-rock claystone member with its group's name and
    its group's pattern. The unit's own name is what a reader needs."""
    ivs = intervals_from_nlog(_column(_ROWS))
    assert ivs[0].name != "Upper North Sea Group"
    assert all(i.colour for i in ivs)          # colour still comes from the GROUP
    groups = intervals_from_nlog(_column(_ROWS), label="group")
    assert [i.colour for i in ivs] == [i.colour for i in groups]


def test_a_sandstone_member_does_not_inherit_the_evaporite_pattern():
    """The Z-series sandstones sit inside the Zechstein, which is an EVAPORITE
    group. A flow zone shaded as salt on a barrier drawing is the worst place
    for the group-rank simplification to land, and a consumer hit exactly this."""
    ivs = intervals_from_nlog(_column([(0, 100, "ZEZ1S")]))
    assert intervals_from_nlog(_column([(0, 100, "ZEZ1S")]),
                               label="group")[0].pattern == 668     # salt
    assert ivs[0].pattern == 607                                    # sand


def test_the_raw_code_is_still_reachable():
    for lbl in ("code", "unit"):
        assert intervals_from_nlog(_column([(0, 100, "NU")]),
                                   label=lbl)[0].name == "NU"


# --- lithology from the NAME, on word boundaries ---------------------------- #
@pytest.mark.parametrize("name,code", [
    ("Vlieland Claystone Formation", 620),
    ("Z4 Fringe Sandstone Member", 607),
    ("Chalk Group", 626),
    ("Holland Marl Member, Upper", 623),      # FGDC 623 IS calcareous shale/marl
    ("Coal Measures", 658),
])
def test_pattern_from_name_reads_the_rock(name, code):
    assert pattern_from_name(name) == code


@pytest.mark.parametrize("name,why", [
    ("Lower Buntsandstein Formation", "proper noun containing a lithology word"),
    ("Ommelanden Formation", "name says nothing about the rock"),
    ("Sandstone and Claystone Member", "two lithologies; picking one is a coin toss"),
    ("Anhydrite Member", "anhydrite is not gypsum and 667 is gypsum only"),
    ("", "no name"),
    (None, "no name"),
])
def test_pattern_from_name_refuses_rather_than_guesses(name, why):
    assert pattern_from_name(name) is None, why


def test_buntsandstein_is_the_case_word_boundaries_exist_for():
    """It contains 'sandstein' and logs 127 gAPI -- claystone-dominated. A
    substring match draws the Dutch unit as clean sand."""
    assert pattern_from_name("Lower Buntsandstein Formation") is None
    assert pattern_from_name("Bunter Sandstone Formation") == 607


def test_every_referenced_pattern_code_has_the_standard_wording():
    """A pattern code with no name cannot be explained to a reader, and 623 was
    referenced by the NL table while missing from the map."""
    from welleng.lithology import nl_groups, pattern_names
    named = set(pattern_names())
    assert {g.pattern for g in nl_groups()} <= named


def test_a_long_unit_name_is_not_broken_mid_word():
    """'Lower Buntsa / ndstein Formation' is not a wrap, it is a different
    word -- and the name is a proper noun."""
    import textwrap
    wrapped = textwrap.fill("Lower Buntsandstein Formation", 11,
                            break_long_words=False, break_on_hyphens=False)
    assert "Buntsandstein" in wrapped


def test_unknown_unit_is_white_not_guessed():
    ivs = intervals_from_nlog(_column([(0, 100, "NU"), (100, 200, "ZZZZ")]))
    assert ivs[1].colour == "#ffffff"
    assert ivs[1].pattern is None


def test_interval_without_base_is_dropped():
    ivs = intervals_from_nlog(_column([(0, 100, "NU"), (100, None, "NL")]))
    assert len(ivs) == 1


def test_intervals_from_tops_unknown_is_white():
    ivs = intervals_from_tops([("Chalk Group", 0), ("Mystery Fm", 500)], base=900)
    assert ivs[0].pattern == 626
    assert ivs[1].colour == "#ffffff" and ivs[1].pattern is None


# --- patterns --------------------------------------------------------------- #
def test_patterns_absent_is_not_fatal(tmp_path):
    p = FgdcPatterns(None)
    assert p.available is False
    assert p.tile(626, 100.0, 10.0) is None
    out = tmp_path / "flat.png"
    plot_lithology(intervals_from_nlog(_column([(0, 500, "CK")])),
                   out=str(out), patterns=None)
    assert out.exists() and out.stat().st_size > 500


def test_tile_repeats_are_zoom_independent(assets):
    """Regression: repeats must follow the VISIBLE window, not a fixed depth.

    Sizing tiles from interval thickness against a constant made a thick band
    render as one stretched tile when zoomed in.
    """
    p = FgdcPatterns(assets)
    thickness = 213.0
    wide = p.tile(668, thickness, 2093.0 / TILES_PER_AXIS)     # whole well
    zoom = p.tile(668, thickness, 279.0 / TILES_PER_AXIS)      # 280 m window
    assert wide.shape[0] == 16                                  # a single tile
    assert zoom.shape[0] > wide.shape[0]                        # many more
    assert zoom.shape[0] // 16 == pytest.approx(8, abs=1)


def test_native_alpha_is_used(assets):
    p = FgdcPatterns(assets)
    t = p.tile(626, 100.0, 50.0)
    assert t.shape[-1] == 4
    assert t[..., 3].min() == 0.0 and t[..., 3].max() == 1.0


# --- rendering -------------------------------------------------------------- #
def test_depth_range_filters_and_clips(assets, tmp_path):
    ivs = intervals_from_nlog(_column([(0, 1000, "NU"), (1000, 2000, "ZE")]))
    ax = plot_lithology(ivs, depth_range=(1200, 1800), patterns=assets)
    assert ax.get_ylim() == (1800, 1200)          # deep at the BOTTOM
    with pytest.raises(LithologyError):
        plot_lithology(ivs, depth_range=(5000, 6000), patterns=assets)


def test_empty_intervals_raises():
    with pytest.raises(LithologyError):
        plot_lithology([])


def test_composed_plot_orientation_and_size(assets, tmp_path):
    """Regressions: depth must increase DOWNWARD, and a narrow log window on a
    full-well column must not blow the canvas up via unclipped labels."""
    from welleng.exchange.las import open_las
    from welleng.lithology import plot_log_with_lithology
    las_text = (
        "~Version\nVERS. 2.0 :\nWRAP. NO :\n~Well\nSTRT.M 1700.0 :\n"
        "STOP.M 1710.0 :\nSTEP.M 1.0 :\nNULL. -999.25 :\nWELL. T-1 :\n"
        "~Curve\nDEPT.M :\nGR .GAPI :\n~ASCII\n"
        + "".join(f" {1700.0+i:.4f} {40+i:.4f}\n" for i in range(11))
    )
    las = open_las(las_text)
    ivs = intervals_from_nlog(_column([(0, 1500, "NU"), (1500, 2100, "ZE")]))
    out = tmp_path / "composed.png"
    fig = plot_log_with_lithology(las, ivs, depth_range=(1700, 1710),
                                  patterns=assets, out=str(out))
    lo, hi = fig.axes[0].get_ylim()
    assert lo > hi, "depth must increase downward"
    assert (lo, hi) == (1710, 1700)
    w, h = fig.get_size_inches()
    assert h < 20, "figure grew to fit out-of-window artists"
    assert out.exists()


# --- measured label fit, and the leader that catches what does not fit ------ #
def _band_column():
    from welleng.lithology import Interval
    return [
        Interval(name="North Sea Group, Upper", top=0.0, base=350.0,
                 colour="#dcefb8", pattern=607, note=""),
        Interval(name="Holland Marl Member, Upper", top=350.0, base=391.0,
                 colour="#9ec27a", pattern=623, note=""),      # 41 m — thin
        Interval(name="Z4 Fringe Sandstone Member", top=391.0, base=1100.0,
                 colour="#d9c7f0", pattern=607, note=""),
    ]


def test_the_measurement_actually_measures():
    """The regression for a measurement that silently returned None for every
    label: everything then 'did not fit' and every name went to the gutter,
    while the code reported itself as measuring. A proxy that always says the
    same thing is not a measurement."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from welleng.lithology import _text_depth_extent

    fig, ax = plt.subplots(figsize=(2.0, 9.0))
    ax.set_ylim(1100, 0)
    ax.set_xlim(0, 1)
    fig.canvas.draw()
    art = ax.text(0.5, 500.0, "North Sea\nGroup, Upper", fontsize=5.6,
                  va="center", ha="center")
    need = _text_depth_extent(art, ax)
    assert need is not None, "measurement unavailable — must not be silent"
    assert 5.0 < need < 500.0, need
    plt.close(fig)


def test_a_thin_band_leads_out_and_a_thick_one_does_not():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    ax = plot_lithology(_band_column(), label_gutter=0.5, title=None)
    leaders = [c for c in ax.get_children() if isinstance(c, Annotation)]
    led = {a.get_text() for a in leaders}
    assert "Holland Marl Member, Upper" in led, "a 41 m band kept its label"
    assert "Z4 Fringe Sandstone Member" not in led, "a 709 m band was moved"
    plt.close(ax.get_figure())


def test_nothing_is_dropped_when_it_does_not_fit():
    """An unlabelled band reads as an UNNAMED one. Suppression is the one
    outcome that must never happen."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation, Text

    ivs = _band_column()
    ax = plot_lithology(ivs, label_gutter=0.5, title=None)
    shown = {t.get_text().replace("\n", " ") for t in ax.get_children()
             if isinstance(t, (Text, Annotation)) and t.get_text()}
    for iv in ivs:
        assert any(iv.name.split(",")[0].split()[0] in s for s in shown), iv.name
    plt.close(ax.get_figure())


def test_no_gutter_keeps_every_label_in_the_band():
    """The previous behaviour stays reachable: a long name overruns."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.text import Annotation

    ax = plot_lithology(_band_column(), title=None)
    assert not [c for c in ax.get_children() if isinstance(c, Annotation)]
    plt.close(ax.get_figure())
