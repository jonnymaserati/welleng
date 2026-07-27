"""Survey-coupled kick-tolerance geometry.

Annular volume lives in the MD domain (capacity is bbl per foot of ALONG-HOLE
length) and pressure lives in the TVD domain; the survey couples them. These
tests pin that coupling:

  * a piece's ``dMD/dTVD`` is the exact mean ``sec(inc)`` -- taken from the
    survey, never from a single representative angle;
  * an interface is PLACED by the survey's closed-form TVD interpolation
    (Sawaryn & Thorogood 2005, SPE-84246-PA), not stepped up to by a march and
    not linearly apportioned within a piece;
  * a section with no MD extent behaves exactly as it did before, so every
    vertical result is unchanged.
"""
import numpy as np
import pytest

import welleng as we
from welleng.architecture import BHA, WellBore
from welleng.kick_tolerance.geometry import (
    annular_capacity,
    sections_from_architecture,
    split_at_tvd,
)
from welleng.kick_tolerance.migration import WellSection

M_TO_FT = 1.0 / 0.3048

CSG_ID_IN = 8.681   # 9-5/8 in, 47 ppf
HOLE_IN = 8.5
COLLAR_OD_IN = 6.5
PIPE_OD_IN = 5.0
SHOE_MD_M = 1800.0


@pytest.fixture
def survey():
    """Vertical to 1000 m, build to 60 deg by 2000 m, hold to 3000 m."""
    return we.survey.Survey(
        md=[0, 1000, 2000, 3000], inc=[0, 0, 60, 60], azi=[0, 0, 30, 30],
    )


@pytest.fixture
def wellbore():
    wb = WellBore('hole', top=0.0, bottom=3000.0, method='top_down')
    wb.add_section(bottom=SHOE_MD_M, id=CSG_ID_IN, coeff_friction_sliding=0.24)
    wb.add_section(bottom=3000.0, id=HOLE_IN, coeff_friction_sliding=0.24)
    return wb


@pytest.fixture
def string():
    st = BHA('string', top=0.0, bottom=3000.0, method='bottom_up')
    st.add_section(length=200.0, od=COLLAR_OD_IN)  # collars, 2800-3000 m
    st.add_section(top=0.0, od=PIPE_OD_IN)
    return st


@pytest.fixture
def sections(wellbore, string, survey):
    return sections_from_architecture(
        wellbore, string, survey, shoe_md=SHOE_MD_M
    )


# ---------------------------------------------------------------------------
# architecture accessors
# ---------------------------------------------------------------------------
def test_string_at_returns_the_spanning_section(wellbore, string):
    assert wellbore.at(1000.0)['id'] == CSG_ID_IN
    assert wellbore.at(2500.0)['id'] == HOLE_IN
    assert string.at(2900.0)['od'] == COLLAR_OD_IN
    assert string.at(1000.0)['od'] == PIPE_OD_IN


def test_breakpoints_are_the_geometry_changes(wellbore, string):
    assert wellbore.breakpoints() == [0.0, SHOE_MD_M, 3000.0]
    assert string.breakpoints() == [0.0, 2800.0, 3000.0]


# ---------------------------------------------------------------------------
# TVD turning points
# ---------------------------------------------------------------------------
def test_turning_point_is_where_the_well_is_exactly_horizontal():
    """Sawaryn Eq. 31: the turning point is where the tangent goes horizontal,
    which is also where TVD is stationary."""
    s = we.survey.Survey(md=[0, 1000, 2000], inc=[0, 60, 120], azi=[0, 0, 0])
    tps = s.tvd_turning_points()

    assert len(tps) == 1
    node = s.interpolate_md(float(tps[0]))
    assert float(node.inc_deg) == pytest.approx(90.0, abs=1e-9)

    # TVD is stationary there, so it is the deepest point of the well: no
    # sampled depth can exceed it (a discrete scan can only fall short).
    tvd_tp = float(node.pos_nev[2])
    scanned = [
        float(s.interpolate_md(md).pos_nev[2])
        for md in np.linspace(1.0, 1999.0, 4000)
    ]
    assert tvd_tp >= max(scanned)
    assert tvd_tp == pytest.approx(max(scanned), abs=1e-4)

    # ... and it is a genuine turning point, not an endpoint of the scan
    for delta in (1.0, 10.0):
        assert float(s.interpolate_md(float(tps[0]) - delta).pos_nev[2]) < tvd_tp
        assert float(s.interpolate_md(float(tps[0]) + delta).pos_nev[2]) < tvd_tp


def test_monotonic_well_has_no_turning_points(survey):
    assert survey.tvd_turning_points().size == 0


def test_a_tvd_above_the_turning_point_is_crossed_twice():
    s = we.survey.Survey(md=[0, 1000, 2000], inc=[0, 60, 120], azi=[0, 0, 0])
    tvd_tp = float(s.interpolate_md(float(s.tvd_turning_points()[0])).pos_nev[2])

    crossings = s.interpolate_tvd(tvd_tp - 5.0)

    assert len(crossings) == 2


# ---------------------------------------------------------------------------
# the builder
# ---------------------------------------------------------------------------
def test_pieces_cut_at_the_union_of_geometry_and_survey(sections):
    """Every hole change, string change, survey station and the shoe."""
    cuts_m = [s.top_md / M_TO_FT for s in sections]
    cuts_m.append(sections[-1].bottom_md / M_TO_FT)

    for expected in (0.0, 1000.0, SHOE_MD_M, 2000.0, 2800.0, 3000.0):
        assert min(abs(c - expected) for c in cuts_m) < 1e-6


def test_extents_sum_to_the_survey(sections, survey):
    assert sum(s.md_extent for s in sections) == pytest.approx(
        3000.0 * M_TO_FT, rel=1e-12
    )
    assert sum(s.bottom_tvd - s.top_tvd for s in sections) == pytest.approx(
        float(survey.pos_nev[-1, 2]) * M_TO_FT, rel=1e-12
    )


def test_hold_section_ratio_is_exactly_sec_inc(sections):
    """In a constant-inclination hold the coupling has a closed form:
    dMD/dTVD == sec(60 deg) == 2. Anything else is a broken coupling."""
    hold = [s for s in sections if s.top_md / M_TO_FT >= 2000.0]
    assert hold

    for s in hold:
        ratio = s.md_extent / (s.bottom_tvd - s.top_tvd)
        assert ratio == pytest.approx(1.0 / np.cos(np.radians(60.0)), rel=1e-9)


def test_capacity_per_tvd_ft_reproduces_the_along_hole_volume(sections):
    for s in sections:
        assert s.capacity_per_tvd_ft * (s.bottom_tvd - s.top_tvd) == pytest.approx(
            s.annular_capacity_bbl_per_ft * s.md_extent, rel=1e-12
        )


def test_capacity_is_the_annulus_not_the_bore(sections):
    below_bha_top = [s for s in sections if s.top_md / M_TO_FT >= 2800.0]
    assert below_bha_top
    for s in below_bha_top:
        assert s.annular_capacity_bbl_per_ft == pytest.approx(
            annular_capacity(HOLE_IN, COLLAR_OD_IN)
        )


def test_open_hole_flag_follows_the_shoe(sections):
    for s in sections:
        below_shoe = s.top_md / M_TO_FT >= SHOE_MD_M - 1e-9
        assert s.is_open_hole is below_shoe


def test_horizontal_interval_is_refused(wellbore, string):
    """A horizontal section holds volume over no TVD at all; the TVD-domain
    engines cannot express it, and must say so rather than divide by zero."""
    s = we.survey.Survey(
        md=[0, 1000, 2000, 3000], inc=[0, 90, 90, 90], azi=[0, 0, 0, 0],
    )
    with pytest.raises(ValueError, match="horizontal"):
        sections_from_architecture(wellbore, string, s, shoe_md=SHOE_MD_M)


# ---------------------------------------------------------------------------
# exact placement
# ---------------------------------------------------------------------------
def test_split_lands_on_the_survey(sections, survey):
    """The cut is placed by the survey's TVD interpolation, so re-evaluating
    the survey at the cut MD returns the requested TVD."""
    target = sections[1]  # inside the build
    tvd = 0.5 * (target.top_tvd + target.bottom_tvd)

    upper, _ = split_at_tvd(target, tvd, survey)

    node = survey.interpolate_md(upper.bottom_md / M_TO_FT)
    assert float(node.pos_nev[2]) * M_TO_FT == pytest.approx(tvd, abs=1e-6)


def test_split_conserves_length_and_volume(sections, survey):
    target = sections[1]
    tvd = 0.5 * (target.top_tvd + target.bottom_tvd)

    upper, lower = split_at_tvd(target, tvd, survey)

    assert upper.md_extent + lower.md_extent == pytest.approx(
        target.md_extent, rel=1e-12
    )
    volume = target.annular_capacity_bbl_per_ft * target.md_extent
    assert sum(
        s.annular_capacity_bbl_per_ft * s.md_extent for s in (upper, lower)
    ) == pytest.approx(volume, rel=1e-12)


def test_repeated_splitting_is_exact(sections, survey):
    target = sections[1]
    upper, lower = split_at_tvd(
        target, 0.5 * (target.top_tvd + target.bottom_tvd), survey
    )
    quarters = (
        list(split_at_tvd(upper, 0.5 * (upper.top_tvd + upper.bottom_tvd), survey))
        + list(split_at_tvd(lower, 0.5 * (lower.top_tvd + lower.bottom_tvd), survey))
    )

    assert sum(s.md_extent for s in quarters) == pytest.approx(
        target.md_extent, rel=1e-12
    )


def test_split_differs_from_apportioning_linearly_in_tvd(sections, survey):
    """Inside a build, MD is NOT linear in TVD. Splitting a piece by its TVD
    fraction misplaces the interface and misstates the gas volume above it --
    this is the error the exact placement removes, so it must be shown to be
    large enough to matter."""
    target = sections[1]

    worst = 0.0
    for frac in (0.1, 0.25, 0.5, 0.75, 0.9):
        tvd = target.top_tvd + frac * (target.bottom_tvd - target.top_tvd)
        upper, _ = split_at_tvd(target, tvd, survey)
        linear_md_extent = frac * target.md_extent
        worst = max(worst, abs(linear_md_extent - upper.md_extent) / upper.md_extent)

    assert worst > 0.02


def test_split_needs_an_md_extent(survey):
    bare = WellSection(1000.0, 2000.0, 0.05, True)
    with pytest.raises(ValueError, match="MD extent"):
        split_at_tvd(bare, 1500.0, survey)


def test_split_outside_the_section_is_refused(sections, survey):
    target = sections[1]
    with pytest.raises(ValueError, match="not inside"):
        split_at_tvd(target, target.bottom_tvd + 100.0, survey)


# ---------------------------------------------------------------------------
# the vertical guarantee
# ---------------------------------------------------------------------------
def test_a_section_without_an_md_extent_is_unchanged():
    """Pre-survey sections must behave exactly as before: dMD == dTVD, so the
    capacity per foot of TVD is the raw along-hole capacity."""
    s = WellSection(1000.0, 2500.0, 0.0489, True)

    assert s.md_extent == 1500.0
    assert s.capacity_per_tvd_ft == s.annular_capacity_bbl_per_ft


def test_a_vertical_survey_gives_the_raw_capacity(wellbore, string):
    s = we.survey.Survey(
        md=[0, 1000, 2000, 3000], inc=[0, 0, 0, 0], azi=[0, 0, 0, 0],
    )
    sections = sections_from_architecture(
        wellbore, string, s, shoe_md=SHOE_MD_M
    )

    for section in sections:
        assert section.capacity_per_tvd_ft == pytest.approx(
            section.annular_capacity_bbl_per_ft, rel=1e-12
        )
        assert section.md_extent == pytest.approx(
            section.bottom_tvd - section.top_tvd, rel=1e-12
        )


# ---------------------------------------------------------------------------
# bounding the partial-span error on a coarse survey
# ---------------------------------------------------------------------------
def test_max_piece_md_bounds_the_partial_span_error(wellbore, string):
    """A piece reports ONE mean sec(inc), so a partial span taken inside it --
    which is what happens where a gas face lands -- carries that mean rather
    than the local value. On a coarsely surveyed well the pieces are long and
    that matters; subdividing recovers it entirely, and costs only a longer
    list because the extra cuts get their TVDs from the same batch
    interpolation as every other cut.

    Measured worst interface misplacement as a fraction of a 1000 ft bubble:
    25.4% at 1000 m stations unbounded, 0.117% with max_piece_md=30 -- which is
    what an actual 30 m survey gives (0.123%).
    """
    md = np.arange(0.0, 3001.0, 1000.0)          # deliberately coarse
    inc = np.clip((md - 1000.0) / 1000.0 * 60.0, 0.0, 60.0)
    coarse = we.survey.Survey(md=md, inc=inc, azi=np.full_like(md, 30.0))

    def worst_error(max_piece):
        sections = sections_from_architecture(
            wellbore, string, coarse, shoe_md=SHOE_MD_M, max_piece_md=max_piece
        )
        worst = 0.0
        for s in sections:
            span = s.bottom_tvd - s.top_tvd
            if span < 1e-6:
                continue
            for frac in np.linspace(0.05, 0.95, 19):
                try:
                    upper, _ = split_at_tvd(s, s.top_tvd + frac * span, coarse)
                except ValueError:
                    continue
                worst = max(worst, abs(frac * s.md_extent - upper.md_extent))
        return worst

    unbounded = worst_error(None)
    bounded = worst_error(30.0)

    assert unbounded > 100.0                      # ~127 ft on a 1000 m survey
    assert bounded < 1.0                          # ~0.6 ft
    assert bounded < unbounded / 100.0

    # and it is monotone in the bound
    assert worst_error(10.0) < bounded < worst_error(100.0)


def test_max_piece_md_must_be_positive(wellbore, string, survey):
    with pytest.raises(ValueError, match="must be positive"):
        sections_from_architecture(wellbore, string, survey,
                                   shoe_md=SHOE_MD_M, max_piece_md=0.0)
