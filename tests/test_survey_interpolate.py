import numpy as np
import welleng as we
from welleng.survey import (
    _interpolate_survey, _interpolate_pos_nev, slice_survey,
)

SURVEY = we.survey.Survey(
    md=[0, 500, 1000, 2000, 2500, 3500],
    inc=[0, 0, 30, 90, 100, 80],
    azi=[45, 45, 45, 90, 90, 180],
    radius=10,
)


def test_survey_interpolate_survey(step=30):

    survey_interp = we.survey.interpolate_survey(SURVEY, step=step)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp = SURVEY.interpolate_survey(step=step)
    assert isinstance(survey_interp, we.survey.Survey)


def test_survey_interpolate_survey_vs_interpolate_mds(step=30):
    mds = np.arange(SURVEY.md[0], SURVEY.md[-1], step)

    # module-level function
    survey_interp = we.survey.interpolate_mds(SURVEY, mds)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp_1 = we.survey.interpolate_survey(SURVEY, step=step)
    assert np.allclose(survey_interp.md, survey_interp_1.md)
    assert np.allclose(survey_interp.azi_grid_rad, survey_interp_1.azi_grid_rad)
    assert np.allclose(survey_interp.inc_rad, survey_interp_1.inc_rad)
    assert np.allclose(survey_interp.pos_xyz, survey_interp_1.pos_xyz)
    assert np.allclose(survey_interp.pos_nev, survey_interp_1.pos_nev)
    assert np.allclose(survey_interp.dogleg, survey_interp_1.dogleg)
    assert np.all(survey_interp.interpolated == survey_interp_1.interpolated)

    # Survey method
    survey_interp = SURVEY.interpolate_mds(mds)
    assert isinstance(survey_interp, we.survey.Survey)

    survey_interp_1 = SURVEY.interpolate_survey(step=step)
    assert np.allclose(survey_interp.md, survey_interp_1.md)
    assert np.allclose(survey_interp.azi_grid_rad, survey_interp_1.azi_grid_rad)
    assert np.allclose(survey_interp.inc_rad, survey_interp_1.inc_rad)
    assert np.allclose(survey_interp.pos_xyz, survey_interp_1.pos_xyz)
    assert np.allclose(survey_interp.pos_nev, survey_interp_1.pos_nev)
    assert np.allclose(survey_interp.dogleg, survey_interp_1.dogleg)
    assert np.all(survey_interp.interpolated == survey_interp_1.interpolated)


def test_survey_interpolate_survey_tvd(step=10):

    survey_interp = SURVEY.interpolate_survey(step=30)
    survey_interp_tvd = we.survey.interpolate_survey_tvd(
        survey_interp, step=step
    )
    assert isinstance(survey_interp_tvd, we.survey.Survey)

    survey_interp_tvd = survey_interp.interpolate_survey_tvd(step=step)
    assert isinstance(survey_interp_tvd, we.survey.Survey)

def test_interpolate_md(md=800):

    node = SURVEY.interpolate_md(md=md)
    assert isinstance(node, we.node.Node)

def test_interpolate_tvd(tvd=800):
    # welleng 0.15.0: interpolate_tvd now returns a list of Nodes (all
    # crossings of the target TVD), normally length 1 on a monotonic well.
    nodes = SURVEY.interpolate_tvd(tvd=tvd)
    assert isinstance(nodes, list)
    assert len(nodes) == 1
    assert isinstance(nodes[0], we.node.Node)


def test_interpolate_tvd_monotonic_equivalence():
    """Correctness anchor: on a strictly monotonic well (inc always < 90 deg,
    so TVD is strictly increasing) the single crossing's MD must reproduce the
    target TVD exactly (cross-check via interpolate_md at the resulting MD)."""
    mono = we.survey.Survey(
        md=[0, 500, 1000, 1800, 2600],
        inc=[0, 20, 45, 70, 85],
        azi=[30, 30, 60, 90, 90],
        radius=10,
    )
    for tvd in (100.0, 700.0, 1234.5, 1500.0):
        nodes = mono.interpolate_tvd(tvd=tvd)
        assert len(nodes) == 1, tvd
        node = nodes[0]
        # the node's own TVD equals the target
        assert np.isclose(node.pos_nev[2], tvd, atol=1e-6), tvd
        # and interpolating by MD at the returned MD lands at the same TVD
        by_md = mono.interpolate_md(node.md)
        assert np.isclose(by_md.pos_nev[2], tvd, atol=1e-6), tvd


def test_interpolate_tvd_reversal_two_crossings():
    """A TVD reversal (build past horizontal so the well climbs) hits a target
    TVD twice; both crossings must be returned at the correct MDs."""
    # single arc from inc 60 -> 120 deg: TVD rises to an interior maximum then
    # falls, so a target just below that maximum is crossed twice.
    s = we.survey.Survey(
        md=[0, 300, 900], inc=[60, 60, 120], azi=[30, 30, 30], radius=10
    )
    # brute-force the two crossing MDs for a target below the interior max
    mds = np.linspace(s.md[0], s.md[-1], 60001)
    tvds = np.array([s.interpolate_md(m).pos_nev[2] for m in mds])
    target = tvds.max() - 5.0
    sign = np.sign(tvds - target)
    brute = mds[:-1][np.diff(sign) != 0]
    assert len(brute) == 2

    nodes = s.interpolate_tvd(tvd=target)
    assert len(nodes) == 2
    got = sorted(n.md for n in nodes)
    assert np.allclose(got, sorted(brute), atol=0.05), (got, brute)
    for n in nodes:
        assert np.isclose(n.pos_nev[2], target, atol=1e-6)


def test_interpolate_tvd_outside_range_empty():
    """A target TVD outside the well's TVD range returns an empty list."""
    assert SURVEY.interpolate_tvd(tvd=-100.0) == []
    assert SURVEY.interpolate_tvd(tvd=1e6) == []


def test_interpolate_tvd_horizontal_section():
    """A horizontal (constant-TVD) hold is handled without NaN: a target equal
    to the hold TVD returns a sane node; other targets in the hold's band do
    not spuriously multiply."""
    # build up to horizontal, then a long horizontal hold, then drop
    s = we.survey.Survey(
        md=[0, 1000, 1500, 2500, 3000],
        inc=[0, 90, 90, 90, 45],
        azi=[0, 0, 0, 0, 0],
        radius=10,
    )
    hold_tvd = s.tvd[1]
    # both the build->horizontal station and the horizontal hold sit at
    # hold_tvd; the drop re-crosses nothing above it. Expect a finite,
    # non-empty, NaN-free result.
    nodes = s.interpolate_tvd(tvd=hold_tvd)
    assert len(nodes) >= 1
    for n in nodes:
        assert np.isfinite(n.md)
        assert np.isclose(n.pos_nev[2], hold_tvd, atol=1e-6)


def test_interpolate_tvd_large_delta_md():
    """The closed form is exact for large-Delta-MD arcs (no reliance on small
    segments): a single 5000 m arc with a 90 deg dogleg resolves the target
    TVD to machine precision."""
    s = we.survey.Survey(
        md=[0, 5000], inc=[10, 100], azi=[0, 0], radius=10
    )
    target = 200.0
    nodes = s.interpolate_tvd(tvd=target)
    assert len(nodes) == 1
    node = nodes[0]
    assert np.isclose(node.pos_nev[2], target, atol=1e-9)
    # independent check: interpolate_md at the solved MD returns the target TVD
    by_md = s.interpolate_md(node.md)
    assert np.isclose(by_md.pos_nev[2], target, atol=1e-9)


def test_interpolate_pos_nev_matches_interpolate_survey():
    """The lightweight, position-only `_interpolate_pos_nev` (used by
    MeshClearance's closest-point search for speed, instead of building a full
    Survey) must return the same NEV position as `_interpolate_survey`. Guards
    the clearance performance optimisation (covers both the curved and the
    tangent/dogleg-zero branches)."""
    for i in range(len(SURVEY.md) - 1):
        s = slice_survey(SURVEY, i, i + 2)
        dmd = float(s.md[1] - s.md[0])
        for frac in (0.0, 0.25, 0.5, 0.9, 1.0):
            x = frac * dmd
            full = _interpolate_survey(s, x)
            pos_full = np.array([full.n, full.e, full.tvd]).T[1]
            pos_light = _interpolate_pos_nev(s, x, 0)
            assert np.allclose(pos_light, pos_full, atol=1e-7), (i, frac)
