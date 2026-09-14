"""A survey does not necessarily start at surface.

Reported by a consumer against a real sidetrack (P11-A-03-SIDETRACK1, whose
definitive survey begins at its 1841 m tie-in). Two defects, both silent, and
the shallow one is the dangerous shape -- ``np.interp`` clamps, so an MD above
the first station came back as the first station's depth: a plausible number,
in range for the well, wrong. It drew a well 200 m deep with three shoes
stacked at the top of the panel, and was caught only because the picture was
obviously wrong.

The second defect is the one the report did not separate out: without a tie-in
TVD the survey is built from zero, so EVERY TVD is measured from the first
station rather than the datum -- in range as much as out.
"""
import warnings

import pytest

from welleng.schematic import depth as depth_mod
from welleng.schematic.depth import DepthResolver
from welleng.schematic.models import Casing, SurveyRef, Wellbore


SIDETRACK = dict(                      # starts at a tie-in, not at surface
    md=[1841.0, 2000.0, 2300.0, 2600.0],
    inc=[35.0, 40.0, 55.0, 70.0],
    azi=[120.0, 122.0, 125.0, 130.0],
)


def _resolver(**kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DepthResolver(SurveyRef(**{**SIDETRACK, **kw}))


# --- coverage is now askable ---------------------------------------------- #
def test_the_resolver_says_what_it_covers():
    """`md_min` is the point of the report: a caller could not test before
    asking, so it had no way to know the answer was fabricated."""
    r = _resolver()
    assert r.md_min == 1841.0
    assert r.md_max == 2600.0
    assert not r.covers(161.2)
    assert r.covers(2296.0)
    assert not r.covers(9999.0)


def test_the_grid_does_not_overshoot_the_last_station():
    """`arange(lo, hi + step, step)` overshoots whenever the span is not a
    whole number of steps, and np.interp clamps -- so the tail was fabricated
    hold section and md_max overstated the coverage."""
    r = _resolver()
    assert r.md_max == 2600.0, "grid ran past the last station"
    assert r.md[-1] == pytest.approx(2600.0)


# --- an out-of-range ask is reported -------------------------------------- #
@pytest.mark.parametrize("md", [161.2, 789.6, 1586.0, 9999.0])
def test_an_out_of_range_depth_is_not_returned_silently(md):
    r = _resolver()
    with pytest.warns(UserWarning, match="falls outside the survey"):
        r.depth(md, "TVD")


def test_the_warning_states_the_survey_range():
    """A refusal that does not name what it would have needed moves the
    debugging cost to whoever has least context."""
    r = _resolver()
    with pytest.warns(UserWarning) as rec:
        r.depth(161.2, "TVD")
    msg = str(rec[0].message)
    assert "1841.0" in msg and "2600.0" in msg
    assert "compose the parent wellbore" in msg


def test_every_interpolator_checks_not_just_tvd():
    """MD mode clamps exactly as TVD mode does -- it was the same np.interp."""
    r = _resolver()
    for call in (lambda: r.depth(161.2, "MD"),
                 lambda: r.tvd_at(161.2),
                 lambda: r.pos(161.2),
                 lambda: r.vs(161.2, 120.0)):
        with pytest.warns(UserWarning, match="falls outside"):
            call()


def test_in_range_asks_stay_quiet():
    """A warning that fires on ordinary use gets filtered, and then it is not
    a warning any more."""
    r = _resolver()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r.depth(2296.0, "TVD")
        r.pos(2000.0)


def test_strict_range_refuses():
    r = _resolver()
    depth_mod.STRICT_RANGE = True
    try:
        with pytest.raises(ValueError, match="falls outside the survey"):
            r.depth(161.2, "TVD")
    finally:
        depth_mod.STRICT_RANGE = False


# --- the tie-in ------------------------------------------------------------ #
def test_a_survey_starting_below_surface_reports_its_missing_tie_in():
    with pytest.warns(UserWarning, match="not at surface"):
        DepthResolver(SurveyRef(**SIDETRACK))


def test_without_a_tie_in_every_tvd_is_run_relative():
    """The half the report read as an out-of-range artefact. It is not: the
    survey is built from zero, so in-range answers are short by the tie-in
    TVD too."""
    r = _resolver()
    assert r.tvd[0] == pytest.approx(0.0)


def test_the_tie_in_references_the_survey_to_the_datum():
    r = _resolver(tie_in_tvd=1520.0)
    assert r.tvd[0] == pytest.approx(1520.0)
    assert float(r.depth(2296.0, "TVD")) > 1520.0


def test_a_surface_survey_needs_no_tie_in():
    """The common case must not acquire a warning -- md[0] == 0 is not a
    missing tie-in, it is a survey that starts at surface."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = DepthResolver(SurveyRef(md=[0.0, 1000.0, 2000.0],
                                    inc=[0.0, 0.0, 0.0],
                                    azi=[0.0, 0.0, 0.0]))
    assert r.tvd[0] == pytest.approx(0.0)


def test_tie_in_is_not_defaulted_to_zero():
    """None means NOT RECORDED. Zero is a claim that the first station is at
    the datum, which for a sidetrack is false by construction."""
    assert SurveyRef(**SIDETRACK).tie_in_tvd is None


# --- an unplaceable string is omitted, and said so ------------------------- #
def _bore(**kw):
    sr = SurveyRef(md=[0.0, 1000.0, 2000.0], inc=[0.0] * 3, azi=[0.0] * 3)
    return Wellbore(survey=sr, **kw)


def test_a_string_with_no_shoe_depth_does_not_lose_the_schematic():
    """A string itemised joint by joint with no depth stated anywhere was
    still RUN. Requiring shoe_md lost the whole drawing to one such string."""
    with pytest.warns(UserWarning, match="no shoe_md"):
        bore = _bore(casings=[
            Casing(name="13-3/8in", od_in=13.375, id_in=12.415, shoe_md=800.0),
            Casing(name="9-5/8in", od_in=9.625, id_in=8.681),
        ])
    assert [c.name for c in bore.drawable_casings] == ["13-3/8in"]
    assert [c.name for c in bore.unplaceable_casings] == ["9-5/8in"]


def test_the_omission_is_announced():
    """An omitted string is a SILENT omission: the schematic still looks
    complete, and a barrier drawing quietly one string short is exactly the
    failure this model exists to avoid."""
    with pytest.warns(UserWarning) as rec:
        _bore(casings=[Casing(name="junk in hole", od_in=7.0, id_in=6.276)])
    msg = " ".join(str(w.message) for w in rec)
    assert "junk in hole" in msg and "OMITTED" in msg


def test_a_fully_placed_bore_stays_quiet():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        bore = _bore(casings=[
            Casing(name="13-3/8in", od_in=13.375, id_in=12.415, shoe_md=800.0)])
    assert bore.unplaceable_casings == []


def test_an_unplaceable_string_reports_no_profile():
    """It must not report (top_md, None) or (top_md, 0.0) -- a zero-depth
    interval is the fabrication the omission exists to avoid."""
    c = Casing(name="x", od_in=9.625, id_in=8.681)
    assert c.profile() == []
