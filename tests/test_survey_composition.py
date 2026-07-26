"""Tests for welleng.composition.SurveyComposition.

Ties a wellbore's ordered survey sections (legs) into one continuous survey
with per-component-correct covariance carry across tool changes.
"""
import json

import numpy as np
import pytest

from welleng.survey import Survey, SurveyHeader
from welleng.composition import SurveyComposition, SurveySection
from welleng.conditioning import combine_covariances

EM = "ISCWSA MWD Rev5.11"
WELL_DATA = "tests/test_data/clearance_iscwsa_well_data.json"


def _header():
    sh = SurveyHeader(b_total=50_000., dip=72., declination=-2.)
    sh.azi_reference = "grid"
    sh.b_total = 50000.0
    sh.dip = 70.0
    sh.latitude = 60.0
    return sh


def _geometry():
    md = np.arange(0.0, 2001.0, 100.0)
    inc = np.linspace(0.0, 90.0, len(md))
    azi = np.full(len(md), 45.0)
    return md, inc, azi


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #
def test_same_tool_sections_form_one_group():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T1"),
    ])
    assert len(comp._groups) == 1


def test_tool_change_forms_two_groups():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2"),
    ])
    assert len(comp._groups) == 2


# --------------------------------------------------------------------------- #
# GATE 1: same-tool continuation is IDENTICAL to one Survey
# --------------------------------------------------------------------------- #
def test_same_tool_equals_single_survey():
    md, inc, azi = _geometry()
    sh = _header()
    full = Survey(md=md, inc=inc, azi=azi, header=sh, error_model=EM)
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T1"),
    ]).survey()

    assert np.allclose(comp.md, full.md)
    # the key correctness test: systematic accumulates correlated, not restarted
    assert np.allclose(comp.cov_nev, full.cov_nev, atol=1e-9)
    assert np.allclose(comp.cov_nev_global, full.cov_nev_global, atol=1e-9)
    assert np.allclose(comp.cov_nev_systematic, full.cov_nev_systematic,
                       atol=1e-9)
    assert np.allclose(comp.cov_nev_random, full.cov_nev_random, atol=1e-9)


def test_three_same_tool_sections_equal_single_survey():
    md, inc, azi = _geometry()
    sh = _header()
    full = Survey(md=md, inc=inc, azi=azi, header=sh, error_model=EM)
    a, b = 5, 12
    comp = SurveyComposition([
        SurveySection(md=md[:a + 1], inc=inc[:a + 1], azi=azi[:a + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[a:b + 1], inc=inc[a:b + 1], azi=azi[a:b + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[b:], inc=inc[b:], azi=azi[b:],
                      header=sh, error_model=EM, tool_id="T1"),
    ]).survey()
    assert np.allclose(comp.cov_nev, full.cov_nev, atol=1e-9)


# --------------------------------------------------------------------------- #
# GATE 2: a tool change carries the position covariance (does not reset)
# --------------------------------------------------------------------------- #
def test_tool_change_carries_covariance():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2"),
    ]).survey()

    # covariance at the tie is non-zero (carried) and grows downstream
    assert np.trace(comp.cov_nev[i]) > 0.0
    assert np.trace(comp.cov_nev[-1]) > np.trace(comp.cov_nev[i])
    # component buckets always sum to the total
    assert np.allclose(
        comp.cov_nev,
        comp.cov_nev_global + comp.cov_nev_systematic + comp.cov_nev_random,
    )


def test_new_tool_systematic_is_independent():
    """A tool change restarts the (independent) systematic, so the end
    systematic is smaller than if one tool accumulated it correlated."""
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    same = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T1"),
    ]).survey()
    changed = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2"),
    ]).survey()
    # random is identical (always independent); systematic is smaller when the
    # tool changes (independent realisation, not one correlated accumulation).
    assert np.allclose(same.cov_nev_random, changed.cov_nev_random, atol=1e-9)
    assert np.trace(changed.cov_nev_systematic[-1]) < np.trace(
        same.cov_nev_systematic[-1]
    )


# --------------------------------------------------------------------------- #
# GATE 3: share_mode overrides
# --------------------------------------------------------------------------- #
def _reference_and_composed(tie_share_mode):
    md, inc, azi = _geometry()
    sh = _header()
    i = 5
    reference = Survey(md=md, inc=inc, azi=azi, header=sh, error_model=EM)
    composed = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2",
                      share_mode=tie_share_mode),
    ]).survey()
    return reference, composed


def _relative_sigma(a, b, combine_mode):
    return combine_covariances(
        a.cov_nev, b.cov_nev,
        cov_global_a=a.cov_nev_global, cov_global_b=b.cov_nev_global,
        cov_systematic_a=a.cov_nev_systematic,
        cov_systematic_b=b.cov_nev_systematic,
        share_mode=combine_mode,
    ).sigma_combined


def test_share_mode_tie_all_independent_larger_than_globals_shared():
    """An all_independent tie leaves its post-tie global in the non-cancelling
    bucket, so the difference against a well sharing the campaign geomag is
    LARGER than for a globals_shared tie (globals don't cancel)."""
    ref, gs = _reference_and_composed("globals_shared")
    _, ai = _reference_and_composed("all_independent")
    rel_gs = _relative_sigma(ref, gs, "globals_shared")[-1]
    rel_ai = _relative_sigma(ref, ai, "globals_shared")[-1]
    assert rel_ai > rel_gs
    # and the cancellable global bucket is (much) smaller for the independent tie
    assert np.trace(ai.cov_nev_global[-1]) < np.trace(gs.cov_nev_global[-1])


def test_combine_share_mode_ordering_on_composed_buckets():
    """The composed per-component buckets drive combine_covariances: more
    sharing -> more cancellation -> smaller combined covariance."""
    ref, comp = _reference_and_composed("globals_shared")
    rel_indep = _relative_sigma(ref, comp, "all_independent")[-1]
    rel_glob = _relative_sigma(ref, comp, "globals_shared")[-1]
    rel_both = _relative_sigma(ref, comp, "globals_and_systematic_shared")[-1]
    assert rel_indep > rel_glob > rel_both


# --------------------------------------------------------------------------- #
# auto share-mode from context keys
# --------------------------------------------------------------------------- #
def test_auto_share_mode_far_apart_dates_independent():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1",
                      survey_date="2001-01-01"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2",
                      survey_date="2020-06-01"),  # ~19 years later
    ])
    assert comp._groups[1].share_mode == "all_independent"


def test_auto_share_mode_close_dates_globals_shared():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1",
                      survey_date="2020-01-01"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2",
                      survey_date="2020-03-01"),
    ])
    assert comp._groups[1].share_mode == "globals_shared"


def test_auto_share_mode_different_geomag_model_independent():
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1",
                      geomag_model="BGGM2015"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2",
                      geomag_model="IFR3"),
    ])
    assert comp._groups[1].share_mode == "all_independent"


# --------------------------------------------------------------------------- #
# GATE 4: ISCWSA well-10 side-track tie-on to the Reference well
# --------------------------------------------------------------------------- #
def _load_wells():
    wells = json.load(open(WELL_DATA))["wells"]

    def mk_header(h):
        sh = SurveyHeader(b_total=50_000., dip=72., declination=-2.)
        for k, v in h.items():
            if k != "name":
                setattr(sh, k, v)
        return sh
    return wells, mk_header


def test_sidetrack_tie_carries_reference_covariance():
    wells, mk_header = _load_wells()
    ref, w10 = wells["Reference well"], wells["10 - well"]
    header = mk_header(ref["header"])
    tie_idx = ref["MD"].index(900.0)  # side-track ties on at MD 900

    # compose: reference trunk (to MD 900) + side-track (well 10)
    composed = SurveyComposition([
        SurveySection(md=ref["MD"][:tie_idx + 1], inc=ref["IncDeg"][:tie_idx + 1],
                      azi=ref["AziDeg"][:tie_idx + 1], header=header,
                      error_model=EM, tool_id="trunk"),
        SurveySection(md=w10["MD"], inc=w10["IncDeg"], azi=w10["AziDeg"],
                      header=header, error_model=EM, tool_id="sidetrack"),
    ]).survey()

    # a single unified survey over the SAME stitched geometry (one tool) is the
    # reference for what the tie station's covariance MUST be.
    umd = np.concatenate([ref["MD"][:tie_idx + 1], w10["MD"][1:]])
    uinc = np.concatenate([ref["IncDeg"][:tie_idx + 1], w10["IncDeg"][1:]])
    uazi = np.concatenate([ref["AziDeg"][:tie_idx + 1], w10["AziDeg"][1:]])
    unified = Survey(md=umd, inc=uinc, azi=uazi, header=header, error_model=EM)

    # the tie station carries the parent's covariance exactly (not reset)
    assert np.allclose(composed.cov_nev[tie_idx], unified.cov_nev[tie_idx],
                       atol=1e-9)
    assert np.trace(composed.cov_nev[tie_idx]) > 0.0
    # covariance grows down the side-track
    assert np.trace(composed.cov_nev[-1]) > np.trace(composed.cov_nev[tie_idx])


def test_sidetrack_relative_to_parent_zero_at_tie():
    """The side-track inherits the parent's position error at the tie, so its
    uncertainty RELATIVE to the tie point is zero there and grows downstream."""
    wells, mk_header = _load_wells()
    ref, w10 = wells["Reference well"], wells["10 - well"]
    header = mk_header(ref["header"])
    tie_idx = ref["MD"].index(900.0)

    composed = SurveyComposition([
        SurveySection(md=ref["MD"][:tie_idx + 1], inc=ref["IncDeg"][:tie_idx + 1],
                      azi=ref["AziDeg"][:tie_idx + 1], header=header,
                      error_model=EM, tool_id="trunk"),
        SurveySection(md=w10["MD"], inc=w10["IncDeg"], azi=w10["AziDeg"],
                      header=header, error_model=EM, tool_id="sidetrack"),
    ]).survey()

    carried = composed.cov_nev[tie_idx]
    # side-track uncertainty relative to the tie carry
    rel = composed.cov_nev - carried[None, :, :]
    assert np.allclose(rel[tie_idx], 0.0, atol=1e-9)          # zero at the tie
    assert np.trace(rel[-1]) > 0.0                            # grows below it


# --------------------------------------------------------------------------- #
# misc API
# --------------------------------------------------------------------------- #
def test_tie_mismatch_raises():
    md, inc, azi = _geometry()
    sh = _header()
    with pytest.raises(ValueError):
        SurveyComposition([
            SurveySection(md=md[:7], inc=inc[:7], azi=azi[:7],
                          header=sh, error_model=EM, tool_id="T1"),
            SurveySection(md=md[9:], inc=inc[9:], azi=azi[9:],  # gap: no tie
                          header=sh, error_model=EM, tool_id="T2"),
        ])


def test_survey_is_cached():
    md, inc, azi = _geometry()
    sh = _header()
    comp = SurveyComposition([
        SurveySection(md=md, inc=inc, azi=azi, header=sh, error_model=EM),
    ])
    assert comp.survey() is comp.survey()


def test_section_from_survey_object():
    md, inc, azi = _geometry()
    sh = _header()
    s = Survey(md=md, inc=inc, azi=azi, header=sh, error_model=EM)
    comp = SurveyComposition([SurveySection(survey=s)]).survey()
    single = Survey(md=md, inc=inc, azi=azi, header=sh, error_model=EM)
    assert np.allclose(comp.cov_nev, single.cov_nev, atol=1e-9)


def test_chained_global_component_is_telescoping_exact():
    """Williamson SPE-67616-PA Eq. A-14: a global source chained across legs
    must equal the SAME source evaluated over one continuous survey — the
    two-sided station differentials telescope, so a leg-wise evaluation that
    zeroes its tie station over-counts (historically ~x2 per leg for
    depth-scale weights). Split one synthetic tool into three legs with
    distinct model objects (forcing the per-term chained path) and require
    exact agreement with the continuous survey.
    """
    import warnings

    import numpy as np

    import welleng as we
    from welleng.composition import SurveyComposition, SurveySection

    def toy(name):
        return {
            "metadata": {"model_id": name, "short_name": name,
                         "tool_type": "MWD"},
            "parameters": {"inc_min": 0, "inc_max": 180},
            "edm_intermediates": [],
            "terms": [
                {"name": "dsf", "value": 6e-4, "units": "-",
                 "propagation_mode": "Global", "depth_formula": "tmd",
                 "inclination_formula": "0", "azimuth_formula": "0",
                 "north_singularity": None, "east_singularity": None,
                 "vertical_singularity": None},
                {"name": "dstb", "value": 2.5e-7, "units": "1/m",
                 "propagation_mode": "Global", "depth_formula": "tmd*tvd",
                 "inclination_formula": "0", "azimuth_formula": "0",
                 "north_singularity": None, "east_singularity": None,
                 "vertical_singularity": None},
            ],
        }

    sh = we.survey.SurveyHeader(b_total=50000., dip=72., declination=-2.)
    md = np.arange(0., 3001., 100.)
    inc = np.full_like(md, 60.)
    inc[0] = 0
    azi = np.full_like(md, 90.)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = we.survey.Survey(
            md=md, inc=inc, azi=azi, header=sh, error_model=toy("T")
        ).err.errors.cov_NEVs
        com = SurveyComposition(sections=[
            SurveySection(md=md[:11], inc=inc[:11], azi=azi[:11], header=sh,
                          error_model=toy("A"), tool_id="a"),
            SurveySection(md=md[10:21], inc=inc[10:21], azi=azi[10:21],
                          header=sh, error_model=toy("B"), tool_id="b",
                          share_mode="globals_shared"),
            SurveySection(md=md[20:], inc=inc[20:], azi=azi[20:], header=sh,
                          error_model=toy("C"), tool_id="c",
                          share_mode="globals_shared"),
        ]).survey().cov_nev
    assert np.allclose(com, ref, atol=1e-10)


def test_composition_does_not_re_propagate_per_covariance_component():
    """One propagation per (run, depth datum) -- not one per component.

    `_run_component`'s `attr` selects which covariance to READ off a build; it
    does not change the build. The only thing that does is the severed-run
    `_tmd_datum` override. So the four components must share propagations, and
    the result must be BIT-IDENTICAL to computing each separately -- this is an
    efficiency fix on an MC-gated path, so "close" is not good enough.

    a consumer's profile: `SurveyComposition` was 93% of their programme
    setup, running 8 full ErrorModel propagations for a 2-section compose and
    discarding three quarters of each result.
    """
    import welleng.error as error_module

    md, inc, azi = _geometry()
    sh = _header()
    i = 6

    def build(model_a, model_b):
        return SurveyComposition([
            SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                          header=sh, error_model=model_a, tool_id="T1"),
            SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                          header=sh, error_model=model_b, tool_id="T2"),
        ])

    def count(model_a, model_b):
        n = [0]
        original = error_module.ErrorModel.__init__

        def counted(self, *args, **kwargs):
            n[0] += 1
            return original(self, *args, **kwargs)

        error_module.ErrorModel.__init__ = counted
        try:
            survey = build(model_a, model_b).survey()
        finally:
            error_module.ErrorModel.__init__ = original
        return n[0], survey

    # single-model (shared realisation) and multi-model (per-term chaining)
    for a, b in ((EM, EM), ("ISCWSA MWD Rev4", EM)):
        n, _ = count(a, b)
        # 2 groups x 4 components = 8 if nothing is shared. The floor is a
        # handful; assert well below the un-shared count so a regression that
        # re-introduces per-component propagation fails here.
        assert n <= 6, f"{a}/{b}: {n} propagations, components are not shared"


def test_run_cache_does_not_leak_between_calls():
    """The cache is scoped to one `survey()` call, so a mutated composition can
    never read a stale run."""
    md, inc, azi = _geometry()
    sh = _header()
    i = 6
    comp = SurveyComposition([
        SurveySection(md=md[:i + 1], inc=inc[:i + 1], azi=azi[:i + 1],
                      header=sh, error_model=EM, tool_id="T1"),
        SurveySection(md=md[i:], inc=inc[i:], azi=azi[i:],
                      header=sh, error_model=EM, tool_id="T2"),
    ])
    first = comp.survey()
    assert comp._run_cache == {}
    second = comp.survey()
    assert np.array_equal(first.cov_nev, second.cov_nev)
