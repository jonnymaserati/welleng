"""EDM-IPM normalisation: inline intermediates into self-contained formulas.

``normalise_edm_model`` rewrites a model so its term formulas reference only
base survey variables and numeric constants — no ``edm_intermediates`` section
— for a generic/symbolic formula engine (e.g. the vectorised EOU path) that has
no per-station intermediate mechanism. The transform is an exact algebraic
substitution, so welleng's own engine must give the identical covariance on the
normalised model. Synthetic tests guarantee CI coverage of the inliner; the
Volve tests prove bit-for-bit equivalence on every real gyro/MWD-gyro tool.
"""

import os
import re

import numpy as np
import pytest

import welleng as we
from welleng.errors.edm_ipm import (
    IPMTerm,
    IPMTool,
    ipm_to_error_model,
    normalise_edm_model,
    parse_edm_ipm,
)

VOLVE = os.path.join(os.path.dirname(__file__), "..", "data", "Volve.xml")

_IDENT = re.compile(r"[A-Za-z_]\w*")
# variables + functions the formula engine binds itself (COMPASS/EDM lowercase
# vocabulary + math); anything else left in a normalised formula is a bug.
_BASE_VOCAB = {
    "inc", "azm", "azt", "azi", "tmd", "dmd", "tvd", "gtot", "mtot", "dip",
    "lat", "erot", "mtf",
    "sin", "cos", "tan", "sqrt", "abs", "exp", "log", "maximum", "minimum",
    "arcsin", "arccos", "arctan", "asin", "acos", "atan", "sign", "pi", "e",
}
_FORMULA_FIELDS = (
    "depth_formula", "inclination_formula", "azimuth_formula",
    "north_singularity", "east_singularity",
)


def _identifiers_in_terms(model):
    ids = set()
    for term in model["terms"]:
        for field_name in _FORMULA_FIELDS:
            formula = term.get(field_name)
            if formula:
                ids |= set(_IDENT.findall(formula))
    return ids


# --------------------------------------------------------------------------
# synthetic — always runs (mirrors the real gyro intermediate patterns)
# --------------------------------------------------------------------------

def _synthetic_tool():
    """A gyro-shaped tool: degree constants, a chained intermediate, and a
    hyphenated (mangled) name — the patterns the Volve gyros use."""
    n = lambda name, val, formula, units="-": IPMTerm(  # noqa: E731
        name=name, sequence_no=0, vector_type="n", tie_type="n",
        value=val, units=units, formula=formula,
    )
    err = lambda name, seq, vec, val, formula, units="d": IPMTerm(  # noqa: E731
        name=name, sequence_no=seq, vector_type=vec, tie_type="s",
        value=val, units=units, formula=formula,
    )
    return IPMTool(
        tool_id="synth", name="synthetic gyro",
        terms=[
            # intermediates (tie 'n'): a degree constant, a cant, a chain
            n("ainit", 45.0, "45", units="d"),          # -> 45 deg in rad
            n("cant", 20.0, "20", units="d"),
            n("w_12", 1.0, "sin(inc)"),
            n("w_34", 1.0, "sqrt(1.0001-(w_12)^2)"),    # references w_12
            n("deltad", 1.0, "abs(tmd)"),
            # error terms referencing them
            err("gbxy", 1, "a", 0.5, "cos(ainit)/erot"),
            err("abxy", 2, "i", 0.1, "1/(gtot*cos(inc-cant))"),
            err("mis3", 3, "i", 0.06, "cos(azi)*w_34"),
            err("grw", 4, "a", 0.2, "sqrt(deltad/3600)"),
        ],
    )


def test_normalise_removes_intermediate_section():
    model = ipm_to_error_model(_synthetic_tool())
    assert model["edm_intermediates"]                      # present pre-norm
    norm = normalise_edm_model(model)
    assert "edm_intermediates" not in norm


def test_normalise_leaves_only_base_vocabulary():
    norm = normalise_edm_model(ipm_to_error_model(_synthetic_tool()))
    stray = _identifiers_in_terms(norm) - _BASE_VOCAB
    assert stray == set(), f"non-base identifiers survived inlining: {stray}"


def test_chained_intermediate_fully_resolved():
    # w_34 references w_12; after inlining neither name may remain
    norm = normalise_edm_model(ipm_to_error_model(_synthetic_tool()))
    joined = " ".join(
        norm["terms"][i].get(f, "") or ""
        for i in range(len(norm["terms"])) for f in _FORMULA_FIELDS
    )
    assert "w_12" not in joined and "w_34" not in joined


def test_whole_token_substitution_no_substring_clobber():
    # an intermediate 'd' must not be hit inside 'dmd'/'deltad'; build a case
    tool = IPMTool(
        tool_id="t", name="t",
        terms=[
            IPMTerm("d", 0, "n", "n", 2.0, "-", "3"),
            IPMTerm("hit", 1, "i", "s", 0.1, "d", "dmd*d + tvd"),
        ],
    )
    norm = normalise_edm_model(ipm_to_error_model(tool))
    f = norm["terms"][0]["inclination_formula"]
    assert "dmd" in f and "tvd" in f          # base vars untouched
    assert "((2.0)*(3))" in f                 # the 'd' token inlined


def test_no_intermediates_is_passthrough():
    tool = IPMTool(
        tool_id="mwd", name="mwd",
        terms=[IPMTerm("dbh", 0, "i", "s", 0.1, "d", "1")],
    )
    model = ipm_to_error_model(tool)
    assert model["edm_intermediates"] == []
    norm = normalise_edm_model(model)
    assert "edm_intermediates" not in norm
    assert norm["terms"] == model["terms"]


def _synthetic_survey(model):
    return we.survey.Survey(
        md=[0., 300., 800., 1500., 2500.],
        inc=[0., 8., 25., 55., 85.],
        azi=[0., 40., 60., 110., 160.],
        header=we.survey.SurveyHeader(
            name="t", latitude=58.44, longitude=1.89,
            b_total=50365., dip=72., declination=-2.08,
        ),
        error_model=model,
    )


def test_synthetic_parity_engine_identical():
    tool = _synthetic_tool()
    base = _synthetic_survey(ipm_to_error_model(tool)).err.errors.cov_NEVs
    norm = _synthetic_survey(
        ipm_to_error_model(tool, normalise=True)
    ).err.errors.cov_NEVs
    assert np.max(np.abs(base - norm)) < 1e-12


# --------------------------------------------------------------------------
# Volve — the real proof (skipped if the export is absent)
# --------------------------------------------------------------------------

volve = pytest.mark.skipif(
    not os.path.isfile(VOLVE), reason="Volve.xml not present"
)


@pytest.fixture(scope="module")
def ipm():
    return parse_edm_ipm(VOLVE)


@volve
def test_volve_every_intermediate_tool_fully_inlined(ipm):
    tools_with_interm = [
        tid for tid, t in ipm.tools.items() if t.intermediates
    ]
    assert tools_with_interm                       # the gyros carry them
    for tid in tools_with_interm:
        norm = ipm.error_model(tid, normalise=True)
        assert "edm_intermediates" not in norm
        stray = _identifiers_in_terms(norm) - _BASE_VOCAB
        assert stray == set(), f"{tid}: non-base identifiers survived: {stray}"


@volve
def test_volve_parity_bit_identical_across_tools(ipm):
    md = np.arange(0, 2000 + 30, 30.0)
    inc = np.clip((md - 300) / 1000 * 60, 0, 60.0)
    inc[md < 300] = 0.0
    azi = np.where(inc == 0, 0.0, 135.0)
    header = we.survey.SurveyHeader(
        name="t", latitude=58.44, longitude=1.89,
        b_total=50365., dip=72., declination=-2.08,
    )

    def cov(model):
        return we.survey.Survey(
            md=md, inc=inc, azi=azi, header=header, error_model=model
        ).err.errors.cov_NEVs

    worst = 0.0
    for tid, t in ipm.tools.items():
        if not t.intermediates:
            continue
        base = cov(ipm.error_model(tid))
        norm = cov(ipm.error_model(tid, normalise=True))
        worst = max(worst, float(np.max(np.abs(base - norm))))
    assert worst < 1e-9, f"worst normalised-vs-original cov diff {worst:.2e}"
