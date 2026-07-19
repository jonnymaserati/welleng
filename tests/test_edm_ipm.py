"""EDM IPM import: parse the COMPASS error-model layer and run it.

Uses the public Volve EDM export (``data/Volve.xml``, Equinor CC-BY) — the
only real EDM in the repo and the validation target: its ``DP_TOOL_TERM``
table carries COMPASS's actual instrument performance models, so welleng can
compute uncertainty with the very tool models the operator ran.
"""

import os

import numpy as np
import pytest

import welleng as we
from welleng.errors.edm_ipm import (
    EDMIPM,
    IPMTerm,
    IPMTool,
    ipm_to_error_model,
    parse_edm_ipm,
)
from welleng.exchange.edm_stream import ToolKind, classify_tool

VOLVE = os.path.join(os.path.dirname(__file__), "..", "data", "Volve.xml")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(VOLVE), reason="Volve.xml not present"
)


@pytest.fixture(scope="module")
def ipm() -> EDMIPM:
    return parse_edm_ipm(VOLVE)


MAG_REF = dict(b_total=50365., dip=72., declination=-2.08)


def _survey(model):
    return we.survey.Survey(
        md=[0., 500., 1000., 2000., 3000.],
        inc=[0., 10., 30., 60., 88.],
        azi=[45., 45., 60., 90., 120.],
        header=we.survey.SurveyHeader(
            name="t", latitude=58.4416, longitude=1.8875, **MAG_REF
        ),
        error_model=model,
    )


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

def test_parse_volve_ipm_layer(ipm):
    assert len(ipm.tools) == 30
    assert sum(len(t.terms) for t in ipm.tools.values()) == 836
    assert len(ipm.run_tool_map) > 100      # every raw run links to its tool
    assert len(ipm.magnetics) == 54
    # intermediates (tie 'n') are exactly the vector 'n' rows
    for t in ipm.tools.values():
        for term in t.intermediates:
            assert term.vector_type == "n"
        for term in t.error_terms:
            assert term.vector_type in ("a", "e", "i", "l")
            assert term.tie_type in ("r", "s", "w", "g")


def test_tool_lookup_by_id_and_name(ipm):
    by_id = ipm.tool("e3rtk")
    by_name = ipm.tool("Magnetic, std, non-mag")
    assert by_id is by_name
    assert by_id.kind is ToolKind.MWD


def test_terms_ordered_by_sequence(ipm):
    for t in ipm.tools.values():
        seqs = [term.sequence_no for term in t.terms]
        assert seqs == sorted(seqs)


def test_classify_name_wins_over_description():
    # Volve's MWD tools carry descriptions like "Magnetic Tools (MWD, EMS)
    # without gyro-verification" — the word "gyro" there must not flip a
    # magnetic tool to a gyro.
    assert classify_tool(
        "Magn, IFR, mag-corr", "Magnetic Tools without gyro-verification"
    ) is ToolKind.MWD
    assert classify_tool("MWD gyro, GyroTrak") is ToolKind.GYRO


# --------------------------------------------------------------------------
# conversion
# --------------------------------------------------------------------------

def test_hyphenated_intermediate_names_mangled():
    tool = IPMTool(tool_id="x", name="x", terms=[
        IPMTerm(name="ngxy-b1", sequence_no=1, vector_type="n", tie_type="n",
                value=2.0, units="-", formula="1.0"),
        IPMTerm(name="azt1", sequence_no=2, vector_type="a", tie_type="s",
                value=1.0, units="d", formula="ngxy-b1*sin(inc)"),
    ])
    model = ipm_to_error_model(tool)
    assert model["edm_intermediates"][0]["name"] == "ngxy_b1"
    assert model["terms"][0]["azimuth_formula"] == "(ngxy_b1*sin(inc))"


def test_lateral_maps_to_azimuth_over_sin_inc():
    tool = IPMTool(tool_id="x", name="x", terms=[
        IPMTerm(name="lat1", sequence_no=1, vector_type="l", tie_type="s",
                value=1.0, units="d", formula="cos(inc)"),
    ])
    term = ipm_to_error_model(tool)["terms"][0]
    assert term["inclination_formula"] == "0"
    assert "maximum(sin(inc), 1e-9)" in term["azimuth_formula"]


def test_units_and_propagation_mapping():
    tool = IPMTool(tool_id="x", name="x", terms=[
        IPMTerm(name=n, sequence_no=i, vector_type="i", tie_type=tie,
                value=1.0, units=u, formula="1")
        for i, (n, tie, u) in enumerate([
            ("t1", "r", "d"), ("t2", "s", "m"), ("t3", "w", "nt"),
            ("t4", "g", "dnt"), ("t5", "s", "im"), ("t6", "s", "-"),
        ])
    ])
    model = ipm_to_error_model(tool)
    assert [t["propagation_mode"] for t in model["terms"]] == [
        "Random", "Systematic", "Well", "Global", "Systematic", "Systematic"
    ]
    assert [t["units"] for t in model["terms"]] == [
        "deg", "m", "nT", "deg/nT", "1/m", "-"
    ]


# --------------------------------------------------------------------------
# engine integration
# --------------------------------------------------------------------------

def test_volve_mwd_model_runs_clean(ipm, recwarn):
    model = ipm.error_model("Magnetic, std, non-mag")
    s = _survey(model)
    cov = s.err.errors.cov_NEVs
    # every formula + intermediate evaluated — no "contributes zero" warnings
    assert not [w for w in recwarn if "could not be evaluated" in str(w.message)]
    assert np.all(np.isfinite(cov))
    sigma_td = np.sqrt(cov[-1].diagonal())
    assert np.all(sigma_td > 0.1)           # real uncertainty accumulated
    assert np.all(sigma_td < 100.0)         # and a sane magnitude
    # propagation buckets populated per the tie types present in the IPM
    assert s.err.errors.cov_NEVs_systematic[-1, 0, 0] > 0
    assert s.err.errors.cov_NEVs_global[-1, 0, 0] > 0


def test_all_volve_tools_run(ipm):
    """Every tool with an IPM must run the engine without unbound variables."""
    import warnings as w

    failures = {}
    for tool in ipm.tools.values():
        if not tool.error_terms:
            continue
        model = ipm_to_error_model(tool)
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            s = _survey(model)
            _ = s.err.errors.cov_NEVs
        bad = [str(c.message) for c in caught
               if "could not be evaluated" in str(c.message)]
        if bad:
            failures[tool.name] = bad[0][:100]
    assert not failures, failures


def test_dict_models_respect_the_mag_reference_gate(ipm):
    """A magnetic IPM dict refuses a default-reference header (the 0.21 gate);
    a gyro IPM dict is exempt — classification from metadata.tool_type."""
    bare = we.survey.SurveyHeader()      # no mag data, no location
    mwd = ipm.error_model("Magnetic, std, non-mag")
    with pytest.raises(ValueError, match="magnetic model"):
        we.survey.Survey(
            md=[0., 500., 1000.], inc=[0., 30., 60.], azi=[45.] * 3,
            header=bare, error_model=mwd,
        ).err
    gyro = ipm.error_model("Wellbore Surveyor, stat")
    we.survey.Survey(     # bare header: no lookup fires, gyro is exempt
        md=[0., 500., 1000.], inc=[0., 30., 60.], azi=[45.] * 3,
        header=we.survey.SurveyHeader(), error_model=gyro,
    ).err


def test_dict_model_equivalent_via_error_model_class(ipm):
    model = ipm.error_model("e3rtk")
    s = _survey(model)
    em = we.error.ErrorModel(s, error_model=model)
    assert np.allclose(
        em.errors.cov_NEVs, s.err.errors.cov_NEVs, atol=1e-12
    )


def test_ipm_file_bridge_runs_engine():
    """A parsed .IPM FILE model runs the same conversion/engine path as an
    EDM-embedded one (welleng.exchange.ipm -> tool_from_ipm_model).

    Toolface-EXPLICIT terms (``tfo`` in the formula) are a known gap — the
    welleng weights are toolface-integrated — so such a term warns and
    contributes zero; the toolface-free terms evaluate.
    """
    import textwrap
    import warnings as w

    from welleng.errors.edm_ipm import tool_from_ipm_model
    from welleng.exchange.ipm import loads_ipm

    sample = textwrap.dedent(
        """\
        #Tool Name  :TEST_MWD
        #ShortName  :TEST
        #Description:synthetic test model
        #Name\tVector\tTie-On\tUnit\tValue\tFormula
        abx\ti\ts\t-\t0.004\t(-cos(inc)*sin(tfo))/gtot
        abz\ti\ts\t-\t0.004\t(-sin(inc))/gtot
        mbz\ta\ts\tnt\t70\t(-sin(inc)*sin(azm))/(mtot*cos(dip))
        dref\te\tr\t-\t0.35\t1.0
        dsf\te\ts\t-\t0.00024\ttmd
        decg\ta\tg\tdeg\t0.36\t1.0
        """
    )
    tool = tool_from_ipm_model(loads_ipm(sample))
    assert tool.name == "TEST"
    model = ipm_to_error_model(tool)
    with w.catch_warnings(record=True) as caught:
        w.simplefilter("always")
        s = _survey(model)
        cov = s.err.errors.cov_NEVs
    tfo_warns = [c for c in caught if "abx" in str(c.message)]
    assert tfo_warns, "toolface-explicit term should warn (known gap)"
    assert np.all(np.isfinite(cov))
    assert cov[-1, 2, 2] > 0        # depth terms accumulated fine


# --------------------------------------------------------------------------
# validation: reproduce COMPASS's own covariance (the whole point)
# --------------------------------------------------------------------------

def test_f12_plan_reproduces_compass_covariance(ipm):
    """F-12 PLAN definitive: single tool ('Magnetic, std, mag-corr') over the
    whole path, with COMPASS's own per-station 1-sigma covariances stored in
    the export — ground truth we did not compute.

    Built with the ACTUAL tool IPM (DP_TOOL_TERM) + the as-run geomagnetic
    reference (DP_MAGNETIC), welleng matches COMPASS:

    - sigma_N and sigma_E within ±1% (or 2 cm absolute, whichever is
      larger — the first few stations carry centimetre sigmas where the
      start-station conventions differ) at every station;
    - sigma_V within -8%..+1% — welleng slightly UNDER, a known residual
      with the shape const(0.12 m) + ~3.1e-4*TVD, i.e. a well-level
      vertical reference term COMPASS carries outside DP_TOOL_TERM (no
      depth-uncertainty table exists in the export to source it from).

    COMPASS station frame: x=East, y=North, z=Vertical; feet; 1-sigma.
    """
    from welleng.exchange.edm_stream import EDMReader, FEET_TO_METERS

    r = EDMReader(VOLVE)
    hid = "AtCw9"
    h = r.headers[hid]
    stations = sorted(r._def_stations[hid], key=lambda s: s["md"])
    mag = [m for m in ipm.magnetics
           if m.get("wellbore_id") == h.wellbore_id][0]
    prog = r.programs[hid][0]
    tool_id = prog.survey_tool_id or ipm.run_tool_map[prog.survey_header_id]
    assert ipm.tools[tool_id].name == "Magnetic, std, mag-corr"

    F = FEET_TO_METERS
    md = np.array([s["md"] for s in stations]) * F
    inc = np.array([s["inc"] for s in stations])
    azi = np.array([s["azi"] for s in stations])
    cc = np.array([s["cov"] for s in stations]) * F ** 2

    s = we.survey.Survey(
        md=md, inc=inc, azi=azi,
        header=we.survey.SurveyHeader(
            name="F-12 PLAN", azi_reference="grid",
            b_total=float(mag["field_strength"]),
            dip=float(mag["dip_angle"]),
            declination=float(mag["declination"]),
        ),
        error_model=ipm.error_model(tool_id),
    )
    cov = s.err.errors.cov_NEVs

    sig_we = np.sqrt(np.stack(
        [cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]], axis=1))
    sig_cp = np.sqrt(np.stack([cc[:, 3], cc[:, 0], cc[:, 5]], axis=1))

    diff = np.abs(sig_we - sig_cp)
    tol = np.maximum(0.01 * sig_cp, 0.02)
    assert np.all(diff[:, 0] <= tol[:, 0]), "sigma_N off >1% (>2 cm)"
    assert np.all(diff[:, 1] <= tol[:, 1]), "sigma_E off >1% (>2 cm)"
    v_ok = (sig_we[:, 2] >= 0.90 * sig_cp[:, 2] - 0.02) \
        & (sig_we[:, 2] <= 1.01 * sig_cp[:, 2] + 0.02)
    assert np.all(v_ok), "sigma_V outside the documented -8%..+1% band"


def test_f14_actual_composed_multi_tool_vs_compass(ipm):
    """F-14 ACTUAL definitive: gyro to 2601 m, then two MWD runs — composed
    with SurveyComposition using the per-section ACTUAL tool IPMs and
    compared against COMPASS's own composed covariances.

    With per-term cross-run realisation sharing and the 'well' bucket
    composed (both landed after probcol's multi-well lateral-under-run
    report), the perpendicular-to-heading axes sit within ~1% of COMPASS at
    TD and NOTHING runs materially under — the residuals are on the
    conservative side (sigma_V up to ~+28% mid-well). Bands assert exactly
    that: no non-conservative under-run, bounded overshoot.
    """
    from welleng.composition import SurveyComposition, SurveySection
    from welleng.exchange.edm_stream import EDMReader, FEET_TO_METERS

    r = EDMReader(VOLVE)
    st = None
    for hid, stations in r._def_stations.items():
        h = r.headers[hid]
        wb = r.wellbores.get(h.wellbore_id)
        if wb and wb.name == "F-14" and h.phase == "ACTUAL":
            st = sorted(stations, key=lambda s: s["md"])
            header = h
    assert st is not None
    mag = [m for m in ipm.magnetics
           if m.get("wellbore_id") == header.wellbore_id][0]

    F = FEET_TO_METERS
    md = np.array([s["md"] for s in st]) * F
    inc = np.array([s["inc"] for s in st])
    azi = np.array([s["azi"] for s in st])
    cc = np.array([s["cov"] for s in st]) * F ** 2

    sh = we.survey.SurveyHeader(
        name="F-14", azi_reference="grid",
        latitude=58.4416, longitude=1.8875,
        b_total=float(mag["field_strength"]),
        dip=float(mag["dip_angle"]),
        declination=float(mag["declination"]),
    )
    gyro = ipm.error_model("Wellbore Surveyor, stat")
    mwd = ipm.error_model("Magnetic, std, non-mag")
    i1 = int(np.argmin(np.abs(md - 2601.0)))    # gyro -> MWD (program break)
    i2 = int(np.argmin(np.abs(md - 2733.1)))    # MWD run 1 -> run 2

    comp = SurveyComposition(sections=[
        SurveySection(md=md[:i1 + 1], inc=inc[:i1 + 1], azi=azi[:i1 + 1],
                      header=sh, error_model=gyro, tool_id="gyro"),
        # a gyro and an MWD share no error realisations
        SurveySection(md=md[i1:i2 + 1], inc=inc[i1:i2 + 1],
                      azi=azi[i1:i2 + 1], header=sh, error_model=mwd,
                      tool_id="mwd1", share_mode="all_independent"),
        # same magnetic reference across the two MWD runs
        SurveySection(md=md[i2:], inc=inc[i2:], azi=azi[i2:], header=sh,
                      error_model=mwd, tool_id="mwd2",
                      share_mode="globals_shared"),
    ]).survey()
    cov = comp.cov_nev

    sig_we = np.sqrt(np.stack(
        [cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]], axis=1))
    sig_cp = np.sqrt(np.stack([cc[:, 3], cc[:, 0], cc[:, 5]], axis=1))

    m = sig_cp > 0.05
    for col, label, lo, hi in ((0, "N", 0.95, 1.12), (1, "E", 0.95, 1.12)):
        ratio = sig_we[m[:, col], col] / sig_cp[m[:, col], col]
        assert np.all((ratio > lo) & (ratio < hi)), \
            f"sigma_{label} outside [{lo}, {hi}] of COMPASS"
    v_ratio = sig_we[m[:, 2], 2] / sig_cp[m[:, 2], 2]
    assert np.all((v_ratio > 0.90) & (v_ratio < 1.30)), \
        "sigma_V outside the conservative-side band"


def test_f15d_ew_high_inc_no_lateral_underrun(ipm):
    """F-15D — the E-W high-inclination case (azi ~106 deg holding while
    building to ~80 deg inc): the well that exposed the systematic lateral
    under-run (sigma_N -11% at TD, the NON-conservative direction for
    anti-collision). With per-term cross-run global sharing (the IFR
    reference is ONE realisation across both MWD runs) the
    perpendicular-to-heading axis is within ~1% at TD and nothing runs
    materially under COMPASS.
    """
    from welleng.composition import SurveyComposition, SurveySection
    from welleng.exchange.edm_stream import EDMReader, FEET_TO_METERS

    r = EDMReader(VOLVE)
    st = None
    for hid, stations in r._def_stations.items():
        h = r.headers[hid]
        wb = r.wellbores.get(h.wellbore_id)
        if wb and wb.name == "F-15D" and h.phase == "ACTUAL":
            st = sorted(stations, key=lambda s: s["md"])
            header = h
    assert st is not None
    mag = [m for m in ipm.magnetics
           if m.get("wellbore_id") == header.wellbore_id][0]

    F = FEET_TO_METERS
    md = np.array([s["md"] for s in st]) * F
    inc = np.array([s["inc"] for s in st])
    azi = np.array([s["azi"] for s in st])
    cc = np.array([s["cov"] for s in st]) * F ** 2

    sh = we.survey.SurveyHeader(
        name="F-15D", azi_reference="grid",
        latitude=58.4416, longitude=1.8875,
        b_total=float(mag["field_strength"]),
        dip=float(mag["dip_angle"]),
        declination=float(mag["declination"]),
    )
    gyro = ipm.error_model("Wellbore Surveyor, stat")
    mwd1 = ipm.error_model("Magn, IFR, mag-corr, dual incl")
    mwd2 = ipm.error_model("Magn, IFR, non-mag, dual incl")
    # The ACTUAL survey program (CD_SURVEY_PROGRAM) — four runs. Composing
    # with a simplified 3-section split (both mag-corr runs merged) leaves a
    # +10% sigma_E excess through the turn: the mid-leg systematic restart
    # is real information. Use the program's own boundaries.
    i1 = int(np.argmin(np.abs(md - 1310.0)))   # drop-gyro end
    i1b = int(np.argmin(np.abs(md - 2560.0)))  # mag-corr run 1 -> run 2
    i2 = int(np.argmin(np.abs(md - 3220.0)))   # mag-corr -> non-mag
    comp = SurveyComposition(sections=[
        SurveySection(md=md[:i1 + 1], inc=inc[:i1 + 1], azi=azi[:i1 + 1],
                      header=sh, error_model=gyro, tool_id="gyro"),
        SurveySection(md=md[i1:i1b + 1], inc=inc[i1:i1b + 1],
                      azi=azi[i1:i1b + 1], header=sh, error_model=mwd1,
                      tool_id="mwd1a", share_mode="all_independent"),
        # same-tool second run: fresh systematic, shared IFR geomag globals
        SurveySection(md=md[i1b:i2 + 1], inc=inc[i1b:i2 + 1],
                      azi=azi[i1b:i2 + 1], header=sh, error_model=mwd1,
                      tool_id="mwd1b", share_mode="globals_shared"),
        SurveySection(md=md[i2:], inc=inc[i2:], azi=azi[i2:], header=sh,
                      error_model=mwd2, tool_id="mwd2",
                      share_mode="globals_shared"),
    ]).survey()
    cov = comp.cov_nev

    sig_we = np.sqrt(np.stack(
        [cov[:, 0, 0], cov[:, 1, 1], cov[:, 2, 2]], axis=1))
    sig_cp = np.sqrt(np.stack([cc[:, 3], cc[:, 0], cc[:, 5]], axis=1))
    td = sig_we[-1] / sig_cp[-1]

    # with the true program, every axis reproduces COMPASS at TD
    assert 0.95 < td[0] < 1.05, f"sigma_N TD ratio {td[0]:.3f}"
    assert 0.95 < td[1] < 1.05, f"sigma_E TD ratio {td[1]:.3f}"
    assert 0.95 < td[2] < 1.08, f"sigma_V TD ratio {td[2]:.3f}"
    # nothing materially under COMPASS anywhere sigma is macroscopic.
    # sigma_V's floor is wider: the characterised well-level vertical
    # reference term (documented on F-12) that the export does not carry
    # dips the mid-well vertical ratio to ~0.86.
    m = sig_cp > 0.5
    for col, floor in ((0, 0.90), (1, 0.90), (2, 0.85)):
        ratio = sig_we[m[:, col], col] / sig_cp[m[:, col], col]
        assert np.all(ratio > floor), \
            f"non-conservative under-run returned (axis {col})"
