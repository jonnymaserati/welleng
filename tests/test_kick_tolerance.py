"""Golden test for the clean-room kick-tolerance module.

Fixtures are the PUBLISHED Table-1 numbers from the public SPE paper
SPE-208788-PA. The gas properties (Z_s, Z_td, rho_gas_s) are now COMPUTED by the
clean-room Hall & Yarborough (1973) backend (``gas_z.py``) from pure methane and
the model's own P/T conditions -- NOT injected. ``test_gas_z.py`` validates that
backend against CoolProp and the paper's printed Z values.

Exact assertions (independent of gas properties):
    P_td = 6893 psi   (max-credible PP = PP + kick = 12.6 ppg governs)
    B    = 7688 psi

~1% assertions (kick outputs):
    V_drilled  ~= 27.86 bbl   (computed backend: 27.58, 1.0%)
    V_swab     ~= 43.79 bbl   (computed backend: 43.25, 1.2%)
    PP @ KT=25 ~= 12.74 ppg   (computed backend: 12.72, 0.1%)

Why ~1% and NOT bit-exact
-------------------------
The paper's Table-1 is internally inconsistent. Its printed shoe Z (1.123) sits
above the value at the shoe *fracture* pressure P_lot - P_apl (~1.04); it
corresponds to a near-bottom-hole influx-gas pressure. The clean-room backend
evaluates the influx gas at the physically-defensible shoe pressure (bottom-hole
pressure reduced by the static influx-gas column), giving Z_s ~= 1.134,
rho_gas_s ~= 1.72, and A ~= 239.4. The kick volumes then land within ~1-1.3% of
the published cells. That residual is the PAPER'S OWN Table-1 inconsistency; we
do NOT tune any constant/pressure to force a closer match.
"""

import math
from dataclasses import replace


from welleng.kick_tolerance.core import (
    KickInputs,
    annular_capacity_dpa,
    constant_A,
    constant_B,
    drill_kick,
    influx_volume_A7,
    ppg_to_psi,
    resolve_gas_properties,
    swab_kick,
)

# Documented residual from the paper's Table-1 gas-property inconsistency.
KICK_VOLUME_REL_TOL = 0.013


def make_inputs(model_revision: str = "spe-208788") -> KickInputs:
    # Table-1 published fixtures (public SPE-208788-PA). Gas properties are
    # COMPUTED by the Hall-Yarborough backend (Z_s / Z_td / rho_gas_s = None).
    #
    # PINNED to the "spe-208788" revision. Reproducing the paper's table means
    # reproducing its MODEL, and the shipped default ("column-mean-2026")
    # deliberately differs from it -- it evaluates the influx state at the
    # gas-column mean temperature rather than at the shoe. That is exactly what
    # a frozen revision is for; see test_model_revision.py for the delta.
    return KickInputs(
        model_revision=model_revision,
        rho_mud=11.9,
        PP=11.5,
        kick_intensity=1.1,
        P_lot=16.0,
        P_apl=210.0,
        D_td=10500.0,
        D_lot=6500.0,
        T_s=212.0,
        T_td=302.0,
        # A-3 simplification: drillpipe (4") in the 6.125" hole at the shoe.
        V_dpa=annular_capacity_dpa(6.125, 4.0),
        # Gas properties left as None -> computed from pure methane.
        kt_threshold=25.0,
    )


def test_gas_properties_computed():
    """Backend reproduces the paper's printed Table-1 gas properties (~1%)."""
    Z_s, Z_td, rho_gas_s = resolve_gas_properties(make_inputs())
    assert math.isclose(Z_td, 1.1650, rel_tol=0.005)     # paper 1.1650
    assert math.isclose(Z_s, 1.1230, rel_tol=0.011)      # paper 1.1230
    assert math.isclose(rho_gas_s, 1.710, rel_tol=0.011)  # paper 1.710


def test_P_td_exact():
    """A-1: P_td governed by max-credible PP (12.6 ppg) -> 6893 psi."""
    res = drill_kick(make_inputs())
    assert round(res.P_td) == 6893


def test_B_exact():
    """A-6: B = 7688 psi."""
    res = drill_kick(make_inputs())
    assert round(res.B) == 7688


def test_A_constant_paper_band():
    """A-5 with the computed backend lands at ~239.4 (documented Table-1 band).

    The published A cell is 242; back-solving individual Table-1 cells gives
    ~241.2/242.1/242.5. The computed-gas forward evaluation is ~239.4 -- within
    the paper's own ~1% spread.
    """
    A = constant_A(make_inputs())
    assert 238.0 < A < 243.0
    assert math.isclose(A, 239.4, rel_tol=3e-3)


def test_V_drilled_within_1pct():
    """A-7 tolerable influx at max-credible PP ~= 27.86 bbl (~1%)."""
    res = drill_kick(make_inputs())
    assert math.isclose(res.capacity, 27.86, rel_tol=KICK_VOLUME_REL_TOL)


def test_swab_reproduces_spe208788_shared_A():
    """Reproduce SPE-208788-PA Table-1 swab figure (43.79 bbl) under ITS OWN
    convention: one A constant shared across A-7/A-8, i.e. swab gas stationed at
    the drill max-credible pressure (PP+KI). Validation = reproduce the source
    result with the source method.

    Our shipped model (``swab_kick``) does NOT do this -- it stations the swab
    gas at mud hydrostatic (KI-independent), a conscious divergence documented
    in ``swab_kick`` per Nassab SPE-202426-PA. See
    ``test_swab_kick_is_kick_intensity_invariant``.
    """
    inp = make_inputs()
    A_shared = constant_A(inp)                       # scenario-stationed (208788 shared A)
    B = constant_B(inp)
    P_td_swab = ppg_to_psi(inp.rho_mud, inp.D_td)    # A-8 substitution
    v_swab_208788 = influx_volume_A7(A_shared, B, P_td_swab)
    assert math.isclose(v_swab_208788, 43.79, rel_tol=KICK_VOLUME_REL_TOL)


def test_swab_model_diverges_from_208788_consciously():
    """Our swab_kick (mud-stationed gas) sits BELOW the 208788 shared-A figure
    for this underbalanced-KI fixture -- conservative direction, ~1.5%."""
    inp = make_inputs()
    assert inp.PP + inp.kick_intensity > inp.rho_mud   # underbalanced-KI: the two diverge
    ours = swab_kick(inp).capacity
    assert ours < 43.79
    assert math.isclose(ours, 43.15, rel_tol=0.01)


def test_pp_at_threshold_within_1pct():
    """A-9 pore pressure at KT = 25 bbl ~= 12.74 ppg (~1%)."""
    res = drill_kick(make_inputs())
    assert math.isclose(res.pp_at_threshold, 12.74, rel_tol=0.01)


def test_swab_never_overrides_drill_separate_results():
    """Drill and swab are independent first-class results (never blind-min'd)."""
    inp = make_inputs()
    d = drill_kick(inp)
    s = swab_kick(inp)
    assert d.case == "drill" and s.case == "swab"
    # Different bottom-hole pressures -> genuinely separate cases.
    assert d.P_td != s.P_td
    # B is a gas-free constant (shared); A differs because the influx gas is
    # stationed at each case's own P_td (drill: PP+KI scenario; swab: mud
    # hydrostatic) -- make_inputs is underbalanced (PP+KI 12.6 > mud 11.9).
    assert math.isclose(d.B, s.B)
    assert not math.isclose(d.A, s.A)
    assert d.capacity != s.capacity


def test_swab_kick_is_kick_intensity_invariant():
    """swab_kick must not depend on kick_intensity, INCLUDING via gas stationing.

    Nassab SPE-202426-PA Eqs 8-9: the swab bottom-hole pressure is the mud
    hydrostatic, independent of PP/KI. Regression for the leak where swab gas
    Z/rho were stationed at scenario_P_td (PP+KI) -> KI crept in anti-
    conservatively. make_inputs is underbalanced so the leak, if present, shows.
    """
    base = make_inputs()
    caps = [swab_kick(replace(base, kick_intensity=ki)).capacity
            for ki in (0.0, 1.1, 2.5, 5.0)]
    assert max(caps) - min(caps) < 1e-9
    # sanity: drill DOES respond to KI (the invariance is swab-specific)
    drills = [drill_kick(replace(base, kick_intensity=ki)).capacity
              for ki in (0.0, 5.0)]
    assert abs(drills[0] - drills[1]) > 1.0


def test_deviated_scales_A_by_inverse_cos_inc_shoe():
    """Deviated form (Nassab et al., SPE-202426-PA): the gas column's vertical
    height H_gas becomes an along-hole length L_gas = H_gas / cos(inc_shoe), so
    A scales by 1 / cos(inc_shoe). inc_shoe = 0 recovers the vertical Table-1
    form; a deviated well tolerates a LARGER influx volume for the same
    fracture-limiting vertical gas column."""
    A_vert = constant_A(make_inputs())               # inc_shoe = 0.0 (vertical)
    for inc in (15.0, 30.0, 45.0, 60.0):
        A_dev = constant_A(replace(make_inputs(), inc_shoe=inc))
        assert math.isclose(
            A_dev, A_vert / math.cos(math.radians(inc)), rel_tol=1e-12
        )
    # inc_shoe = 0 is exactly the vertical constant (regression guard)
    assert math.isclose(constant_A(replace(make_inputs(), inc_shoe=0.0)), A_vert)
    # deviated -> larger tolerable influx capacity than vertical
    dev = drill_kick(replace(make_inputs(), inc_shoe=45.0)).capacity
    vert = drill_kick(make_inputs()).capacity
    assert dev > vert


def test_from_survey_reads_tvd_and_inclination():
    """KickInputs.from_survey pulls D_lot/D_td (TVD) and inc_shoe off a welleng
    Survey via min-curvature interpolation, so callers don't hand-transcribe them."""
    import numpy as np
    import welleng as we

    survey = we.survey.Survey(
        md=np.array([0.0, 1000.0, 2000.0, 3000.0]),
        inc=np.array([0.0, 0.0, 30.0, 60.0]),
        azi=np.array([0.0, 0.0, 45.0, 45.0]),
    )
    params = dict(
        rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        T_s=212.0, T_td=302.0, V_dpa=annular_capacity_dpa(6.125, 4.0),
    )
    inp = KickInputs.from_survey(survey, shoe_md=2500.0, td_md=3000.0, **params)
    # deviated section (inc 30@2000 -> 60@3000): inclination at shoe ~45 deg
    assert math.isclose(inp.inc_shoe, 45.0, abs_tol=1.0)
    # TVD is shallower than MD for the deviated stations
    assert inp.D_lot < 2500.0 and inp.D_td < 3000.0
    # produces a usable inputs object
    assert drill_kick(inp).case == "drill"


if __name__ == "__main__":
    inp = make_inputs()
    d = drill_kick(inp)
    s = swab_kick(inp)
    print(f"A          = {d.A:.4f} bbl")
    print(f"P_td(drill)= {d.P_td:.2f} psi   B = {d.B:.2f} psi")
    print(f"V_drilled  = {d.capacity:.4f} bbl")
    print(f"P_td(swab) = {s.P_td:.2f} psi")
    print(f"V_swab     = {s.capacity:.4f} bbl")
    print(f"PP @ KT=25 = {d.pp_at_threshold:.4f} ppg")


# --- influx-fluid shortlist (single source of truth for the API picker) -------
def test_fluid_presets_valid_and_copy_safe():
    """Curated shortlist: normalised compositions, CoolProp-resolvable names,
    and defensively copied (mutating the return must not corrupt the source)."""
    from welleng.kick_tolerance import fluid_presets, fluid_aliases, FLUID_PRESETS

    presets = fluid_presets()
    assert len(presets) == len(FLUID_PRESETS) >= 5
    aliases = fluid_aliases()
    known = set(aliases.values()) | {v.lower() for v in aliases}
    for p in presets:
        assert p["label"] and isinstance(p["composition"], dict)
        assert math.isclose(sum(p["composition"].values()), 1.0, abs_tol=1e-9)
        for name in p["composition"]:
            # every component is a CoolProp canonical name or a known alias
            assert name in aliases.values() or name.lower() in aliases, name
    # copy-safety: the returned structures are independent of the module constant
    presets[0]["composition"][next(iter(presets[0]["composition"]))] = 0.123
    assert math.isclose(sum(FLUID_PRESETS[0]["composition"].values()), 1.0, abs_tol=1e-9)


def test_fluid_aliases_map_co2():
    from welleng.kick_tolerance import fluid_aliases

    a = fluid_aliases()
    assert a["co2"] == "CarbonDioxide" and a["n2"] == "Nitrogen"
    a["co2"] = "BROKEN"                      # mutating the copy must not leak
    from welleng.kick_tolerance import fluid_aliases as fa2
    assert fa2()["co2"] == "CarbonDioxide"


def test_the_gravitational_constant_cancels_out_of_the_answer():
    """Guards a claim that was got WRONG, confidently, and committed.

    It was recorded that welleng's g = 0.0521 (against an exact 0.05194805) made
    the kick tolerance ~5.7% high, and that latitude was worth a further ~10%.
    Both were artefacts of comparing two models by matching their PSI values
    instead of their equivalent mud weights. The real figure is ~0.001%.

    Where pressures are equivalent mud weights -- the industry convention --
    BHP = PP.g.TD and FRAC = LOT.g.shoe are both gradient-derived, so

        h = [rho_mud.(TD - shoe) - (PP.TD - LOT.shoe)] / (rho_mud - rho_gas)

    contains no g, and the Boyle ratio FRAC/BHP = (LOT.shoe)/(PP.TD) cancels it
    as well. g survives only where an absolute pressure enters that is NOT
    gradient-derived -- here just the atmospheric term in A-7.

    Measured directly: over the full planetary range of gravity (equator to
    pole) the gas height is identical and the tolerable influx moves 0.0012%.

    If someone "improves" the constant and expects the answer to move, this test
    is why it does not.
    """
    import welleng.kick_tolerance.core as core

    td, shoe, mud, cap = 12000.0, 6000.0, 10.0, (8.5 ** 2 - 5.0 ** 2) / 1029.4
    pp, lot, gas = 11.2019230769, 12.8846153846, 1.15021

    def solve():
        return drill_kick(KickInputs(
            rho_mud=mud, PP=pp, kick_intensity=0.0, P_lot=lot, P_apl=0.0,
            D_td=td, D_lot=shoe, V_dpa=cap, T_s=100.0, T_td=100.0,
            Z_s=1.0, Z_td=1.0, rho_gas_s=gas, ideal_gas=True,
        ))

    original = core.G_PSI_PER_PPG_FT
    try:
        results = []
        for g in (0.0521, 0.05194805, 0.0518086, 0.0520832):   # ours, exact, equator, pole
            core.G_PSI_PER_PPG_FT = g
            results.append(solve())
    finally:
        core.G_PSI_PER_PPG_FT = original

    heights = [r.H_gas for r in results]
    volumes = [r.capacity for r in results]

    # the hydrostatic balance is exactly g-free
    assert max(heights) - min(heights) < 1e-9
    # and the volume moves only through the atmospheric term
    assert (max(volumes) - min(volumes)) / min(volumes) < 1e-4

    # it is also the g-free closed form, not merely self-consistent
    g_free = ((td - shoe) * mud - (pp * td - lot * shoe)) / (mud - gas)
    assert math.isclose(heights[0], g_free, rel_tol=1e-9)


def test_matches_an_independent_practitioner_worked_case():
    """External check against an independently derived Driller's Method
    result -- 12,000 ft TD, 6,000 ft shoe, 10.0 ppg mud, 6,990 psi reservoir,
    4,020 psi fracture at shoe, 8-1/2 in hole, 5 in pipe.

    The reference reports a gas height of 326.15986 ft and a kick tolerance of
    8.65186401728 bbl.

    This is the only anchor welleng has that is not a paper we also transcribe,
    and it is the one that caught the g mistake above. The source is an
    unpublished third-party working document; it is deliberately not cited here
    (see the local validation record).
    """
    td, shoe, mud, cap = 12000.0, 6000.0, 10.0, (8.5 ** 2 - 5.0 ** 2) / 1029.4

    result = drill_kick(KickInputs(
        rho_mud=mud, PP=11.2019230769, kick_intensity=0.0, P_lot=12.8846153846,
        P_apl=0.0, D_td=td, D_lot=shoe, V_dpa=cap, T_s=100.0, T_td=100.0,
        Z_s=1.0, Z_td=1.0, rho_gas_s=1.15021, ideal_gas=True,
    ))

    assert math.isclose(result.H_gas, 326.15986, rel_tol=0.002)      # 0.06%
    assert math.isclose(result.capacity, 8.65186401728, rel_tol=0.01)  # 0.76%


def test_a_clamped_bubble_height_is_flagged_not_silently_reported():
    """A response that reports overflow in one field and no overflow in another is
    worse than either answer alone.

    The column solve clips H_gas to the OPEN-HOLE LENGTH. When that clamp binds,
    the reported geometry says the bubble exactly fills the open hole -- its top
    lands on the shoe to the digit -- while `capacity` is NOT clipped with it and
    says 2.6x more. Both cannot be true.

    The configuration is bubble-length-limited in the sense of SPE/IADC-140113:
    a single coherent bubble cannot be longer than the open hole it occupies, so
    the limiting configuration is unreachable. Flagged rather than silently
    returned; the geometry and the volume must not be read together.
    """
    v_dpa = annular_capacity_dpa(6.125, 4.0)
    inp = KickInputs(rho_mud=11.9, PP=12.2, kick_intensity=1.0, P_lot=15.0,
                     P_apl=0.0, D_td=10500.0, D_lot=9800.0,
                     T_s=212.0, T_td=302.0, V_dpa=v_dpa)

    r = drill_kick(inp)
    open_hole = (inp.D_td - inp.D_lot) * v_dpa

    assert r.bubble_length_limited is True
    assert math.isclose(r.H_gas, inp.D_td - inp.D_lot, rel_tol=1e-9)  # the clamp
    assert r.capacity > open_hole                                     # the contradiction

    # and an ordinary case is not flagged
    ok = drill_kick(KickInputs(rho_mud=11.9, PP=11.5, kick_intensity=1.1,
                               P_lot=16.0, P_apl=210.0, D_td=10500.0, D_lot=6500.0,
                               T_s=212.0, T_td=302.0, V_dpa=v_dpa))
    assert ok.bubble_length_limited is False


def test_a_negative_tolerance_is_flagged_because_it_is_not_a_volume():
    """A-7 can return a negative influx. That is not a tolerance -- it means the
    maximum-credible pore pressure already exceeds what the shoe holds with NO
    influx, so the section is undrillable on these inputs.

    Left unfloored so the magnitude still says how far past the limit the case
    is, but flagged so a consumer cannot display it as a volume.
    """
    r = drill_kick(KickInputs(rho_mud=11.9, PP=12.6, kick_intensity=1.5,
                              P_lot=12.6, P_apl=0.0, D_td=10500.0, D_lot=6500.0,
                              T_s=212.0, T_td=302.0,
                              V_dpa=annular_capacity_dpa(6.125, 4.0)))

    assert r.capacity < 0.0
    assert r.capacity_negative is True


def test_d_already_fractured_is_not_the_shoe_holding():
    """A design-curve sweep that shifts the FP profile down and re-solves can, at the
    weakest shoe, return `open_hole_unconstrained`
    with the SAME volume as the strongest shoe, so the sweep breaks on its first
    point and the curve came back empty.

    An empty breach-candidate set has two OPPOSITE causes and the per-depth solves
    return None for both: the shoe is far too strong to breach, or the mud column
    ALONE already meets FP so there is no intact state to grow a bubble from. The
    second reported the full open-hole capacity for a well that is losing returns
    before any gas enters -- the unsafe direction.
    """
    from welleng.kick_tolerance.analytical import analytical_kick_tolerance
    from welleng.kick_tolerance.core import fahrenheit_to_rankine
    from welleng.kick_tolerance.migration import WellSection

    g, shoe, td, mud = 0.0521, 6500.0, 9800.0, 11.4
    sections = [WellSection(0.0, shoe, 0.1215, False),
                WellSection(shoe, td, 0.1215, True)]
    solve = lambda fp_shoe, fp_td: analytical_kick_tolerance(  # noqa: E731
        sections=sections, pp=([0.0, td], [11.0, 11.0]),
        fp=([shoe, td], [fp_shoe, fp_td]), bhp_psi=g * mud * td, rho_mud_ppg=mud,
        gas_bh_state=(g * mud * td, fahrenheit_to_rankine(180.0), None, None),
        geothermal=([0.0, td], [fahrenheit_to_rankine(60.0),
                                fahrenheit_to_rankine(180.0)]))

    # mud alone (3860.6 psi at the shoe) already exceeds a 10.4 ppg frac (3522.0)
    bad = solve(10.4, 11.0)
    assert bad.already_fractured is True
    assert bad.max_influx_bbl == 0.0
    assert bad.open_hole_unconstrained is False       # the bug: this used to be True
    assert bad.binding_depth_tvd == shoe

    # a genuinely strong shoe is still unconstrained, and must NOT set the new flag
    strong = solve(14.2, 15.5)
    assert strong.open_hole_unconstrained is True
    assert strong.already_fractured is False

    # and the two must not report the same volume any more
    assert bad.max_influx_bbl != strong.max_influx_bbl

    # monotone in shoe strength -- the property api's sweep relies on
    kts = [solve(12.4 + o, 13.0 + o).max_influx_bbl for o in (-2.0, -1.0, 0.0, 1.0)]
    assert kts == sorted(kts), kts


def test_maasp_single_shoe_is_the_frac_minus_mud_identity():
    """MAASP = P_frac(shoe) - mud hydrostatic to the shoe, shut in.

    `P_apl` is NOT deducted: annular friction is a CIRCULATING term and MAASP is a
    closed-in limit, so subtracting it would understate what the annulus can hold.
    The fixture carries P_apl=210 psi precisely so a regression that started
    deducting it would move this number.
    """
    from welleng.kick_tolerance.core import ppg_to_psi

    inp = KickInputs(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0,
                     P_apl=210.0, D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
                     V_dpa=annular_capacity_dpa(6.125, 4.0))
    r = drill_kick(inp)

    expected = ppg_to_psi(16.0, 6500.0) - ppg_to_psi(11.9, 6500.0)
    assert math.isclose(r.maasp_psi, expected, rel_tol=1e-12)
    assert math.isclose(r.maasp_psi, 1388.46, abs_tol=0.01)
    assert not math.isclose(r.maasp_psi, expected - 210.0, rel_tol=1e-9)  # P_apl kept


def test_maasp_a_weak_zone_below_the_shoe_governs():
    """The industry convention evaluates MAASP at the shoe. With a CONSTANT
    fracture gradient that is exact -- `g.d.(FP_emw - rho_mud)` grows with depth,
    so the shallowest exposed point governs. It stops being right the moment a
    weak zone sits BELOW the shoe, and then the shoe-only number is too HIGH.

    Same failure mode as assuming the swab bubble top sits at the shoe.
    """
    from welleng.kick_tolerance.migration import (
        G_PSI_PER_PPG_FT as G, WellSection, maasp,
    )

    shoe, td = 6000.0, 10000.0
    sections = [WellSection(0.0, shoe, 0.0459, False),
                WellSection(shoe, td, 0.0459, True)]
    fp = 4020.0 / (G * shoe)                        # 4020 psi at the shoe

    flat = maasp(sections, ([shoe, td], [fp, fp]), rho_mud_ppg=10.0)
    assert flat.governed_by_shoe is True
    assert flat.governing_tvd == shoe
    assert math.isclose(flat.maasp_psi, 4020.0 - G * 10.0 * shoe, rel_tol=1e-12)
    # constant fracture gradient -> the convention IS the limit
    assert math.isclose(flat.limiting_psi, flat.maasp_psi, rel_tol=1e-12)

    # a realistic weak zone: 1.0 ppg below the surrounding fracture gradient,
    # still comfortably above the 10.0 ppg mud, i.e. a hole you would actually drill
    weak = maasp(sections,
                 ([shoe, 7999.0, 8000.0, 8001.0, td], [fp, fp, fp - 1.0, fp, fp]),
                 rho_mud_ppg=10.0)
    # the CONVENTION is unchanged -- MAASP is a shoe property and does not move
    assert math.isclose(weak.maasp_psi, flat.maasp_psi, rel_tol=1e-12)
    # but the real limit is deeper and lower
    assert weak.governed_by_shoe is False
    assert weak.governing_tvd == 8000.0
    assert weak.limiting_psi < weak.maasp_psi


def test_maasp_requires_an_exposed_section():
    """Fully cased: nothing is exposed, so MAASP is undefined -- the limit is
    casing burst instead. Refuse rather than return the shoe's number."""
    from welleng.kick_tolerance.migration import WellSection, maasp

    try:
        maasp([WellSection(0.0, 6000.0, 0.0459, False)],
              ([0.0, 6000.0], [14.0, 14.0]), rho_mud_ppg=10.0)
    except ValueError as e:
        assert "open-hole" in str(e)
    else:
        raise AssertionError("fully cased hole must refuse a MAASP")


def test_influx_density_and_gradient_convert_both_ways():
    """The well-control literature quotes influx GRADIENTS, not densities --
    NOGEPA-50 states a gas range of 0.05-0.15 psi/ft -- so a user needs to compare
    a quoted gradient against what welleng computed from (P, T, composition).

    Comparison only. Converting a quoted gradient to a density and feeding it back
    is hand-injected gas properties through the front door, which is what `Z_s`,
    `Z_td` and `rho_gas_s` were removed for.
    """
    import numpy as np

    from welleng.kick_tolerance import gradient_to_ppg, ppg_to_gradient
    from welleng.kick_tolerance.migration import G_PSI_PER_PPG_FT as G

    rho = 1.70133
    grad = ppg_to_gradient(rho)
    assert math.isclose(grad, G * rho, rel_tol=1e-15)
    assert math.isclose(gradient_to_ppg(grad), rho, rel_tol=1e-12)   # round trip
    assert math.isclose(ppg_to_gradient(gradient_to_ppg(0.1)), 0.1, rel_tol=1e-12)

    # vectorised both ways
    arr = np.array([1.6, 1.7, 1.8])
    assert np.allclose(gradient_to_ppg(ppg_to_gradient(arr)), arr, rtol=1e-12)


def test_the_reported_gradient_uses_THIS_engines_constant():
    """Three ppg->psi/ft constants are in circulation and they differ by ~0.3%:
    the engine's 0.0521, NOGEPA's 0.052, and the exact 0.0519481 from standard
    gravity. The reported gradient MUST use the engine's, or it does not reproduce
    the column weight the engine actually applied and a consumer reconciling the
    two chases a discrepancy that is purely a choice of constant.

    Unlike a kick tolerance in equivalent mud weights, the constant does NOT divide
    out here -- this is an absolute gradient, so the full ~0.3% is in the answer.
    """
    from welleng.kick_tolerance.migration import G_PSI_PER_PPG_FT as G
    from welleng.kick_tolerance.nogepa import NOGEPA_G
    from welleng.units import PSI_PER_PPG_PER_FT

    inp = KickInputs(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0,
                     P_apl=210.0, D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0,
                     V_dpa=annular_capacity_dpa(6.125, 4.0))
    r = drill_kick(inp)

    assert math.isclose(r.rho_influx_gradient_psi_per_ft, G * r.rho_influx,
                        rel_tol=1e-15)
    # and demonstrably NOT the other two
    assert not math.isclose(r.rho_influx_gradient_psi_per_ft,
                            PSI_PER_PPG_PER_FT * r.rho_influx, rel_tol=1e-6)
    assert not math.isclose(r.rho_influx_gradient_psi_per_ft,
                            NOGEPA_G * r.rho_influx, rel_tol=1e-6)
    # a real-gas methane influx lands inside NOGEPA-50's stated gas range
    assert 0.05 <= r.rho_influx_gradient_psi_per_ft <= 0.15


def test_the_ppg_to_psi_constant_has_exactly_one_definition():
    """`core` and `migration` each declared their own `G_PSI_PER_PPG_FT = 0.0521`.
    Two definitions of one physical constant, agreeing only by coincidence of
    maintenance, with nothing that would fail if one moved. `migration` is now the
    single source and everything else imports it.

    Asserted as identity, not equality: two separate literals would be equal today
    and that is exactly the failure this guards.
    """
    from welleng.kick_tolerance import analytical, core, migration

    assert core.G_PSI_PER_PPG_FT is migration.G_PSI_PER_PPG_FT
    assert analytical.G_PSI_PER_PPG_FT is migration.G_PSI_PER_PPG_FT

    # NOGEPA's constant is a DIFFERENT quantity and must stay distinct -- it is what
    # that standard mandates for its own formula, not what this engine computes with.
    from welleng.kick_tolerance.nogepa import NOGEPA_G

    assert NOGEPA_G != migration.G_PSI_PER_PPG_FT


def test_a_density_does_not_determine_a_state():
    """`rho = P.M/(Z.R.T)` is ONE equation in TWO unknowns, so a density is a curve
    in (P, T), not a point. The solver must refuse an under-determined request rather
    than pick a branch."""
    from welleng.kick_tolerance import gas_state_from_density as solve

    for kwargs, why in (
        (dict(rho_ppg=2.0), "neither P nor T"),
        (dict(rho_ppg=2.0, p_psia=5400.0, t_rankine=640.0), "both P and T"),
        (dict(rho_ppg=2.0, gradient_psi_per_ft=0.1, p_psia=5400.0), "two targets"),
        (dict(p_psia=5400.0), "no target"),
    ):
        try:
            solve(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"must refuse: {why}")


def test_solving_a_quoted_density_exposes_an_impossible_temperature():
    """The diagnostic this exists for. A published case quotes 2.00 ppg with no
    temperature; the well has a BHP. Pin P, solve T, and the answer is ~77 F -- absurd
    at 10,000 ft TVD, which tells the user the figure does not describe this well's gas.

    Feeding that temperature back into the design is the hand-injected gas-property
    path that `Z_s`/`Z_td`/`rho_gas_s` were removed for. This is a comparison only.
    """
    from welleng.kick_tolerance import gas_state_from_density as solve
    from welleng.kick_tolerance.core import fahrenheit_to_rankine
    from welleng.kick_tolerance.migration import ppg_to_gradient

    s = solve(rho_ppg=2.00, p_psia=5400.0)
    assert s.solved_for == "temperature"
    assert math.isclose(s.rho_ppg, 2.00, rel_tol=1e-9)          # hits the target
    assert 530.0 < s.t_rankine < 545.0                          # ~70-85 F
    assert math.isclose(s.gradient_psi_per_ft, ppg_to_gradient(2.00), rel_tol=1e-12)

    # a gradient is an equivalent way in
    assert math.isclose(
        solve(gradient_psi_per_ft=ppg_to_gradient(2.00), p_psia=5400.0).t_rankine,
        s.t_rankine, rel_tol=1e-9,
    )

    # the SAME density at a REALISTIC 180 F needs a heavier gas (gas gravity ~0.69),
    # which is the other half of the diagnostic
    heavy = solve(rho_ppg=2.00, p_psia=5400.0, molar_mass_lbm=19.86)
    assert math.isclose(heavy.t_rankine - 460.0, 180.0, abs_tol=1.0)

    # converse direction: known BHT, solve the pressure
    conv = solve(rho_ppg=2.00, t_rankine=fahrenheit_to_rankine(180.0))
    assert conv.solved_for == "pressure"
    assert math.isclose(conv.rho_ppg, 2.00, rel_tol=1e-9)


def test_an_unreachable_density_says_what_IS_reachable():
    """The 'SG' trap: gas specific gravity is relative to AIR (a COMPOSITION), mud SG
    is relative to WATER (a density). A user typing 0.69 meaning gas gravity, read as
    SG-vs-water, becomes 5.76 ppg -- and methane cannot be that dense at 5400 psi at
    any temperature. It must fail LOUDLY and say what the achievable range is, because
    that is what tells the user which quantity they actually typed.
    """
    from welleng.kick_tolerance import gas_state_from_density as solve

    try:
        solve(rho_ppg=0.69 * 8.3454, p_psia=5400.0)
    except ValueError as e:
        msg = str(e)
        assert "unreachable" in msg
        assert "achievable range" in msg
        assert "composition" in msg
    else:
        raise AssertionError("5.76 ppg methane at 5400 psi must be refused")


def test_gas_gravity_sets_the_pseudo_criticals_not_just_the_molar_mass():
    """The defect this guards, found 2026-07-28.

    `molar_mass_lbm` scales DENSITY but says nothing about Z, and Hall-Yarborough needs
    PSEUDO-CRITICALS. Leaving them on methane's for a heavier gas is internally
    inconsistent and not by a little: at 5400 psi the temperature at which a
    0.686-gravity gas reaches 2.00 ppg is 179.9 F on methane's pseudo-criticals and
    194.6 F on Standing's -- a 14.7 F error, in a diagnostic whose whole purpose is to
    expose an implausible temperature.

    `gas_gravity=` sets both, together, and is the way in.
    """
    from welleng.kick_tolerance import gas_state_from_density as solve
    from welleng.kick_tolerance.gas_z import (
        M_AIR_LBM_PER_LBMOL, standing_pseudo_criticals,
    )

    gravity = 0.686
    tpc, ppc = standing_pseudo_criticals(gravity)
    assert tpc > 343.0                                  # heavier than methane

    consistent = solve(rho_ppg=2.00, p_psia=5400.0, gas_gravity=gravity)
    with pytest_warns_userwarning():
        inconsistent = solve(rho_ppg=2.00, p_psia=5400.0,
                             molar_mass_lbm=gravity * M_AIR_LBM_PER_LBMOL)

    # both hit the density; they disagree on the TEMPERATURE by ~15 F
    assert math.isclose(consistent.rho_ppg, 2.00, rel_tol=1e-9)
    assert math.isclose(inconsistent.rho_ppg, 2.00, rel_tol=1e-9)
    assert consistent.t_rankine - inconsistent.t_rankine > 10.0

    # gas gravity is relative to AIR and must reject a value that is plainly a
    # different quantity -- 6.9 as a typo for 0.69, or an SG-vs-water
    for bad in (6.9, 0.2):
        try:
            solve(rho_ppg=2.00, p_psia=5400.0, gas_gravity=bad)
        except ValueError as e:
            assert "relative to AIR" in str(e)
        else:
            raise AssertionError(f"gas_gravity={bad} must be refused")

    try:
        solve(rho_ppg=2.0, p_psia=5400.0, gas_gravity=0.69, molar_mass_lbm=19.86)
    except ValueError:
        pass
    else:
        raise AssertionError("gas_gravity AND molar_mass_lbm must be refused")


def pytest_warns_userwarning():
    """Minimal context manager -- this module does not import pytest."""
    import warnings
    from contextlib import contextmanager

    @contextmanager
    def _cm():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            yield
            assert any(issubclass(x.category, UserWarning) for x in w), \
                "expected a UserWarning about inconsistent pseudo-criticals"
    return _cm()


def test_the_solver_bracket_comes_from_the_correlation_validity_band():
    """The bracket started as a round 300-2000 degR and blew up on its own first probe:
    outside Hall-Yarborough's band (1.15 <= Tpr <= 3.0) the Newton does not lose
    accuracy, it FAILS TO CONVERGE and raises. The bracket is now derived from the band
    for the composition in use, so an unreachable target also reports the range the
    CORRELATION can speak to rather than one it was extrapolated across.
    """
    from welleng.kick_tolerance import gas_state_from_density as solve
    from welleng.kick_tolerance.gas_z import METHANE_TPC_RANKINE as TPC

    s = solve(rho_ppg=2.00, p_psia=5400.0)              # must not raise
    assert 1.15 * TPC <= s.t_rankine <= 3.0 * TPC

    try:
        solve(rho_ppg=25.0, p_psia=5400.0)
    except ValueError as e:
        assert "validity" in str(e) and "achievable range" in str(e)
    else:
        raise AssertionError("25 ppg methane must be refused")


def test_the_reference_temperature_makes_the_comparison_the_answer():
    """The comparison IS the output, so assembling it from a solved value plus a
    separately-held BHT is where it goes wrong --
    nothing guarantees they came from the same well.
    """
    from welleng.kick_tolerance import gas_state_from_density as solve
    from welleng.kick_tolerance.core import fahrenheit_to_rankine

    bht = fahrenheit_to_rankine(180.0)
    s = solve(rho_ppg=2.00, p_psia=5400.0, reference_t_rankine=bht)
    assert s.reference_t_rankine == bht
    assert math.isclose(s.t_discrepancy_rankine, s.t_rankine - bht, rel_tol=1e-12)
    assert s.t_discrepancy_rankine < -90.0        # methane: ~-103 F, wildly implausible

    # with the gravity that actually produces 2.00 ppg at 180 F, they agree
    ok = solve(rho_ppg=2.00, p_psia=5400.0, gas_gravity=0.6687,
               reference_t_rankine=bht)
    assert abs(ok.t_discrepancy_rankine) < 1.0

    # absent a reference there is no comparison to report
    assert solve(rho_ppg=2.00, p_psia=5400.0).t_discrepancy_rankine is None


def test_the_analytical_result_reports_its_own_influx_gradient():
    """Reported directly rather than derived, so a caller
    never converts with one of the three circulating ppg->psi/ft constants."""
    from welleng.kick_tolerance.analytical import analytical_kick_tolerance
    from welleng.kick_tolerance.core import fahrenheit_to_rankine
    from welleng.kick_tolerance.migration import (
        G_PSI_PER_PPG_FT as G, WellSection,
    )

    shoe, td = 6500.0, 9800.0
    cap = (8.5 ** 2 - 5.0 ** 2) / 1029.4
    bhp = 5871.67
    r = analytical_kick_tolerance(
        sections=[WellSection(0.0, shoe, cap, False),
                  WellSection(shoe, td, cap, True)],
        pp=([0.0, td], [11.0, 11.0]), fp=([shoe, td], [14.2, 15.5]),
        bhp_psi=bhp, rho_mud_ppg=11.4,
        gas_bh_state=(bhp, fahrenheit_to_rankine(180.0), None, None),
    )
    assert r.rho_influx_bh_ppg > 0.0
    assert math.isclose(r.rho_influx_bh_gradient_psi_per_ft,
                        G * r.rho_influx_bh_ppg, rel_tol=1e-15)


def test_the_gas_property_overrides_exist_and_the_docs_must_not_claim_otherwise():
    """Three shipped docstrings in 0.27.1 asserted that `Z_s`, `Z_td` and `rho_gas_s`
    "were removed". They were not. I made a RULING to remove them, never executed it,
    and then repeated the decision as accomplished fact -- in published documentation
    and three times to a consumer.

    The ruling's premise was also wrong. It claimed "welleng's CI covers none of it";
    in fact `tests/test_spe208788_worked_example.py` passes all three deliberately, to
    validate the closed form against the paper's OWN tabulated gas properties and so
    independently of our Z backend. That is a real and load-bearing use.

    This pins the fields' existence so the docs cannot drift back, and pins the reason.
    """
    import dataclasses

    from welleng.kick_tolerance import core, migration

    fields = {f.name for f in dataclasses.fields(KickInputs)}
    assert {"Z_s", "Z_td", "rho_gas_s"} <= fields

    for mod in (core, migration, __import__(
            "welleng.kick_tolerance", fromlist=["x"])):
        doc = mod.__doc__ or ""
        assert "were removed" not in doc, (
            f"{mod.__name__} claims the gas-property overrides were removed; "
            f"they are still present"
        )


def test_unconstrained_returns_a_bh_influx_not_the_open_hole_capacity():
    """On a hot circulated-kill well `case="drill"` makes the
    open hole non-constraining; a note claiming the returned
    volume is "the full open-hole gas capacity" would be wrong. It is NOT.

    Measured: the returned value is the BOTTOM-HOLE INFLUX whose expanded gas column
    just fills the open hole -- the same KIND of quantity as a fracture-limited
    tolerance (both bottom-hole influx), and smaller than the open-hole volumetric
    capacity by the gas-expansion ratio. The note now says so; this pins the fact so it
    cannot drift back to the false wording.
    """
    from welleng.kick_tolerance.analytical import analytical_kick_tolerance
    from welleng.kick_tolerance.core import fahrenheit_to_rankine
    from welleng.kick_tolerance.migration import WellSection

    shoe, td = 6500.0, 9800.0
    cap = (8.5 ** 2 - 5.0 ** 2) / 1029.4
    mud, pp, ki = 11.4, 11.0, 0.5
    bhp = 0.0521 * (pp + ki) * td
    sections = [WellSection(0.0, shoe, cap, False),
                WellSection(shoe, td, cap, True)]

    r = analytical_kick_tolerance(
        sections=sections, pp=([0.0, td], [pp, pp]),
        fp=([shoe, td], [14.2, 15.5]), bhp_psi=bhp, rho_mud_ppg=mud,
        gas_bh_state=(bhp, fahrenheit_to_rankine(180.0), None, None),
        case="drill", gas_density_mode="exact",
    )

    assert r.open_hole_unconstrained is True
    v_hole = cap * (td - shoe)                       # open-hole VOLUMETRIC capacity

    # the returned value is a BH influx: strictly less than the volumetric capacity,
    # by roughly the gas-expansion ratio (here ~2.6x), never equal to it
    assert 0.0 < r.max_influx_bbl < v_hole
    assert r.max_influx_bbl < 0.6 * v_hole           # expansion is real, not marginal

    # and the note must not resurrect the "capacity" claim
    note = r.breakpoints.get("note", "")
    assert "bottom-hole influx" in note.lower()
    assert "full open-hole gas capacity" not in note


def test_drill_kick_temperature_optional_when_density_given():
    """A drilling kick may omit T_s/T_td if an influx density is supplied: it then
    runs ideal-isothermal, where temperature CANCELS out of the column expansion.
    Requested by TA0 (2026-07-30) for density-based programs / simple well-control
    tools (e.g. Valaris WCT) that quote an influx density and no formation temperature.
    """
    v = annular_capacity_dpa(6.125, 4.0)
    base = dict(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
                D_td=10500.0, D_lot=6500.0, V_dpa=v)

    r = drill_kick(KickInputs(**base, rho_gas_s=0.69))            # NO temperature
    # identical to supplying any isothermal temperature in ideal mode -- T cancels
    for degf in (60.0, 250.0):
        ref = drill_kick(KickInputs(**base, rho_gas_s=0.69, T_s=degf, T_td=degf,
                                    ideal_gas=True))
        assert math.isclose(r.capacity, ref.capacity, rel_tol=1e-12), degf
    # the reported influx temperature is None, NOT a fabricated placeholder
    assert r.T_influx is None
    assert r.rho_influx is not None          # density path still populates the rest


def test_drill_kick_no_temperature_and_no_density_is_refused():
    """Omitting temperature is only meaningful with a density (ideal-isothermal).
    Without either, the gas state is undetermined -- refuse with a clear message,
    do not silently pick one."""
    v = annular_capacity_dpa(6.125, 4.0)
    try:
        drill_kick(KickInputs(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0,
                              P_apl=210.0, D_td=10500.0, D_lot=6500.0, V_dpa=v))
    except ValueError as e:
        assert "rho_gas_s" in str(e) and "temperature" in str(e).lower()
    else:
        raise AssertionError("no temperature AND no density must be refused")


def test_swab_kick_requires_temperature():
    """Temperature is optional ONLY for a drilling kick. A swab (trip) relaxes the
    gas column toward formation temperature, so it cannot be dropped."""
    v = annular_capacity_dpa(6.125, 4.0)
    try:
        swab_kick(KickInputs(rho_mud=11.9, PP=11.5, kick_intensity=1.1, P_lot=16.0,
                             P_apl=210.0, D_td=10500.0, D_lot=6500.0, V_dpa=v,
                             rho_gas_s=0.69))                     # density but no T
    except ValueError as e:
        assert "swab" in str(e).lower() and "temperature" in str(e).lower()
    else:
        raise AssertionError("swab_kick must require a formation temperature")
