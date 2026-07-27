"""SPE-202426-PA (Kiani Nassab et al., 2022) Fig. 12 -- temperature sign validation.

Case 1 (underbalanced kick) geothermal-vs-isothermal influx-temperature effect on
kick tolerance. welleng is a static "Company Model"-type calculator, so it maps to
the two STATIC Company-Model points of Fig. 12 (it cannot reproduce the dynamic
Simulator-D point, a transient multiphase annulus-temperature simulation). Inputs
transcribed from the paper's Fig. 2 Case-1 table (PDF p.4).

Fig. 12 static dots: geothermal ~16 bbl, isothermal(150 degC everywhere) ~13 bbl.

The SIGN (geothermal > isothermal) is not interpretation: the closed form gives
A ∝ T_td / T_s, so a geothermal profile (T_td > T_s) inflates A -- and hence KT --
above any single-temperature isothermal (ratio = 1). Isothermal-at-TD is the
hottest shoe case and the most conservative. (NB: SPE-208788-PA's prose summary of
this figure states the opposite sign -- see docs/dev/VALIDATION.md; the primary
figure + this closed form + welleng all give geothermal > isothermal.)
"""
import pytest

from welleng.kick_tolerance import KickInputs, drill_kick

M_TO_FT = 3.280839895
_c2f = lambda c: c * 9.0 / 5.0 + 32.0

# SPE-202426 Fig. 2, Case 1.
HOLE_IN, DP_IN = 6.125, 4.0
CASE1 = dict(
    rho_mud=12.0, PP=12.5, kick_intensity=0.0,     # max-credible PP (11.5 + 1.0 unc)
    P_lot=16.0, P_apl=160.0,
    D_lot=1500.0 * M_TO_FT, D_td=3200.0 * M_TO_FT,
    V_dpa=(HOLE_IN ** 2 - DP_IN ** 2) / 1029.4,
    # Fig. 12 is a published-result reproduction, so it runs the published
    # model. test_fig12_sign_survives_the_column_mean_revision checks that the
    # SIGN -- the physics claim, as opposed to the magnitudes -- is unchanged
    # under the shipped default.
    model_revision="spe-208788",
)
T_SHOE_GEO, T_TD_GEO = _c2f(100.0), _c2f(150.0)    # geothermal 100/150 degC
T_ISO = _c2f(150.0)                                 # isothermal = TD temp everywhere


def _kt(t_s, t_td):
    return drill_kick(KickInputs(T_s=t_s, T_td=t_td, **CASE1)).capacity


def test_fig12_geothermal_greater_than_isothermal():
    """Sign: geothermal KT > isothermal KT (isothermal-at-TD is the conservative,
    hottest-shoe case). Backed by A ∝ T_td/T_s."""
    geo = _kt(T_SHOE_GEO, T_TD_GEO)
    iso = _kt(T_ISO, T_ISO)
    assert geo > iso
    # welleng's self-consistent magnitudes (locks the reproduction).
    assert geo == pytest.approx(15.60, abs=0.15)
    assert iso == pytest.approx(13.28, abs=0.15)
    # ~15% apart, matching Fig. 12's ~16 vs ~13 dots.
    assert (geo - iso) / geo == pytest.approx(0.148, abs=0.02)


def test_fig12_sign_survives_every_revision():
    """The SIGN is physics, not a convention, so it must hold under EVERY
    revision -- including the default, where the influx is evaluated at the
    bubble's own state (mud beneath it) rather than after a whole-open-hole gas
    gradient. Magnitudes move; geothermal > isothermal does not."""
    kw = {k: v for k, v in CASE1.items() if k != "model_revision"}

    def kt(t_s, t_td):
        return drill_kick(KickInputs(T_s=t_s, T_td=t_td, **kw)).capacity

    from welleng.kick_tolerance.core import MODEL_REVISIONS
    for revision in sorted(MODEL_REVISIONS):
        g = drill_kick(KickInputs(T_s=T_SHOE_GEO, T_td=T_TD_GEO,
                                  model_revision=revision, **kw)).capacity
        i = drill_kick(KickInputs(T_s=T_ISO, T_td=T_ISO,
                                  model_revision=revision, **kw)).capacity
        assert g > i, f"{revision}: geothermal {g:.4f} !> isothermal {i:.4f}"


def test_fig12_matches_static_company_model_dots():
    """welleng lands within ~1 bbl of Fig. 12's two static Company-Model dots
    (geothermal ~16, isothermal ~13). The dynamic Simulator-D point (17 bbl) is a
    transient sim welleng cannot and does not attempt to reproduce."""
    assert _kt(T_SHOE_GEO, T_TD_GEO) == pytest.approx(16.0, abs=1.0)
    assert _kt(T_ISO, T_ISO) == pytest.approx(13.0, abs=1.0)


def test_fig12_A_scales_as_Ttd_over_Ts():
    """The mechanism behind the sign: A ∝ T_td / T_s (absolute degR). Isolated by
    INJECTING fixed gas properties, so only the temperatures move (with auto
    Hall-Yarborough props T_s also drags Z_s / rho_gas_s, adding a few % on top --
    which is why the KT gap is ~14.8% vs the pure-T 13.4%; the sign is unaffected)."""
    from welleng.kick_tolerance.core import constant_A, fahrenheit_to_rankine
    fixed = dict(Z_s=1.10, Z_td=1.10, rho_gas_s=1.5, **CASE1)  # gas props held fixed
    a_geo = constant_A(KickInputs(T_s=T_SHOE_GEO, T_td=T_TD_GEO, **fixed))
    a_iso = constant_A(KickInputs(T_s=T_ISO, T_td=T_ISO, **fixed))
    # T_td is 150 degC in both; only T_s differs -> A ratio == T_s_iso / T_s_geo.
    ratio = fahrenheit_to_rankine(T_ISO) / fahrenheit_to_rankine(T_SHOE_GEO)
    assert a_geo / a_iso == pytest.approx(ratio, rel=1e-9)
    assert a_geo > a_iso


def test_the_default_revision_reads_ABOVE_nassab_by_a_pinned_amount():
    """The tests above pin the FROZEN ``spe-208788`` revision. On the shipped
    default the magnitudes move, and they move the OTHER WAY from SPE-208788.

        SPE-208788 (gas properties PINNED by the paper)   default  -2.7 %
        SPE-202426 here      (properties COMPUTED)        default  +9.1 %

    The difference is the override surface, not the physics: SPE-208788's case
    supplies Z_s / Z_td / rho_gas_s, which bypass the gas backend entirely, so
    the bubble-state pressure basis cannot move the density there. Here nothing
    is overridden, so it can.

    Direction matters and is asserted: against Nassab the default reports MORE
    tolerance than the published dots, 14.44 against ~13 -- outside the +-1 bbl
    that the frozen revision meets. That is the anti-conservative direction, it
    is a consequence of the corrected influx basis rather than a defect, and it
    is pinned here so it cannot drift unnoticed.
    """
    kw = {k: v for k, v in CASE1.items() if k != "model_revision"}

    def kt(t_s, t_td):
        return drill_kick(KickInputs(T_s=t_s, T_td=t_td, **kw)).capacity

    geothermal = kt(T_SHOE_GEO, T_TD_GEO)
    isothermal = kt(T_ISO, T_ISO)

    assert geothermal == pytest.approx(17.024, abs=0.02)
    assert isothermal == pytest.approx(14.435, abs=0.02)

    # the SIGN of the physics claim survives -- geothermal above isothermal, ~15%
    assert geothermal > isothermal
    assert (geothermal - isothermal) / geothermal == pytest.approx(0.152, abs=0.01)

    # and the default sits ABOVE the published dots, unlike the frozen revision
    assert isothermal - 13.0 > 1.0
