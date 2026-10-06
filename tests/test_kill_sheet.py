"""Well-kill worksheet: the classic sheet, and the closed-form hydrostatics."""

import pytest

from welleng.kick_tolerance.kill_sheet import (
    G_PSI_PER_PPG_FT,
    KillSheetInputs,
    MudModel,
    column_pressure,
    kill_sheet,
    string_capacity,
)

BASE = dict(mud_weight_ppg=12.0, tvd_ft=10000.0, sidp_psi=500.0,
            scr_pressure_psi=600.0, pump_output_bbl_per_stroke=0.10,
            string_volume_bbl=160.0, annulus_volume_bbl=400.0,
            scr_rate_spm=30.0, schedule_steps=10)


# --- the classic sheet, which is the reference form ------------------------- #
def test_the_classic_sheet_arithmetic():
    s = kill_sheet(KillSheetInputs(**BASE))
    assert s.kill_mud_weight_ppg == pytest.approx(
        12.0 + 500.0 / (G_PSI_PER_PPG_FT * 10000.0))
    assert s.icp_psi == pytest.approx(1100.0)
    assert s.fcp_psi == pytest.approx(600.0 * s.kill_mud_weight_ppg / 12.0)
    assert s.strokes_to_bit == 1600
    assert s.strokes_bit_to_surface == 4000
    assert s.strokes_total == 5600
    assert s.minutes_to_bit == pytest.approx(1600 / 30.0)


def test_the_schedule_runs_from_icp_to_fcp():
    s = kill_sheet(KillSheetInputs(**BASE))
    assert s.schedule[0].drillpipe_psi == pytest.approx(s.icp_psi)
    assert s.schedule[-1].drillpipe_psi == pytest.approx(s.fcp_psi)
    assert s.schedule[-1].strokes == s.strokes_to_bit
    p = [r.drillpipe_psi for r in s.schedule]
    assert p == sorted(p, reverse=True)


def test_pressure_at_holds_fcp_past_the_bit():
    s = kill_sheet(KillSheetInputs(**BASE))
    assert s.pressure_at(0) == pytest.approx(s.icp_psi)
    assert s.pressure_at(s.strokes_to_bit) == pytest.approx(s.fcp_psi)
    assert s.pressure_at(s.strokes_to_bit * 5) == pytest.approx(s.fcp_psi)


def test_string_capacity_is_the_companion_of_annular_capacity():
    from welleng.kick_tolerance import annular_capacity
    assert string_capacity(4.276) == pytest.approx(4.276 ** 2 / 1029.4)
    assert string_capacity(8.5) == pytest.approx(
        annular_capacity(8.5, 0.0))


# --- it must refuse rather than report an unkillable schedule --------------- #
def test_a_sheet_above_maasp_notes_it_loudly():
    s = kill_sheet(KillSheetInputs(**BASE, sicp_psi=1500.0, maasp_psi=1200.0))
    assert any("ABOVE MAASP" in n for n in s.notes)
    assert s.schedule, "the arithmetic is still reported"


def test_every_row_is_self_consistent_with_pressure_at():
    """A driller reads the printed TABLE. Its two columns must be the same
    point on the line -- they were computed from different quantities (a
    rounded stroke count and an un-rounded fraction) and differed by ~0.7% of
    a step."""
    for method in ("classic", "analytical"):
        s = kill_sheet(KillSheetInputs(**BASE), method=method)
        for row in s.schedule:
            assert row.drillpipe_psi == pytest.approx(
                s.pressure_at(row.strokes), rel=1e-9), (method, row.strokes)


def test_a_sheet_below_maasp_is_fine():
    s = kill_sheet(KillSheetInputs(**BASE, sicp_psi=900.0, maasp_psi=1200.0))
    assert s.icp_psi > 0


@pytest.mark.parametrize("bad", [
    {"mud_weight_ppg": 0.0}, {"tvd_ft": 0.0}, {"sidp_psi": -1.0},
    {"pump_output_bbl_per_stroke": 0.0}, {"string_volume_bbl": 0.0},
])
def test_non_physical_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        kill_sheet(KillSheetInputs(**{**BASE, **bad}))


def test_missing_annulus_volume_says_the_influx_is_still_in_the_hole():
    d = {k: v for k, v in BASE.items() if k != "annulus_volume_bbl"}
    s = kill_sheet(KillSheetInputs(**d))
    assert s.strokes_total is None
    assert any("not out of the hole" in n for n in s.notes)


# --- the closed form -------------------------------------------------------- #
def test_column_pressure_sympy_derivation():
    """The defining ODE, solved symbolically, with a zero residual.

    dP/dz = g.rho[1 + c(P - Pref) - alpha(T(z) - Tref)] is LINEAR in P, so it
    has an exact solution; this is the durable record that the implemented
    form is that solution and not an approximation of it.
    """
    sp = pytest.importorskip("sympy")
    s, A, B, C = sp.symbols("s A B C", positive=True)
    u = sp.Function("u")
    exact = sp.dsolve(sp.Eq(u(s).diff(s), A * u(s) + B - C * s),
                      u(s), ics={u(0): 0}).rhs
    implemented = (C / A - B) / A * (1 - sp.exp(A * s)) + (C / A) * s
    assert sp.simplify(exact - implemented) == 0
    assert sp.simplify(implemented.diff(s)
                       - (A * implemented + B - C * s)) == 0


def test_the_incompressible_limit_is_exact_not_approached():
    """c = 0 takes a separate branch: the exponential form is 0/0 there, and a
    tiny-c approximation would be a different function."""
    m = MudModel(12.0)
    assert m.incompressible_isothermal
    assert column_pressure(m, 0.0, 10000.0, 0.0) == pytest.approx(
        G_PSI_PER_PPG_FT * 12.0 * 10000.0, rel=1e-12)


def test_the_compressible_form_is_continuous_into_the_incompressible_one():
    """The departure from the incompressible column must be LINEAR in c, all
    the way down.

    That is the real property, and it is a sharper test than any magnitude
    threshold: the naive algebraic form carries 1/c**2 against a bracket going
    to zero, and its cancellation shows up as a FLOOR that stops scaling.
    (The first version of this test asserted a floor below 1e-6 psi and failed
    — the residual at c = 1e-12 is 1.95e-5 psi, which is the physics, not the
    arithmetic. Half c (g rho s)^2 to two significant figures.)
    """
    plain = column_pressure(MudModel(12.0), 0.0, 10000.0, 14.7)
    errs = {c: abs(column_pressure(MudModel(12.0, c), 0.0, 10000.0, 14.7)
                   - plain)
            for c in (1e-8, 1e-10, 1e-12, 1e-14)}
    for small, large in ((1e-10, 1e-8), (1e-12, 1e-10), (1e-14, 1e-12)):
        assert errs[small] / errs[large] == pytest.approx(0.01, rel=0.02), (
            f"c={small}: departure stopped scaling — cancellation floor")
    # and the magnitude is the second-order term, not something else
    assert errs[1e-12] == pytest.approx(0.5 * 1e-12 * plain ** 2, rel=0.05)


def test_compressibility_raises_the_pressure_and_heat_lowers_it():
    """Signs, on their own, because they partly cancel and a sign error hides."""
    base = column_pressure(MudModel(12.0), 0.0, 10000.0, 14.7)
    comp = column_pressure(MudModel(12.0, 3e-6, 0.0), 0.0, 10000.0, 14.7)
    warm = column_pressure(MudModel(12.0, 0.0, 2.5e-4), 0.0, 10000.0, 14.7,
                           60.0, 0.012)
    assert comp > base, "compression must make the column heavier"
    assert warm < base, "thermal expansion must make it lighter"


def test_a_segment_is_the_sum_of_its_parts():
    """Splitting a column at any depth must not change its base pressure."""
    m = MudModel(12.0, 3e-6, 2.5e-4)
    kw = dict(surface_temp_degf=60.0, geothermal_gradient_degf_per_ft=0.012)
    whole = column_pressure(m, 0.0, 10000.0, 14.7, **kw)
    mid = column_pressure(m, 0.0, 4000.0, 14.7, **kw)
    split = column_pressure(m, 4000.0, 10000.0, mid, **kw)
    assert split == pytest.approx(whole, rel=1e-12)


def test_a_deep_column_differs_from_the_naive_product_by_tens_of_psi():
    """The size of the thing being neglected, stated rather than implied."""
    naive = G_PSI_PER_PPG_FT * 12.0 * 15000.0
    real = column_pressure(MudModel(12.0, 3e-6, 2.5e-4), 0.0, 15000.0, 14.7,
                           60.0, 0.012)
    assert 40.0 < abs(real - naive) < 200.0


# --- parity: the gate stated before the code was written -------------------- #
def test_analytical_reproduces_the_classic_sheet_when_the_mud_is_ideal():
    """With zero compressibility and zero expansion the exact hydrostatics ARE
    the classic straight line. Anything else means the two disagree about the
    same physics."""
    inp = KillSheetInputs(**BASE)
    c = kill_sheet(inp, method="classic")
    a = kill_sheet(inp, method="analytical")
    for rc, ra in zip(c.schedule, a.schedule):
        assert ra.drillpipe_psi == pytest.approx(rc.drillpipe_psi, rel=1e-12)


def test_a_real_mud_moves_the_schedule():
    """...and the parity test above is only meaningful if this one differs."""
    ideal = kill_sheet(KillSheetInputs(**BASE), method="analytical")
    real = kill_sheet(
        KillSheetInputs(**BASE, mud=MudModel(12.0, 3e-6, 2.5e-4),
                        geothermal_gradient_degf_per_ft=0.012),
        method="analytical")
    diffs = [abs(a.drillpipe_psi - b.drillpipe_psi)
             for a, b in zip(ideal.schedule, real.schedule)]
    assert max(diffs) > 1.0, "the mud model changed nothing — vacuous parity"


def test_a_mismatched_mud_model_is_refused():
    with pytest.raises(ValueError, match="does not match"):
        kill_sheet(KillSheetInputs(**BASE, mud=MudModel(9.0)),
                   method="analytical")


def test_a_vertical_front_assumption_is_declared():
    s = kill_sheet(KillSheetInputs(**BASE), method="analytical")
    assert any("VERTICAL" in n for n in s.notes)


def test_a_supplied_front_map_is_used():
    """A deviated string reaches a given TVD after less pumping."""
    dev = kill_sheet(
        KillSheetInputs(**BASE, front_tvd=lambda f: 10000.0 * f ** 0.5),
        method="analytical")
    vert = kill_sheet(KillSheetInputs(**BASE), method="analytical")
    assert not any("VERTICAL" in n for n in dev.notes)
    mid = len(dev.schedule) // 2
    assert dev.schedule[mid].drillpipe_psi != pytest.approx(
        vert.schedule[mid].drillpipe_psi)


def test_an_unknown_method_is_refused():
    with pytest.raises(ValueError, match="classic"):
        kill_sheet(KillSheetInputs(**BASE), method="exact")


# --- validation against a WORKED IWCF sheet --------------------------------- #
# Surface-BOP kill sheet, vertical well, metric — an IWCF worksheet filled in
# with its arithmetic cached, held in the reference library. Cited as the form,
# not reproduced: only the case's numbers appear here.
#
# ⭐ This is the scarce kind of check. The method is in any well-control text;
# what cannot be manufactured is someone else's completed arithmetic.
_BAR = 14.503773773        # psi per bar
_FT = 3.280839895          # ft per m
_BBL = 158.987294928       # litres per bbl
_PPG = 8.345404452         # ppg per kg/l

_IWCF = dict(
    mud_weight_ppg=1.52 * _PPG,          # 1.52 kg/l
    tvd_ft=3000.0 * _FT,                 # 3000 m
    sidp_psi=22.0 * _BAR,
    scr_pressure_psi=42.0 * _BAR,        # dynamic pressure loss at 50 spm
    pump_output_bbl_per_stroke=35.0 / _BBL,
    string_volume_bbl=24328.0 / _BBL,
    annulus_volume_bbl=68078.52 / _BBL,
    sicp_psi=37.0 * _BAR,
    maasp_psi=35.0 * _BAR,
    scr_rate_spm=50.0,
    schedule_steps=1,
)


def test_iwcf_worked_sheet_icp_and_strokes_are_exact():
    s = kill_sheet(KillSheetInputs(**_IWCF))
    assert s.icp_psi / _BAR == pytest.approx(64.0, abs=1e-6)
    assert s.strokes_to_bit == 695              # sheet: 695.086
    assert s.strokes_bit_to_surface == 1945     # sheet: 1945.10
    assert s.minutes_to_bit == pytest.approx(13.9017, abs=2e-3)


def test_iwcf_worked_sheet_fcp_is_exact_on_the_sheets_own_kill_weight():
    """A rig mixes to a weight it can actually weigh up, and this sheet rounds
    1.5948 kg/l to 1.60 before computing FCP. Given that same figure, welleng
    reproduces the sheet to five decimal places."""
    s = kill_sheet(KillSheetInputs(**_IWCF, kill_mud_weight_ppg=1.60 * _PPG))
    assert s.fcp_psi / _BAR == pytest.approx(44.21053, abs=1e-5)


def test_iwcf_worked_sheet_kill_weight_differs_only_by_the_gradient_constant():
    """The ONE quantity that does not match, and it matches its explanation.

    The sheet weights columns with the metric 10.2 (= 0.0519481 psi/ft/ppg,
    exact standard gravity); this engine uses 0.0521 throughout so that a
    pressure here reproduces the column weight the rest of the module applies.
    The SIDP term therefore differs by 0.29%, and nothing else does — ICP, FCP,
    strokes and times are all independent of the constant.
    """
    s = kill_sheet(KillSheetInputs(**_IWCF))
    got = s.kill_mud_weight_ppg / _PPG
    assert got == pytest.approx(1.5948, abs=5e-4)        # the sheet's own value
    sidp_term = got - 1.52
    sheet_term = 1.5948 - 1.52
    assert sidp_term / sheet_term == pytest.approx(
        0.0519481 / G_PSI_PER_PPG_FT, rel=2e-3)


def test_the_schedule_is_linear_in_strokes_like_the_sheet():
    """The sheet steps 64.000 -> 61.153 -> 58.306 ... per 100 strokes, a
    constant 2.847 bar. Same line, same slope."""
    s = kill_sheet(KillSheetInputs(**{**_IWCF, "schedule_steps": 10},
                                   kill_mud_weight_ppg=1.60 * _PPG))
    per_100 = (s.icp_psi - s.fcp_psi) / _BAR / (s.strokes_to_bit / 100.0)
    assert per_100 == pytest.approx(2.847, abs=2e-3)
    for a, b in zip(s.schedule, s.schedule[1:]):
        step = (a.drillpipe_psi - b.drillpipe_psi) / _BAR
        assert step == pytest.approx(per_100 * (b.strokes - a.strokes) / 100.0,
                                     rel=1e-6)


def test_a_sheet_above_maasp_still_reports_its_arithmetic():
    """⚠️ REGRESSION. This guard was fatal, and it refused the worked IWCF
    sheet above — whose SICP (37 bar) is over its initial MAASP (35 bar), and
    which is filled in anyway. A worksheet that will not print the numbers for
    the case the driller is actually in is not a safety feature.
    """
    s = kill_sheet(KillSheetInputs(**_IWCF))
    assert s.icp_psi > 0 and s.schedule
    assert any("ABOVE MAASP" in n for n in s.notes)


def test_strict_still_refuses_for_a_caller_that_wants_it():
    with pytest.raises(ValueError, match="ABOVE MAASP"):
        kill_sheet(KillSheetInputs(**{**_IWCF, "strict": True}))


def test_a_kill_weight_below_the_current_mud_is_refused():
    with pytest.raises(ValueError, match="below the current mud weight"):
        kill_sheet(KillSheetInputs(**_IWCF, kill_mud_weight_ppg=1.0 * _PPG))
