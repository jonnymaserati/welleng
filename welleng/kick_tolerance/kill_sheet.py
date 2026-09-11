"""Well-kill worksheet -- the CIRCULATING half of well control.

:mod:`welleng.kick_tolerance` answers *can this influx be shut in and
circulated out without breaking down the shoe*. This module answers the next
question: **what pressures does the driller hold while doing it.** Kill mud
weight, initial and final circulating pressure, the drill-pipe pressure
schedule between them, and the stroke counts that time it.

Relationship to the rest of the module
--------------------------------------
The two halves share one input, the shut-in drill-pipe pressure, and they are
not interchangeable:

* ``migration.MigrationStep`` reports ``sidp_psi`` / ``sicp_psi`` for a bubble
  MIGRATING in a SHUT-IN well. Nothing is being pumped.
* A kill sheet describes the well being CIRCULATED. The annulus pressure is
  held by the choke, not by a static gas column.

⚠️ **A kill sheet does not establish that the kill is possible.** It computes
the schedule assuming it is. Whether the shoe survives the circulation is the
kick-tolerance question, and :func:`kill_sheet` will refuse to return a sheet
whose surface pressure is already above MAASP -- but a complete feasibility
answer comes from :func:`~welleng.kick_tolerance.evaluate_envelope`, not from
here.

What is exact and what is a convention
--------------------------------------
Two of the standard relations are **approximations taught as formulae**, and a
kill sheet that does not say so invites them to be read as physics:

``FCP = SCRP * KMW / OMW``
    Assumes circulating pressure loss scales **linearly with density**. That
    holds for fully turbulent flow, where the friction term carries density to
    the first power; it does not hold in laminar flow, where viscosity and
    yield point dominate and the ratio understates the change. Standard
    practice, and the basis on which every IWCF sheet is filled in.

The drill-pipe pressure schedule
    A straight line from ICP to FCP in STROKES is a **procedure**, not a
    prediction of the true pressure. The real curve is not linear, because the
    friction contribution changes with the kill-mud fraction in the string, and
    the standard schedule is a safe simplification that both errs on the high
    side and is followable by hand. Reported as a schedule for that reason.

Everything else -- kill mud weight, ICP, capacities, strokes -- is arithmetic.

Units
-----
Field units throughout, as with the rest of this module: ppg, psi, ft, bbl,
bbl/stroke. Column weights use the module's own ``G_PSI_PER_PPG_FT`` (0.0521),
**not** 0.052, so a pressure computed here reproduces the column weight the
rest of the engine applies. The difference is ~0.2% on a typical kill mud
weight and it does not divide out of an absolute pressure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import math

from .migration import G_PSI_PER_PPG_FT

__all__ = [
    "KillSheetInputs",
    "KillSheetResult",
    "MudModel",
    "PumpSchedule",
    "column_pressure",
    "kill_sheet",
    "string_capacity",
]


@dataclass(frozen=True)
class MudModel:
    """A mud whose density varies with pressure and temperature.

    ``rho(P, T) = rho_ref * [1 + c (P - P_ref) - alpha (T - T_ref)]``

    ``c`` is isothermal compressibility [1/psi] and ``alpha`` the volumetric
    thermal expansion [1/degF]. Both default to ZERO, which reduces every
    formula here to the incompressible isothermal one a classic sheet uses --
    so the effect of turning them on is always visible as a difference from the
    sheet, never hidden inside it.

    Typical field values: a water-based mud is ~3e-6 /psi and ~2.5e-4 /degF; an
    oil-based mud is roughly twice as compressible and twice as expansive.
    Defaults stay at zero rather than taking a literature value, because a
    number that changes the answer should be supplied by the caller who can
    say where it came from.
    """

    rho_ref_ppg: float
    compressibility_per_psi: float = 0.0
    thermal_expansion_per_degf: float = 0.0
    #: Reference conditions the density is quoted AT -- surface, normally.
    p_ref_psi: float = 14.7
    t_ref_degf: float = 60.0

    @property
    def incompressible_isothermal(self) -> bool:
        return (self.compressibility_per_psi == 0.0
                and self.thermal_expansion_per_degf == 0.0)


def column_pressure(
    mud: MudModel,
    top_tvd_ft: float,
    base_tvd_ft: float,
    p_top_psi: float,
    surface_temp_degf: float = 60.0,
    geothermal_gradient_degf_per_ft: float = 0.0,
) -> float:
    r"""Pressure at ``base_tvd_ft`` given ``p_top_psi`` at ``top_tvd_ft``.

    **Closed form, not integrated numerically.** With a linear compressibility
    and a linear geothermal gradient, hydrostatic equilibrium

    .. math::
        \frac{dP}{dz} = g\,\rho_{ref}\left[1 + c(P-P_{ref})
                          - \alpha(T(z)-T_{ref})\right]

    is a first-order LINEAR ODE in :math:`P(z)`, so it has an exact solution.
    Measuring depth ``s`` from the top of the segment and writing
    :math:`A = g\rho c`, :math:`B = g\rho[1-\alpha(T_{top}-T_{ref})]`,
    :math:`C = g\rho\alpha\,G`:

    .. math::
        \Delta P(s) = \frac{C/A - B}{A}\left(1 - e^{As}\right)
                       + \frac{C}{A}s

    Verified symbolically: residual against the defining ODE is exactly zero,
    and as :math:`c\to0` it reduces to
    :math:`B s - C s^{2}/2`, the linear-thermal column. See
    ``tests/test_kill_sheet.py::test_column_pressure_sympy_derivation``.

    ⚠️ **Why not just multiply.** ``0.052 * ppg * ft`` assumes the density at
    the bottom of a 15,000 ft column is the density measured in the pit. It is
    not: pressure compacts the mud and temperature expands it, and the two
    partly cancel — which is why the error is small and why it is easy to
    assume it is zero. It is a few tens of psi on a deep well, in a calculation
    whose whole purpose is holding bottom-hole pressure constant.
    """
    if base_tvd_ft < top_tvd_ft:
        raise ValueError("base must be at or below top")
    s_ft = base_tvd_ft - top_tvd_ft
    if s_ft == 0.0:
        return p_top_psi

    g_rho = G_PSI_PER_PPG_FT * mud.rho_ref_ppg          # psi/ft at reference
    t_top = surface_temp_degf + geothermal_gradient_degf_per_ft * top_tvd_ft
    A = g_rho * mud.compressibility_per_psi
    B = g_rho * (1.0 + mud.compressibility_per_psi
                 * (p_top_psi - mud.p_ref_psi)
                 - mud.thermal_expansion_per_degf * (t_top - mud.t_ref_degf))
    C = g_rho * mud.thermal_expansion_per_degf * geothermal_gradient_degf_per_ft

    # RECONDITIONED, and the reason matters. Written as
    #     (C/A - B)/A * (1 - exp(A s)) + (C/A) s
    # this is algebraically exact and numerically useless as c -> 0: 1/A^2
    # reaches 1e24 while the bracket goes to zero, and the cancellation left a
    # 3e-5 psi floor that no amount of tightening removes. Factoring the two
    # removable singularities out gives ONE expression valid at every c,
    # including exactly zero, with no branch to fall down:
    #     dP = B s phi(x) - C s^2 psi(x),     x = A s
    #     phi = expm1(x)/x        -> 1      psi = (expm1(x) - x)/x^2 -> 1/2
    # and at x = 0 that IS B s - C s^2 / 2, the incompressible column.
    x = A * s_ft
    return p_top_psi + B * s_ft * _phi(x) - C * s_ft ** 2 * _psi(x)


def _phi(x: float) -> float:
    """``expm1(x)/x``, continued to ``1`` at ``x = 0``."""
    if abs(x) < 1e-8:
        return 1.0 + x / 2.0 + x * x / 6.0
    return math.expm1(x) / x


def _psi(x: float) -> float:
    """``(expm1(x) - x)/x**2``, continued to ``1/2`` at ``x = 0``.

    The series is used well before ``x`` reaches zero: the closed form loses
    the whole answer to cancellation once ``expm1(x)`` and ``x`` agree to
    machine precision, which happens around ``1e-5``.
    """
    if abs(x) < 1e-4:
        return 0.5 + x / 6.0 + x * x / 24.0
    return (math.expm1(x) - x) / (x * x)


def string_capacity(id_in: float) -> float:
    """Internal capacity [bbl/ft] of a bore of inside diameter ``id_in``.

    The companion of :func:`~welleng.kick_tolerance.annular_capacity`, which
    takes the space OUTSIDE a string.
    """
    if id_in <= 0.0:
        raise ValueError(f"inside diameter must be positive, got {id_in}")
    return id_in ** 2 / 1029.4


@dataclass(frozen=True)
class PumpSchedule:
    """One row of the drill-pipe pressure schedule."""

    strokes: int
    volume_bbl: float
    drillpipe_psi: float
    #: Fraction of the surface-to-bit displacement completed, 0.0 -> 1.0.
    fraction: float


@dataclass(frozen=True)
class KillSheetInputs:
    """Pre-recorded and shut-in data, as a kill sheet is filled in.

    ``scr_pressure_psi`` is the slow-circulating-rate pressure measured at
    ``scr_rate_spm`` in the CURRENT mud, and it must be a measured value: it is
    the only term carrying the well's actual friction, and substituting a
    calculated one silently replaces a measurement with a model.
    """

    #: Original (current) mud weight [ppg].
    mud_weight_ppg: float
    #: True vertical depth of the hole [ft].
    tvd_ft: float
    #: Shut-in drill-pipe pressure [psi].
    sidp_psi: float
    #: Measured slow-circulating-rate pressure [psi] at ``scr_rate_spm``.
    scr_pressure_psi: float
    #: Pump displacement [bbl/stroke].
    pump_output_bbl_per_stroke: float
    #: Drill-string internal volume, surface to bit [bbl].
    string_volume_bbl: float
    #: Annular volume, bit to surface [bbl]. Optional: without it the sheet
    #: reports the surface-to-bit half only, which is the half the drill-pipe
    #: schedule covers.
    annulus_volume_bbl: Optional[float] = None
    #: Shut-in casing pressure [psi], recorded but not used in the schedule.
    sicp_psi: Optional[float] = None
    #: Maximum allowable annular surface pressure [psi]. When given and
    #: exceeded by ``sicp_psi``, the sheet carries a LOUD note -- and raises
    #: only under ``strict``.
    maasp_psi: Optional[float] = None
    #: Refuse rather than warn when the shut-in casing pressure is already
    #: above MAASP. Default False, because a real well-control worksheet is
    #: filled in for exactly that case and the driller needs the numbers.
    strict: bool = False
    #: Kill mud weight [ppg] to USE, overriding the computed one. A rig mixes
    #: to what it can actually weigh up, and a worked sheet is filled in with
    #: the rounded figure -- so reproducing one needs this.
    kill_mud_weight_ppg: Optional[float] = None
    #: Pump rate [strokes/min] the SCR pressure was measured at; used only to
    #: report times.
    scr_rate_spm: Optional[float] = None
    #: Rows in the reported pressure schedule (excluding the ICP row).
    schedule_steps: int = 10
    #: Mud model for the CURRENT mud. ``None`` = incompressible isothermal,
    #: i.e. exactly what the classic sheet assumes.
    mud: Optional[MudModel] = None
    surface_temp_degf: float = 60.0
    geothermal_gradient_degf_per_ft: float = 0.0
    #: Front TVD [ft] as a function of the fraction of the string displaced.
    #: Default assumes a VERTICAL string -- in a deviated well the kill mud
    #: front reaches a given TVD after less pumping than this implies, and the
    #: caller must supply the well's own MD->TVD map.
    front_tvd: Optional[object] = None


@dataclass(frozen=True)
class KillSheetResult:
    """A filled kill sheet."""

    kill_mud_weight_ppg: float
    #: Initial circulating pressure [psi] = SCRP + SIDP.
    icp_psi: float
    #: Final circulating pressure [psi] = SCRP * KMW / OMW.
    fcp_psi: float
    strokes_to_bit: int
    strokes_bit_to_surface: Optional[int]
    strokes_total: Optional[int]
    minutes_to_bit: Optional[float]
    minutes_total: Optional[float]
    #: Drill-pipe pressure vs strokes, ICP at 0 to FCP at ``strokes_to_bit``.
    schedule: List[PumpSchedule] = field(default_factory=list)
    #: Non-fatal observations a driller should see on the sheet.
    notes: List[str] = field(default_factory=list)

    def pressure_at(self, strokes: float) -> float:
        """Scheduled drill-pipe pressure [psi] at ``strokes`` pumped.

        Linear in strokes between ICP and FCP, then held at FCP. See the module
        docstring: this is the schedule to follow, not a prediction of the
        pressure the well will show.
        """
        if self.strokes_to_bit <= 0:
            return self.fcp_psi
        f = min(max(float(strokes) / self.strokes_to_bit, 0.0), 1.0)
        return self.icp_psi + (self.fcp_psi - self.icp_psi) * f


def _analytical_schedule(inp, kmw, icp, fcp, strokes_to_bit, steps, notes):
    """Drill-pipe pressure that holds BHP constant, front by front.

    At every step the string is kill mud above the front and original mud
    below it, and the drill-pipe pressure is

        P_dp = BHP - (kill column) - (original column) + friction

    with each column from the exact closed form. The classic sheet replaces
    the two columns with a straight line, which is right only when density is
    constant with depth.
    """
    old = inp.mud or MudModel(inp.mud_weight_ppg)
    if abs(old.rho_ref_ppg - inp.mud_weight_ppg) > 1e-9:
        raise ValueError(
            f"mud model density {old.rho_ref_ppg} does not match the sheet's "
            f"mud weight {inp.mud_weight_ppg}"
        )
    kill = MudModel(kmw, old.compressibility_per_psi,
                    old.thermal_expansion_per_degf,
                    old.p_ref_psi, old.t_ref_degf)
    ts, gg = inp.surface_temp_degf, inp.geothermal_gradient_degf_per_ft

    # BHP is what the shut-in well is telling us, computed with the SAME
    # column model -- otherwise the exactness is spent reconciling two
    # different hydrostatics rather than on the answer.
    bhp = column_pressure(old, 0.0, inp.tvd_ft, inp.sidp_psi, ts, gg)

    if inp.front_tvd is None:
        notes.append(
            "Front depth assumed proportional to displaced volume (VERTICAL "
            "string). In a deviated well supply front_tvd: the kill mud "
            "reaches a given TVD after less pumping than this implies."
        )

    rows = []
    for k in range(steps + 1):
        n = int(round(strokes_to_bit * k / steps))
        f = n / strokes_to_bit if strokes_to_bit else 1.0
        z_f = (inp.front_tvd(f) if inp.front_tvd is not None
               else inp.tvd_ft * f)
        z_f = min(max(float(z_f), 0.0), inp.tvd_ft)
        p_front = column_pressure(kill, 0.0, z_f, 0.0, ts, gg)
        p_shoe = column_pressure(old, z_f, inp.tvd_ft, p_front, ts, gg)
        friction = inp.scr_pressure_psi + (fcp - inp.scr_pressure_psi) * f
        rows.append(PumpSchedule(
            strokes=n,
            volume_bbl=inp.pump_output_bbl_per_stroke * n,
            drillpipe_psi=bhp - p_shoe + friction,
            fraction=f,
        ))
    return rows


def kill_sheet(inp: KillSheetInputs, method: str = "classic") -> KillSheetResult:
    """Fill a well-kill worksheet from pre-recorded and shut-in data.

    ``method="classic"`` (default)
        The IWCF sheet as taught and as filled in on the rig: a straight line
        in strokes from ICP to FCP. This is the **reference form** and it is
        what a worked sheet can be checked against.

    ``method="analytical"``
        The same sheet with the HYDROSTATICS solved exactly, from
        :func:`column_pressure`, for a mud whose density varies with pressure
        and temperature. The drill-pipe pressure at each step is what actually
        holds bottom-hole pressure constant with the kill-mud front at that
        depth, rather than a line drawn between two endpoints.

        ⚠️ **The friction term is still the classic density ratio.** One
        measured SCR pressure cannot be decomposed into its laminar and
        turbulent parts, so scaling it is the only defensible thing to do with
        it; solving the hydrostatics exactly does not make the friction exact.
        Stated because the combination is easy to mistake for a full model.

        ⚠️ **Mud compression is in the PRESSURE, not in the volume balance.**
        Compressed kill mud occupies slightly less of the string than was
        pumped, so the front sits a little shallower than the stroke count
        implies. That is second order against the pressure effect and is NOT
        modelled -- named rather than silently folded in.

    With a ``MudModel`` of zero compressibility and zero expansion the two
    methods agree to floating point, which is the parity gate in
    ``tests/test_kill_sheet.py``.

    Raises
    ------
    ValueError
        On a non-physical input, or when ``maasp_psi`` is given and the
        recorded shut-in casing pressure already exceeds it -- that well is not
        killable on this schedule and a sheet saying otherwise is worse than no
        sheet.

    Examples
    --------
    >>> from welleng.kick_tolerance import KillSheetInputs, kill_sheet
    >>> s = kill_sheet(KillSheetInputs(
    ...     mud_weight_ppg=12.0, tvd_ft=10000.0, sidp_psi=500.0,
    ...     scr_pressure_psi=600.0, pump_output_bbl_per_stroke=0.10,
    ...     string_volume_bbl=160.0))
    >>> round(s.kill_mud_weight_ppg, 2)
    12.96
    >>> round(s.icp_psi, 1), round(s.fcp_psi, 1)
    (1100.0, 648.0)
    >>> s.strokes_to_bit
    1600
    """
    if inp.mud_weight_ppg <= 0.0:
        raise ValueError("mud weight must be positive")
    if inp.tvd_ft <= 0.0:
        raise ValueError("TVD must be positive")
    if inp.sidp_psi < 0.0:
        raise ValueError("shut-in drill-pipe pressure cannot be negative")
    if inp.scr_pressure_psi < 0.0:
        raise ValueError("slow-circulating-rate pressure cannot be negative")
    if inp.pump_output_bbl_per_stroke <= 0.0:
        raise ValueError("pump output must be positive")
    if inp.string_volume_bbl <= 0.0:
        raise ValueError("string volume must be positive")

    notes: List[str] = []

    over_maasp = (inp.maasp_psi is not None and inp.sicp_psi is not None
                  and inp.sicp_psi > inp.maasp_psi)
    if over_maasp:
        msg = (
            f"shut-in casing pressure {inp.sicp_psi:.0f} psi is ABOVE MAASP "
            f"{inp.maasp_psi:.0f} psi: the shoe is already at risk before "
            "circulation starts. The schedule below is still the arithmetic, "
            "but the well-control decision is not this sheet's to make."
        )
        if inp.strict:
            raise ValueError(msg)
        notes.append(msg)

    # Kill mud weight: the density whose static column balances pore pressure.
    # An explicit value wins -- a rig mixes to what it can weigh up, and the
    # worked sheets are filled in with the rounded figure.
    kmw = inp.kill_mud_weight_ppg
    if kmw is None:
        kmw = inp.mud_weight_ppg + inp.sidp_psi / (
            G_PSI_PER_PPG_FT * inp.tvd_ft)
    elif kmw < inp.mud_weight_ppg:
        raise ValueError(
            f"kill mud weight {kmw} is below the current mud weight "
            f"{inp.mud_weight_ppg}"
        )

    # ICP is exact: the friction the pump must overcome plus the underbalance
    # the mud column is short by.
    icp = inp.scr_pressure_psi + inp.sidp_psi

    # FCP scales the MEASURED friction by the density ratio -- an approximation,
    # see the module docstring.
    fcp = inp.scr_pressure_psi * kmw / inp.mud_weight_ppg

    strokes_to_bit = int(round(
        inp.string_volume_bbl / inp.pump_output_bbl_per_stroke))
    strokes_up = total = None
    if inp.annulus_volume_bbl is not None:
        if inp.annulus_volume_bbl <= 0.0:
            raise ValueError("annulus volume must be positive when given")
        strokes_up = int(round(
            inp.annulus_volume_bbl / inp.pump_output_bbl_per_stroke))
        total = strokes_to_bit + strokes_up
    else:
        notes.append(
            "No annulus volume given: the sheet covers surface-to-bit only. "
            "The influx is not out of the hole at the end of this schedule."
        )

    mins_to_bit = mins_total = None
    if inp.scr_rate_spm:
        mins_to_bit = strokes_to_bit / inp.scr_rate_spm
        if total is not None:
            mins_total = total / inp.scr_rate_spm

    if method not in ("classic", "analytical"):
        raise ValueError(
            f"method must be 'classic' or 'analytical', got {method!r}")

    steps = max(int(inp.schedule_steps), 1)
    if method == "classic":
        # The row's pressure is derived from the row's OWN stroke count, not
        # from the un-rounded fraction that produced it: a driller reads the
        # printed table, so the two columns of a row have to be the same point
        # on the line. They differed by ~0.7% of a step before.
        schedule = []
        for k in range(steps + 1):
            n = int(round(strokes_to_bit * k / steps))
            f = n / strokes_to_bit if strokes_to_bit else 1.0
            schedule.append(PumpSchedule(
                strokes=n,
                volume_bbl=inp.pump_output_bbl_per_stroke * n,
                drillpipe_psi=icp + (fcp - icp) * f,
                fraction=f,
            ))
    else:
        schedule = _analytical_schedule(inp, kmw, icp, fcp, strokes_to_bit,
                                        steps, notes)

    if fcp > icp:
        # Only when the kill mud's extra friction outweighs the underbalance it
        # removes -- possible with a small SIDP and a high SCR pressure.
        notes.append(
            f"FCP ({fcp:.0f} psi) is ABOVE ICP ({icp:.0f} psi): the drill-pipe "
            "pressure schedule RISES. Check the SCR pressure and SIDP before "
            "using this sheet — the usual sheet falls."
        )
    if inp.sidp_psi == 0.0:
        notes.append(
            "SIDP is zero: kill mud weight equals the current mud weight, so "
            "this sheet describes circulating at the existing weight."
        )

    return KillSheetResult(
        kill_mud_weight_ppg=kmw,
        icp_psi=icp,
        fcp_psi=fcp,
        strokes_to_bit=strokes_to_bit,
        strokes_bit_to_surface=strokes_up,
        strokes_total=total,
        minutes_to_bit=mins_to_bit,
        minutes_total=mins_total,
        schedule=schedule,
        notes=notes,
    )
