"""Inflow-performance (IPR) scalar reference oracle — classic forms as-published.

This module implements the standard inflow-performance relationships that map a
flowing bottom-hole pressure to a stock-tank liquid rate: the straight-line
(Darcy) PI, Vogel's dimensionless solution-gas-drive curve, Standing's
undersaturated composite, the Fetkovich backpressure/deliverability form and
the Jones laminar-plus-turbulent drawdown split, together with the definitional
test-point back-outs that recover each form's coefficient from a stabilised
well test. Every function is a **scalar reference form** — pure ``float`` in,
``float`` out. It is welleng's open-core *correctness oracle*: the vectorised /
GPU IPR forms in the commercial layer are parity-gated against these functions
(bit-identical, or a documented tolerance where a solve differs in its tail).

Design contract
---------------
* **Pure float in / float out.** No arrays, no fast paths — that is the
  private/api layer's job.
* **Strict SI, no field seam.** Unlike :mod:`welleng.flow.pvt`, the IPR forms
  carry no oilfield-unit correlations: every input and output is already SI, so
  there is no field↔SI conversion anywhere in this module. Rate ``q`` is
  stock-tank volumetric **sm3/s** (standard-condition liquid rate); pressures
  are **Pa absolute**; the productivity index ``J`` is **sm3/s/Pa**. The
  Fetkovich and Jones coefficients are *dimensional* and their units are stated
  per function (``C`` [sm3/s/Pa^(2n)], ``A`` [Pa/(sm3/s)], ``B`` [Pa/(sm3/s)²]).
* **Rate sign convention.** A positive drawdown (``pr > pwf``) gives a positive
  producing rate. The straight-line :func:`darcy_linear` carries a negative
  drawdown straight through to a negative rate (injection); the power-law forms
  (Vogel/Fetkovich/Jones) are producing-well relations — see each docstring for
  its behaviour outside the producing domain.
* **Validity bands warn, never clamp, never raise.** Each function emits a
  :func:`warnings.warn` when evaluated outside its published band (band stated
  in the docstring) and returns the extrapolated value — matching the
  :mod:`welleng.flow.pvt` house pattern.
* **Citation in every docstring:** source SPE# + specific equation.

References
----------
Vogel, J. V. (1968). *Inflow Performance Relationships for Solution-Gas Drive
    Wells.* JPT 20(1): 83-92 (SPE-1476).
Standing, M. B. (1971). *Concerning the Calculation of Inflow Performance of
    Wells Producing from Solution Gas Drive Reservoirs.* JPT 23(9): 1141-1142
    (the undersaturated Vogel extension; the J-anchored composite form).
Fetkovich, M. J. (1973). *The Isochronal Testing of Oil Wells.* SPE-4529,
    48th Annual Fall Meeting.
Jones, L. G., Blount, E. M. & Glaze, O. H. (1976). *Use of Short Term Multiple
    Rate Flow Tests To Predict Performance of Wells Having Turbulence.*
    SPE-6133, 51st Annual Fall Meeting.
"""
from __future__ import annotations

import math
import warnings


# =============================================================================
# Straight-line (undersaturated Darcy inflow)
# =============================================================================
def darcy_linear(pr_pa: float, pwf_pa: float, j_sm3_s_pa: float) -> float:
    """Straight-line PI inflow rate [sm3/s]. Definitional (Darcy PI).

        q = J·(pr − pwf)

    Single-phase (undersaturated, ``pwf >= Pb``) liquid inflow — the
    pseudo-steady radial Darcy solution collapsed into the productivity index
    ``J`` [sm3/s/Pa]. A negative drawdown (``pwf > pr``) returns a negative rate
    (injection); no clamping.
    """
    return j_sm3_s_pa * (pr_pa - pwf_pa)


def pi_from_test(q_sm3_s: float, pr_pa: float, pwf_pa: float) -> float:
    """Productivity index J [sm3/s/Pa] from one stabilised test point.

    Definitional inverse of :func:`darcy_linear`::

        J = q / (pr − pwf)

    Recovers the well PI from a single (q, pwf) measurement at known reservoir
    pressure ``pr`` — the Volve route where no DST is held and J is backed out
    of production data. The exact inverse of :func:`darcy_linear`, so feeding
    the recovered J back reproduces q.
    """
    return q_sm3_s / (pr_pa - pwf_pa)


# =============================================================================
# Vogel (saturated two-phase, solution-gas drive)
# =============================================================================
def vogel(pr_pa: float, pwf_pa: float, q_max_sm3_s: float) -> float:
    """Vogel IPR rate [sm3/s]. Vogel (1968), SPE-1476, the dimensionless curve.

        q/q_max = 1 − 0.2·(pwf/pr) − 0.8·(pwf/pr)²

    Saturated reservoirs (``pr <= Pb``), solution-gas drive; ``q_max`` is the
    absolute open-flow potential (AOF, at ``pwf = 0``). Band: the curve is
    defined for ``pwf/pr`` in [0, 1] (producing domain); a value outside that
    range is warned (the returned rate is extrapolated). The ``pr <= Pb``
    saturation precondition is the caller's to enforce — ``Pb`` is not an
    argument to this form; use :func:`standing_composite` when ``pr > Pb``.
    Classic as-published.
    """
    x = pwf_pa / pr_pa
    if not 0.0 <= x <= 1.0:
        warnings.warn(
            f"Vogel IPR evaluated outside its producing domain "
            f"(pwf/pr={x:.4g} not in [0, 1]); the returned rate is "
            f"extrapolated.",
            stacklevel=2,
        )
    return q_max_sm3_s * (1.0 - 0.2 * x - 0.8 * x * x)


def vogel_q_max(
    q_test_sm3_s: float, pr_pa: float, pwf_test_pa: float
) -> float:
    """Vogel AOF q_max [sm3/s] backed out of one test point.

    The standard test-point normalisation of the Vogel curve (SPE-1476)::

        q_max = q_test / (1 − 0.2·(pwf/pr) − 0.8·(pwf/pr)²)

    Definitional inverse of :func:`vogel`: feeding the recovered ``q_max`` back
    reproduces ``q_test`` at ``pwf_test``. Classic as-published.
    """
    x = pwf_test_pa / pr_pa
    return q_test_sm3_s / (1.0 - 0.2 * x - 0.8 * x * x)


# =============================================================================
# Standing composite (undersaturated above Pb, Vogel below)
# =============================================================================
def standing_composite(
    pr_pa: float, pwf_pa: float, pb_pa: float, j_sm3_s_pa: float
) -> float:
    """Composite IPR rate [sm3/s]. Standing (1971), the J-anchored construction.

    Undersaturated reservoir (``pr > Pb``), two-phase below ``Pb`` — a straight
    line above the bubble point stitched to Vogel's curve below it::

        pwf >= Pb:  q = J·(pr − pwf)
        pwf <  Pb:  q = q_b + (J·Pb/1.8)·[1 − 0.2·(pwf/Pb) − 0.8·(pwf/Pb)²]
        with q_b = J·(pr − Pb)

    ``J`` [sm3/s/Pa] is the undersaturated PI. The two branches are continuous
    and slope-continuous at ``Pb`` by construction (the Vogel branch's slope
    ``−J`` at ``pwf = Pb`` matches the straight line). Requires ``pr >= Pb``
    (warned otherwise — plain :func:`vogel` applies to a saturated reservoir).
    Classic as-published (Standing's 1971 extension of Vogel).
    """
    if pr_pa < pb_pa:
        warnings.warn(
            f"Standing composite IPR requires pr >= Pb "
            f"(pr={pr_pa:.4g} Pa < Pb={pb_pa:.4g} Pa); the reservoir is "
            f"saturated — use vogel(). The returned rate is extrapolated.",
            stacklevel=2,
        )
    if pwf_pa >= pb_pa:
        return j_sm3_s_pa * (pr_pa - pwf_pa)
    q_b = j_sm3_s_pa * (pr_pa - pb_pa)
    x = pwf_pa / pb_pa
    return q_b + (j_sm3_s_pa * pb_pa / 1.8) * (1.0 - 0.2 * x - 0.8 * x * x)


# =============================================================================
# Fetkovich (backpressure / deliverability)
# =============================================================================
def fetkovich(pr_pa: float, pwf_pa: float, c: float, n: float) -> float:
    """Fetkovich backpressure IPR rate [sm3/s]. Fetkovich (1973), SPE-4529.

        q = C·(pr² − pwf²)^n

    ``C`` is DIMENSIONAL [sm3/s/Pa^(2n)] — stated, not hidden. ``n`` in
    [0.5, 1.0] (warned outside; ``n = 1`` laminar, ``n = 0.5`` fully
    turbulent). For a non-physical reverse drawdown (``pwf > pr``) the sign of
    ``pr² − pwf²`` is preserved through the power so the form degrades to a
    negative rate rather than raising on a fractional power of a negative base.
    Classic as-published.
    """
    if not 0.5 <= n <= 1.0:
        warnings.warn(
            f"Fetkovich exponent n={n:.4g} outside its band [0.5, 1.0] "
            f"(n=1 laminar, n=0.5 fully turbulent); the returned rate is "
            f"extrapolated.",
            stacklevel=2,
        )
    dp2 = pr_pa * pr_pa - pwf_pa * pwf_pa
    return c * math.copysign(abs(dp2) ** n, dp2)


def fetkovich_from_tests(
    q1_sm3_s: float,
    pwf1_pa: float,
    q2_sm3_s: float,
    pwf2_pa: float,
    pr_pa: float,
) -> tuple[float, float]:
    """(C, n) from two stabilised flow-after-flow test points. SPE-4529.

    Log-log solve of the backpressure line ``q = C·(pr² − pwf²)^n`` through the
    two points::

        n = [ln q1 − ln q2] / [ln(pr² − pwf1²) − ln(pr² − pwf2²)]
        C = q1 / (pr² − pwf1²)^n

    Definitional on the Fetkovich form: feeding ``(C, n)`` back into
    :func:`fetkovich` reproduces both ``q1`` and ``q2``. The two points must
    have distinct drawdowns and lie in the producing domain (``pwf < pr``).
    """
    dp1 = pr_pa * pr_pa - pwf1_pa * pwf1_pa
    dp2 = pr_pa * pr_pa - pwf2_pa * pwf2_pa
    n = (math.log(q1_sm3_s) - math.log(q2_sm3_s)) / (
        math.log(dp1) - math.log(dp2)
    )
    c = q1_sm3_s / dp1 ** n
    return c, n


# =============================================================================
# Jones (non-Darcy / rate-dependent skin)
# =============================================================================
def jones(
    pr_pa: float, pwf_pa: float, a_pa_per_q: float, b_pa_per_q2: float
) -> float:
    """Jones non-Darcy IPR rate [sm3/s]. Jones, Blount & Glaze (1976), SPE-6133.

    The laminar-plus-turbulent drawdown split::

        pr − pwf = A·q + B·q²

    solved for the positive root in the numerically-stable *rationalised* form::

        q = 2·Δp / (A + sqrt(A² + 4·B·Δp))      with Δp = pr − pwf

    This is algebraically identical to the quadratic root ``[−A +
    sqrt(A²+4·B·Δp)]/(2·B)`` for ``B > 0`` and degrades to ``Δp/A`` (==
    :func:`darcy_linear` with ``J = 1/A``) as ``B → 0`` automatically — no
    division by ``B``, no epsilon switch. ``A`` [Pa/(sm3/s)] is the laminar
    (Darcy) drawdown coefficient, ``B`` [Pa/(sm3/s)²] the turbulent
    (non-Darcy) coefficient — both DIMENSIONAL, stated. Classic as-published.
    """
    dp = pr_pa - pwf_pa
    return 2.0 * dp / (a_pa_per_q + math.sqrt(a_pa_per_q * a_pa_per_q
                                              + 4.0 * b_pa_per_q2 * dp))
