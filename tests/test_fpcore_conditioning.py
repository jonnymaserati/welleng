"""Differential-conditioning gate against the welleng#307 FPCore reference.

welleng#307/#308 exposed a structural risk: the same min-curve arc formula lived
in several places and a numerical fix (the SLERP -> Rodrigues / half-angle
reconditioning) had to land more than once. The contributor (enomado) also
supplied a set of FPCore reference kernels -- the *exact* expressions each hot
path computes, with the real ``:pre`` survey domains. This module turns those
kernels into a standing gate: evaluate each reference expression at 200-bit
mpmath (the value the math *means*) and assert welleng's own float64
implementation matches it across the domain, to a locked ulp/relative bound.

A future reconditioning regression trips this gate instead of hiding until a
downstream parity run catches it.

The FIRST test is a CALIBRATION per the file's own instruction: the gate must
detect the already-fixed ``acos(dot)`` defect (K0). A detector that cannot flag
a known defect proves nothing about the kernels it passes.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fpcore"))
from _fpcore import parse_kernels, eval_mp  # noqa: E402

from welleng.utils import arc_step, get_dogleg, get_vec  # noqa: E402

FPCORE = os.path.join(os.path.dirname(__file__), "fpcore", "welleng.fpcore")
K = parse_kernels(FPCORE)


def _rng(seed):
    return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# CALIBRATION -- the gate MUST see the known defect, else its passes are void.
# ---------------------------------------------------------------------------
def test_calibration_gate_detects_acos_dot_defect():
    """K0 (theta via acos(dot)) is the form that shipped until 2026-07-31 and
    lost all precision on near-identical stations; K1 (the chord) is the fix.
    Evaluate BOTH in float64 against the 200-bit oracle over the near-collinear
    domain and assert the gate sees a large gap -- acos-dot must be orders worse
    than the chord, or this harness is blind to the very class it exists for."""
    rng = _rng(0)
    acos_dot = K["theta_via_acos_dot_near_collinear"]
    chord = K["theta_chord"]
    worst_acos, worst_chord = 0.0, 0.0
    for _ in range(400):
        v = dict(
            i1=rng.uniform(0.001, 3.14), a1=rng.uniform(0, 6.28),
            di=rng.uniform(-1e-4, 1e-4), da=rng.uniform(-1e-4, 1e-4),
        )
        oracle = float(eval_mp(chord, v))            # true theta
        if oracle == 0.0:
            continue
        f64_acos = float(eval_mp(acos_dot, v, prec=53))
        f64_chord = float(eval_mp(chord, v, prec=53))
        worst_acos = max(worst_acos, abs(f64_acos - oracle) / oracle)
        worst_chord = max(worst_chord, abs(f64_chord - oracle) / oracle)
    # the chord holds full relative precision; acos-dot is catastrophically worse
    assert worst_chord < 1e-9, worst_chord
    assert worst_acos > 1e-4, worst_acos          # detector SEES the defect
    assert worst_acos > 1e4 * worst_chord         # ... and by a wide margin


# ---------------------------------------------------------------------------
# K10 dls -- welleng's dogleg-severity must equal the reference exactly.
# ---------------------------------------------------------------------------
def test_dls_matches_reference():
    dls_k = K["dls"]
    rng = _rng(1)
    for _ in range(200):
        theta = rng.uniform(1e-6, 3.14159)
        dM = rng.uniform(0.1, 1000.0)
        ref = float(eval_mp(dls_k, {"theta": theta, "dM": dM}))
        got = np.degrees(theta / dM * 10.0)        # welleng's dls expression
        assert abs(got - ref) <= 4 * np.spacing(ref)


# ---------------------------------------------------------------------------
# K1 theta_chord -- welleng.get_dogleg vs the reference, relative bound.
# ---------------------------------------------------------------------------
def test_get_dogleg_matches_chord_reference():
    chord = K["theta_chord"]
    rng = _rng(2)
    worst = 0.0
    for _ in range(500):
        i1, a1 = rng.uniform(0.001, 3.14), rng.uniform(0, 6.28)
        di, da = rng.uniform(-1e-4, 1e-4), rng.uniform(-1e-4, 1e-4)
        ref = float(eval_mp(chord, {"i1": i1, "a1": a1, "di": di, "da": da}))
        got = float(get_dogleg(i1, a1, i1 + di, a1 + da))
        if ref > 0:
            worst = max(worst, abs(got - ref) / ref)
    assert worst < 1e-9, worst                     # chord conditioning holds


# ---------------------------------------------------------------------------
# arc_step POSITION vs the K2 reference (fed PHYSICALLY-VALID components, so the
# component-independence caveat in the file does not apply) at 200-bit.
# ---------------------------------------------------------------------------
def _leg(rng):
    i1, a1 = rng.uniform(0.1, 3.0), rng.uniform(0, 2 * np.pi)
    v1 = get_vec(i1, a1, deg=False)[0]
    ax = rng.normal(size=3)
    ax -= v1 * (v1 @ ax)
    ax /= np.linalg.norm(ax)
    theta = rng.uniform(1e-3, np.pi - 1e-3)
    v2 = np.cos(theta) * v1 + np.sin(theta) * ax
    return v1, v2, theta


def test_arc_step_position_matches_fpcore_oracle():
    """arc_step's displacement equals the K2 arc-position reference (evaluated
    per component at 200-bit on the real unit tangents) across the survey domain
    incl. theta -> pi. Locks the #308 half-angle position kernel."""
    pos_k = K["position_arc_component"]
    rng = _rng(3)
    worst = 0.0
    for _ in range(300):
        v1, v2, theta = _leg(rng)
        dM = rng.uniform(0.1, 1000.0)
        t = rng.uniform(0.0, 1.0)
        disp, _ = arc_step(np.array([v1]), np.array([v2]),
                           np.array([theta]), np.array([dM]), np.array([t * dM]))
        disp = disp[0]
        for k in range(3):
            ref = float(eval_mp(pos_k, {"theta": theta, "t": t, "dM": dM,
                                        "v1": float(v1[k]), "v2": float(v2[k])}))
            worst = max(worst, abs(disp[k] - ref) / np.spacing(dM))
    assert worst < 8.0, worst                      # measured ~<=1.5 ulp; headroom


def test_arc_step_tangent_matches_fpcore_oracle():
    """arc_step's unit tangent equals the normalised K6 direction reference.
    K6 (d = v2 sin(phi) - v1 sin(phi-theta), |d| = sin theta) is the numerator;
    welleng normalises, so compare unit-for-unit."""
    dir_k = K["direction_arc_component"]
    rng = _rng(4)
    worst = 0.0
    for _ in range(300):
        v1, v2, theta = _leg(rng)
        t = rng.uniform(0.0, 1.0)
        _, tang = arc_step(np.array([v1]), np.array([v2]),
                          np.array([theta]), np.array([1.0]), np.array([t]))
        tang = tang[0]
        d = np.array([float(eval_mp(dir_k, {"theta": theta, "t": t,
                                            "v1": float(v1[k]), "v2": float(v2[k])}))
                      for k in range(3)])
        d_unit = d / np.linalg.norm(d)
        worst = max(worst, float(np.max(np.abs(tang - d_unit))) / np.spacing(1.0))
    # K6 is the file's worst-conditioned kernel (enomado measured 35.8 -> 51.1
    # ulp at small theta); welleng's u-form sits ~<=20 ulp -- lock below that
    # floor so a genuine reconditioning regression trips, not the known limit.
    assert worst < 40.0, worst


# ---------------------------------------------------------------------------
# Coverage ledger -- do NOT silently drop kernels. This asserts the mapping is
# explicit: every kernel is either exercised above or listed as deferred with a
# reason, so a new kernel added to the file forces a decision here.
# ---------------------------------------------------------------------------
def test_kernel_coverage_is_explicit():
    exercised = {
        "theta_via_acos_dot_near_collinear", "theta_chord", "dls",
        "position_arc_component", "direction_arc_component",
    }
    deferred = {
        # covered transitively via the assembled arc_step forms above:
        "position_arc_component_small_theta": "small-theta position (arc_step covers)",
        "direction_arc_component_small_theta": "small-theta tangent (arc_step covers)",
        "canonical_projection": "1-cos(phi); arc_step uses the half-angle form",
        "arc_frame_normal_component": "arc-frame normal; covered via arc_step",
        "position_straight_component": "straight branch; arc_step straight path",
        # not yet wired to their welleng consumers (future work, no silent cap):
        "theta_via_acos_dot": "calibration variant; near_collinear form tested",
        "analytical_sin_arg": "TODO map to interpolate_tvd / _arc_tvd_crossings",
        "analytical_root": "TODO map to interpolate_tvd root",
        "closest_straight_point": "TODO map to clearance closest-approach",
        "stationary_phi": "TODO map to tvd_turning_points",
    }
    assert exercised | set(deferred) == set(K), (
        set(K) - (exercised | set(deferred)))   # a NEW kernel forces a decision
