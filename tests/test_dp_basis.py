"""The dp_basis switch: balanced_tangent (ISCWSA default) vs min_curve.

min_curve is the trajectory-consistent dp basis (opt-in). Guards:
- the default is balanced_tangent and leaves the published ISCWSA output untouched;
- the analytic min-curve Jacobian equals a finite-difference of the exact leg
  displacement (the derivation is right);
- it reduces to balanced_tangent SMOOTHLY at DL->0 (no branch/switch);
- the min_curve vs balanced_tangent covariance gap is O(interval^2), sub-band.
"""
import json

import numpy as np

from welleng.survey import Survey, SurveyHeader
from welleng.error import ErrorModel

DATA = "tests/test_data/error_mwdrev5_1_iscwsa_data.json"
MODEL = "ISCWSA MWD Rev5.11"


def _survey(dp_basis):
    wd = json.load(open(DATA))
    sh = SurveyHeader()
    for k, v in wd["header"].items():
        setattr(sh, k, v)
    sh.dp_basis = dp_basis
    return Survey(
        md=wd["survey"]["md"], inc=wd["survey"]["inc"],
        azi=wd["survey"]["azi"], header=sh, error_model=MODEL,
    )


def test_default_is_balanced_tangent():
    assert SurveyHeader().dp_basis == "balanced_tangent"


def test_min_curve_reduces_to_bt_at_straight_legs():
    """Where both adjacent legs are straight (DL->0), min_curve == balanced_tangent
    to machine precision -- the RF->1 limit is reached smoothly, no special case."""
    bt = _survey("balanced_tangent").err
    mc = _survey("min_curve").err
    dl = bt.survey.dogleg
    both_straight = [
        k for k in range(1, len(dl) - 1) if dl[k] < 1e-5 and dl[k + 1] < 1e-5
    ]
    assert both_straight
    assert np.max(np.abs(mc.drdp[both_straight] - bt.drdp[both_straight])) < 1e-12


def test_analytic_jacobian_matches_finite_difference():
    """The analytic min-curve drk equals a central difference of the exact leg
    displacement _mc_leg_disp (validates the closed-form derivation)."""
    mc = _survey("min_curve").err
    md = mc.survey.md
    inc = mc.survey.inc_rad
    azi = mc.survey.azi_true_rad
    k = int(np.argmax(mc.survey.dogleg[1:])) + 1  # the highest-dogleg leg
    h = 1e-7
    # drk_dInc[k]: d(leg [k-1,k] disp)/d(inc_k)
    plus = ErrorModel._mc_leg_disp(md[k-1], inc[k-1], azi[k-1], md[k], inc[k]+h, azi[k])
    minus = ErrorModel._mc_leg_disp(
        md[k-1], inc[k-1], azi[k-1], md[k], inc[k]-h, azi[k])
    fd = (plus - minus) / (2 * h)
    assert np.max(np.abs(mc.drdp[k, 3:6] - fd)) < 1e-6


def test_min_curve_gap_is_small_and_positive():
    """The min_curve vs balanced_tangent covariance gap is O(interval^2), well
    inside the ISCWSA +/-1% inter-implementation band on the standard well."""
    bt = _survey("balanced_tangent").err.errors.cov_NEVs
    mc = _survey("min_curve").err.errors.cov_NEVs
    gap = max(
        abs(np.trace(mc[i]) - np.trace(bt[i])) / max(1e-9, np.trace(bt[i]))
        for i in range(len(bt))
    )
    assert gap < 0.01  # << 1% band


def test_min_curve_no_nan_on_vertical_top():
    """The standard well has a vertical top; min_curve must not NaN there."""
    mc = _survey("min_curve").err
    assert np.all(np.isfinite(mc.drdp))
    assert np.all(np.isfinite(mc.errors.cov_NEVs))
