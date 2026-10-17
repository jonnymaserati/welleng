"""``find_attitude_difference``: a finite difference of station inclination or
azimuth, which is not the curvature of the minimum-curvature path."""
import textwrap

import numpy as np

import welleng
from welleng.lint import find_attitude_difference
from welleng.survey import Survey

PKG = welleng.__path__[0]


def _scan(tmp_path, src):
    f = tmp_path / "m.py"
    f.write_text(textwrap.dedent(src))
    return find_attitude_difference(str(f))


def test_flags_gradient_and_diff_of_an_angle_in_every_spelling(tmp_path):
    hits = _scan(tmp_path, """
        import numpy as np
        def f(survey, d):
            a = np.gradient(survey.inc_rad, survey.md)
            b = np.diff(survey.azi_deg)
            inc = np.asarray(survey.inc_rad, float)
            c = np.gradient(inc[1:], survey.md[1:])
            d = np.diff(d["azimuth"])
            i_s = np.radians(survey.inc_deg)
            e = np.diff(i_s)
            g = np.gradient(inc_rad(survey.md), survey.md)
            return a, b, c, d, e, g
    """)
    assert [h.axis for h in hits] == ["attitude:inc_rad", "attitude:azi_deg",
                                      "attitude:inc", "attitude:azimuth",
                                      "attitude:inc_deg", "attitude:inc_rad"]


def test_ignores_depths_positions_and_the_suppressed_line(tmp_path):
    hits = _scan(tmp_path, """
        import numpy as np
        def f(survey, sigma):
            a = np.diff(survey.md)
            b = np.gradient(survey.tvd, survey.md)
            c = np.diff(sigma, axis=0)
            d = np.diff(survey.inc_deg)  # lint: attitude-difference ok
            return a, b, c, d
    """)
    assert hits == []


def test_a_model_output_over_angles_is_not_an_angle(tmp_path):
    # the two shapes a consumer's tests hit: a monotonicity check on a model's
    # output, computed over an inclination array; a wrapper that keeps the values
    # (np.unwrap) still carries the angle
    hits = _scan(tmp_path, """
        import numpy as np
        def f(model, survey, incs, vs):
            v = model(incs, 0.2)
            a = np.all(np.diff(v) > 0)
            b = np.all(np.diff(model(incs)) > 0)
            vs = [model(i) for i in incs]
            c = np.all(np.diff(vs) > 0)
            d = np.diff(np.unwrap(survey.azi_rad))
            return a, b, c, d
    """)
    assert [h.axis for h in hits] == ["attitude:azi_rad"]


def test_the_package_is_clean():
    assert find_attitude_difference(PKG) == []


def test_why_a_difference_of_inclination_is_not_the_curvature():
    """Reference: the survey's own curvature per leg, ``dogleg / delta_md``, on a
    leg that builds and turns at once. The inclination difference over the leg is
    smaller (it drops the turn), so the rule's premise holds on a real survey."""
    s = Survey(md=[0.0, 500.0, 530.0], inc=[0.0, 30.0, 33.0], azi=[0.0, 0.0, 15.0])
    kappa = s.dogleg[2] / s.delta_md[2]
    from_inc = np.radians(s.inc_deg[2] - s.inc_deg[1]) / s.delta_md[2]
    assert kappa > 1.5 * from_inc
