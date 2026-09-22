"""No hand-written SLERP arc tangent in the package: one arc kernel, called.

``MinCurve.interpolate(angles=True)`` is the arc tangent (Rodrigues u-form).
A second implementation agrees today and drifts the first time either one
changes, so the package is scanned for the SLERP blend's signature,
``... sin(A - B) ... / sin(A)``.
"""

import textwrap

import welleng
from welleng.lint import find_slerp_arc_tangent

PKG = welleng.__path__[0]

#: Instances allowed to remain, pinned by source so a NEW one fails and
#: removing one forces this set to be updated. Empty: every arc tangent in the
#: package goes through MinCurve.interpolate.
KNOWN_REMAINING: set = set()


def test_detects_the_slerp_blend(tmp_path):
    """Known positive: the form removed from _interpolate_surveys."""
    f = tmp_path / "blend.py"
    f.write_text(textwrap.dedent("""
        import numpy as np
        def tangent(t1, t2, total, d):
            return (
                t1 * (np.sin(total - d) / np.sin(total))[:, None]
                + t2 * (np.sin(d) / np.sin(total))[:, None]
            )
        def scalar(t1, t2, a, d):
            return (math.sin(a - d) * t1 + math.sin(d) * t2) / math.sin(a)
    """))
    assert len(find_slerp_arc_tangent(str(f))) == 2


def test_ignores_the_rodrigues_kernel_and_half_angle_position():
    """Known negatives: the arc kernel itself must not trip the rule."""
    utils = f"{PKG}/utils.py"
    assert find_slerp_arc_tangent(utils) == []


def test_package_has_no_new_slerp_arc_tangent():
    found = {f.source for f in find_slerp_arc_tangent(PKG)}
    assert found == KNOWN_REMAINING, (
        "hand-written SLERP arc tangent; call MinCurve.interpolate(angles=True) "
        f"instead: {sorted(found - KNOWN_REMAINING)}; "
        f"no longer present (update KNOWN_REMAINING): "
        f"{sorted(KNOWN_REMAINING - found)}"
    )
