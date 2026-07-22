"""Regression: XYM3E / XYM4E weight functions must keep the azimuth-trig SIGN.

ISCWSA Error Model Definition v5.13 §5.1.2:

    XYM3E   inc: M.Abs(Cos(I)) * Cos(AzT)      azi: -M.(Abs(Cos(I)) * Sin(AzT)) / Sin(I)
    XYM4E   inc: M.Abs(Cos(I)) * Sin(AzT)      azi:  M.(Abs(Cos(I)) * Cos(AzT)) / Sin(I)

``Abs`` wraps ``Cos(I)`` ONLY -- the azimuth trig (Cos/Sin(AzT)) keeps its sign,
and so does the overall weight. welleng's hand-coded E-variants previously wrapped
the whole product in ``Abs`` (XYM3E) or dropped the ``Abs(Cos(I))`` entirely
(XYM4E), forcing a wrong sign for azi in 90-270 deg (XYM3E) or inc > 90 deg (XYM4E).

The single ISCWSA Standard Test Well stays azi <= 75 deg and inc <= 90 deg, where
``Abs(cos*cos) == Abs(cos)*cos`` and ``cos(inc) == Abs(cos(inc))`` EXACTLY, so the
bug is a 0.0-difference no-op there and the per-term validation never caught it.
These wells deliberately cross azi = 90/180/270 and inc = 90 to exercise it.
"""
import numpy as np

import welleng as we

MODEL = "ISCWSA MWD Rev5.11"


def _survey(inc, azi):
    md = np.arange(0.0, 300.0 * len(inc), 300.0)
    sh = we.survey.SurveyHeader(
        name="xym-wide", b_total=50000.0, dip=70.0,
        declination=0.0, azi_reference="grid",
    )
    return we.survey.Survey(
        md=md, inc=np.asarray(inc, float), azi=np.asarray(azi, float),
        header=sh, error_model=MODEL,
    )


# stations spanning cos(azi) < 0 (azi 120/200/300/260) and cos(inc) < 0 (inc 95/120)
_INC = [5, 30, 60, 95, 120, 80]
_AZI = [20, 120, 200, 300, 45, 260]


def test_xym3e_inclination_weight_keeps_cos_azi_sign():
    s = _survey(_INC, _AZI)
    e_inc = np.asarray(s.err.errors.errors["XYM3E"].e_DIA)[1:, 1]
    cos_azi = np.cos(np.radians(_AZI))[1:]
    # inc weight = Abs(cos(inc)) * cos(azi) * coeff, and Abs(cos(inc)) >= 0,
    # coeff > 0 => the sign of the weight must equal the sign of cos(azi).
    # (Pre-fix Abs wrapped the whole product => weight was always >= 0.)
    assert np.all(np.sign(e_inc) == np.sign(cos_azi)), (
        f"XYM3E inclination weight lost the cos(azi) sign:\n"
        f"  weight  = {e_inc}\n  cos(azi)= {cos_azi}"
    )
    # and it must actually go negative somewhere here (guards a degenerate pass)
    assert (e_inc < 0).any()


def test_xym4e_inclination_weight_keeps_sin_azi_sign():
    s = _survey(_INC, _AZI)
    e_inc = np.asarray(s.err.errors.errors["XYM4E"].e_DIA)[1:, 1]
    sin_azi = np.sin(np.radians(_AZI))[1:]
    # inc weight = Abs(cos(inc)) * sin(azi) * coeff => sign == sign(sin(azi)),
    # INCLUDING at inc > 90 where cos(inc) < 0 (pre-fix bare cos(inc) flipped it).
    assert np.all(np.sign(e_inc) == np.sign(sin_azi)), (
        f"XYM4E inclination weight lost the sin(azi)/Abs(cos inc) sign:\n"
        f"  weight  = {e_inc}\n  sin(azi)= {sin_azi}"
    )


def test_xym4e_survives_inc_over_90():
    # inc > 90 is where XYM4E's missing Abs(cos(inc)) flipped the azimuth weight.
    inc = [95.0, 120.0, 100.0]
    azi = [30.0, 60.0, 45.0]
    s = _survey(inc, azi)
    e_azi = np.asarray(s.err.errors.errors["XYM4E"].e_DIA)[1:, 2]
    cos_azi = np.cos(np.radians(azi))[1:]
    # azi weight = Abs(cos(inc)) * cos(azi) / sin(inc) * coeff; sin(inc) > 0 for
    # inc in (0,180), Abs(cos inc) >= 0, coeff > 0 => sign == sign(cos(azi)),
    # even though cos(inc) < 0 here (pre-fix bare cos(inc) flipped it).
    assert np.all(np.sign(e_azi) == np.sign(cos_azi))
