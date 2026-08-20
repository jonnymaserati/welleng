"""Tests for welleng.torque_drag."""
from types import SimpleNamespace

import numpy as np

from welleng.torque_drag import TorqueDrag, force_normal


def test_get_azi_delta_wraps_north_crossing():
    # welleng#312: the azimuth delta must wrap into (-pi, pi]. A station where
    # the well crosses north (359.9 -> 0.3 deg) is a 0.4 deg turn, NOT ~360.
    azi = np.radians([350.0, 359.9, 0.3, 10.0])
    td = SimpleNamespace(survey=SimpleNamespace(azi_grid_rad=azi))
    TorqueDrag.get_azi_delta(td)  # exercise the fixed method on a stub
    deg = np.degrees(td.azi_delta)
    # no step is anywhere near a full revolution
    assert np.all(np.abs(deg[1:]) < 45.0), deg
    # the north crossing specifically -> ~0.4 deg
    assert abs(deg[2] - 0.4) < 1e-6, deg[2]
    # first element is the zero seed
    assert deg[0] == 0.0


def test_force_normal_matches_reference_at_north_crossing():
    # With the wrapped delta the Johancsik normal force is the small-turn value,
    # not the ~260x explosion of the unwrapped ~360 deg delta (welleng#312).
    tension, inc_average = 3.0e5, np.radians(30.0)
    inc_delta, weight_buoyed = np.radians(0.5), 1600.0
    raw_delta = np.radians(-359.6)                      # plain subtraction
    wrapped = (raw_delta + np.pi) % (2 * np.pi) - np.pi  # what the fix produces
    assert abs(np.degrees(wrapped) - 0.4) < 1e-9
    f_wrapped = force_normal(
        tension, inc_average, inc_delta, wrapped, weight_buoyed)
    f_raw = force_normal(
        tension, inc_average, inc_delta, raw_delta, weight_buoyed)
    assert f_raw / f_wrapped > 100          # the reported blow-up exists
    assert f_wrapped < 5.0e3                 # the wrapped force is physical
