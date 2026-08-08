"""Tests for welleng.surface.Surface + the OpenWorks OW-IO horizon reader.

Uses a small synthetic OW-IO horizon in the real export layout (# provenance
header, a short text header naming survey/horizon/domain, then comma-delimited
inline,crossline,X,Y,Z rows). The affine X = 1000 + 10*IL + 2*XL,
Y = 5000 + 3*IL + 8*XL makes node<->map mapping exact and checkable.
"""
import numpy as np
import pytest

from welleng.surface import Surface
from welleng.exchange.openworks import read_ow_horizon


def _xy(il, xl):
    return 1000.0 + 10.0 * il + 2.0 * xl, 5000.0 + 3.0 * il + 8.0 * xl


# a 3x3 grid; Z = 100*il + xl so every node value is distinct + predictable
_ILS, _XLS = (10, 11, 12), (20, 21, 22)


def _synthetic_dat(domain="DEPTH"):
    lines = [
        "=",
        "# Export file created by a test",
        "# Source cartographic system name: TEST_UTM31N",
        "SURVEY1",
        "My_Horizon_Top",
        "STAT",
        domain,
        ",",
    ]
    for il in _ILS:
        for xl in _XLS:
            x, y = _xy(il, xl)
            z = 100.0 * il + xl
            lines.append(f"{il}.0,{xl}.0,{x:.4f},{y:.4f},{z:.4f}")
    return lines


# --- reader ------------------------------------------------------------------
def test_reader_parses_header_and_grid():
    s = read_ow_horizon(_synthetic_dat())
    assert s.name == "My_Horizon_Top"
    assert s.domain == "DEPTH"
    assert s.crs == "TEST_UTM31N"
    assert s.z.shape == (3, 3)
    assert np.isclose(s.z[0, 0], 100 * 10 + 20)      # il=10, xl=20


def test_reader_twt_domain():
    s = read_ow_horizon(_synthetic_dat(domain="TWT"))
    assert s.domain == "TWT"


def test_reader_raises_without_data():
    with pytest.raises(ValueError, match="no horizon data"):
        read_ow_horizon(["# only comments", "HEADER", "DEPTH"])


# --- Surface.z_at ------------------------------------------------------------
def test_z_at_hits_nodes_exactly():
    s = read_ow_horizon(_synthetic_dat())
    for il in _ILS:
        for xl in _XLS:
            x, y = _xy(il, xl)
            assert s.z_at(x, y) == pytest.approx(100 * il + xl, abs=1e-9)


def test_z_at_bilinear_midpoint():
    s = read_ow_horizon(_synthetic_dat())
    x0, y0 = _xy(10, 20)
    x1, y1 = _xy(11, 20)
    zmid = s.z_at((x0 + x1) / 2, (y0 + y1) / 2)
    assert zmid == pytest.approx((s.z[0, 0] + s.z[1, 0]) / 2, abs=1e-9)


def test_z_at_off_grid_is_nan():
    s = read_ow_horizon(_synthetic_dat())
    assert np.isnan(s.z_at(0.0, 0.0))


def test_z_at_vectorised():
    s = read_ow_horizon(_synthetic_dat())
    x0, y0 = _xy(10, 20)
    out = s.z_at(np.array([x0, 0.0]), np.array([y0, 0.0]))
    assert out.shape == (2,)
    assert out[0] == pytest.approx(s.z[0, 0]) and np.isnan(out[1])


def test_z_at_nan_neighbour_does_not_poison_exact_node():
    # drop one node -> its neighbours are NaN; querying a *present* node whose
    # zero-weight corners include the hole must still return the node value.
    lines = _synthetic_dat()
    lines = [ln for ln in lines if not ln.startswith("12.0,22.0,")]  # remove a corner
    s = read_ow_horizon(lines)
    x, y = _xy(11, 21)
    assert s.z_at(x, y) == pytest.approx(100 * 11 + 21, abs=1e-9)
    # and the removed node's own location is NaN
    assert np.isnan(s.z_at(*_xy(12, 22)))


# --- interval / above-below --------------------------------------------------
def test_within_formation_interval():
    top = read_ow_horizon(_synthetic_dat())
    # a base surface 500 deeper everywhere
    base_lines = []
    for ln in _synthetic_dat():
        parts = ln.split(",")
        if len(parts) == 5 and parts[0][0].isdigit():
            parts[4] = f"{float(parts[4]) + 500.0:.4f}"
            base_lines.append(",".join(parts))
        else:
            base_lines.append(ln)
    base = read_ow_horizon(base_lines)
    x, y = _xy(11, 21)
    zt = top.z_at(x, y)
    assert top.within(base, x, y, zt + 250) == np.True_          # inside
    assert top.within(base, x, y, zt - 10) == np.False_          # above top
    assert top.within(base, x, y, zt + 600) == np.False_         # below base


def test_is_below():
    s = read_ow_horizon(_synthetic_dat())
    x, y = _xy(11, 21)
    z = s.z_at(x, y)
    assert s.is_below(x, y, z + 1) == np.True_
    assert s.is_below(x, y, z - 1) == np.False_


def test_from_nodes_requires_grid():
    with pytest.raises(ValueError, match="2x2"):
        Surface.from_nodes([10], [20], [1.0], [2.0], [3.0])
