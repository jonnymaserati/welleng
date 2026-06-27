"""Tests for MeshClearance and the WellMesh polygon_fit option.

The uncertainty surface is approximated by an n_verts polygon. "circumscribed"
(the default) scales the polygon out by 1/cos(pi/n_verts) so it contains the
ellipse and never under-represents the uncertainty for the given sigma — the
safety-conservative choice; "inscribed" puts the vertices on the ellipse and
under-counts the area between them. See docs/dev and the MTI/anti-collision work.

Skipped if the optional mesh dependencies (trimesh + python-fcl) are absent.
"""
import json

import numpy as np
import pytest

pytest.importorskip("trimesh")
pytest.importorskip("fcl")

from welleng.survey import Survey, make_survey_header  # noqa: E402
from welleng.clearance import MeshClearance  # noqa: E402
from welleng.mesh import WellMesh  # noqa: E402

DATA = json.load(open("tests/test_data/clearance_iscwsa_well_data.json"))


def _survey(well):
    sh = make_survey_header(DATA["wells"][well]["header"])
    radius = 0.4572 if well == "Reference well" else 0.3048
    w = DATA["wells"][well]
    return Survey(
        md=w["MD"], inc=w["IncDeg"], azi=w["AziDeg"], n=w["N"], e=w["E"],
        tvd=w["TVD"], radius=radius, header=sh, error_model="ISCWSA MWD Rev4",
    )


def test_polygon_fit_default_is_circumscribed():
    """The default polygon_fit is circumscribed (the conservative surface)."""
    ref, off = _survey("Reference well"), _survey("06 - well")
    sf_default = float(np.min(MeshClearance(ref, off, n_verts=12).sf))
    sf_circ = float(np.min(
        MeshClearance(ref, off, n_verts=12, polygon_fit="circumscribed").sf))
    assert np.isclose(sf_default, sf_circ)


def test_circumscribed_is_conservative():
    """Circumscribed SF <= inscribed SF: the conservative surface never reports
    *more* separation than the optimistic one (never under-counts collision
    risk for the given sigma)."""
    ref, off = _survey("Reference well"), _survey("06 - well")
    sf_circ = float(np.min(MeshClearance(ref, off, n_verts=12, polygon_fit="circumscribed").sf))
    sf_insc = float(np.min(MeshClearance(ref, off, n_verts=12, polygon_fit="inscribed").sf))
    assert sf_circ <= sf_insc


def test_wellmesh_circumscribed_contains_inscribed():
    """The circumscribed mesh vertices lie outside the inscribed ones (the
    bloat factor 1/cos(pi/n_verts) is applied)."""
    s = _survey("06 - well")
    m_in = WellMesh(s, n_verts=12, sigma=2.445, polygon_fit="inscribed")
    m_out = WellMesh(s, n_verts=12, sigma=2.445, polygon_fit="circumscribed")
    bloat = 1.0 / np.cos(np.pi / 12)
    assert np.isclose(m_out._bloat, bloat)
    assert m_in._bloat == 1.0
    # circumscribed vertices are further from the well centre-line than inscribed
    centre = np.array([s.n, s.e, s.tvd]).T[:, None, :]
    r_in = np.linalg.norm(m_in.vertices - centre, axis=-1)
    r_out = np.linalg.norm(m_out.vertices - centre, axis=-1)
    assert np.all(r_out >= r_in - 1e-9)
    assert np.mean(r_out) > np.mean(r_in)


def test_invalid_polygon_fit_rejected():
    s = _survey("06 - well")
    with pytest.raises(AssertionError):
        WellMesh(s, n_verts=12, polygon_fit="banana")


def test_mesh_sf_regression_anchor():
    """Lock the mesh min-SF for ISCWSA Well 06 (default circumscribed, n_verts=12)
    to guard the closest-point refactor and future changes."""
    ref, off = _survey("Reference well"), _survey("06 - well")
    sf = float(np.min(MeshClearance(ref, off, n_verts=12).sf))
    assert np.isclose(sf, 1.0100, rtol=1e-2)


def test_mesh_detects_known_collision():
    """ISCWSA Well 03 deeply overlaps the reference — the mesh must flag it
    (min SF < 1)."""
    ref, off = _survey("Reference well"), _survey("03 - well")
    assert float(np.min(MeshClearance(ref, off, n_verts=12).sf)) < 1.0
