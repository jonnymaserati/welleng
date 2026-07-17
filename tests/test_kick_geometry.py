"""Catalogue-backed migration geometry: true annular capacity (bore - string).

Requires welleng.catalog (the API-5CT tubular catalogue); on the integration
branch that carries both subpackages.
"""
import math

import pytest

pytest.importorskip("welleng.catalog")

from welleng.kick_tolerance import (
    annular_capacity, cased_section, open_hole_section, migrate,
)


def test_annular_capacity_subtracts_the_string():
    # 9-5/8" 47 ppf casing ID = 8.681"; with 5" drillpipe.
    cap = annular_capacity(8.681, 5.0)
    assert cap == pytest.approx((8.681 ** 2 - 5.0 ** 2) / 1029.4, rel=1e-9)
    # Full-bore (no string) is larger -> the string subtraction matters.
    assert annular_capacity(8.681, 0.0) > cap
    # Pipe wider than the bore is rejected.
    with pytest.raises(ValueError):
        annular_capacity(5.0, 9.625)


def test_cased_section_resolves_id_from_catalogue():
    sec = cased_section(
        0.0, 5000.0, casing_od_in=9.625, casing_weight_ppf=47.0, pipe_od_in=5.0,
    )
    # Casing ID resolved from the catalogue (8.681") -> annular capacity.
    assert sec.annular_capacity_bbl_per_ft == pytest.approx(
        (8.681 ** 2 - 5.0 ** 2) / 1029.4, rel=1e-6
    )
    assert sec.is_open_hole is False
    assert (sec.top_tvd, sec.bottom_tvd) == (0.0, 5000.0)


def test_open_hole_section_uses_bit_diameter():
    sec = open_hole_section(5000.0, 8000.0, hole_size_in=8.5, pipe_od_in=5.0)
    assert sec.annular_capacity_bbl_per_ft == pytest.approx(
        (8.5 ** 2 - 5.0 ** 2) / 1029.4, rel=1e-9
    )
    assert sec.is_open_hole is True


def test_migration_runs_on_catalogue_backed_geometry():
    """End-to-end: build the section list from the catalogue and migrate."""
    sections = [
        cased_section(0.0, 5000.0, casing_od_in=13.375, casing_weight_ppf=68.0,
                      pipe_od_in=5.0),
        open_hole_section(5000.0, 9000.0, hole_size_in=8.5, pipe_od_in=5.0),
    ]
    import numpy as np
    pp = (np.array([0.0, 9000.0]), np.array([9.0, 11.0]))
    fp = (np.array([0.0, 9000.0]), np.array([12.0, 15.5]))
    bhp = 0.0521 * 12.0 * 9000.0 + 200.0
    res = migrate(
        sections, pp, fp, bhp_psi=bhp, influx_bbl_bh=15.0, rho_mud_ppg=12.0,
        gas_bh_state=(bhp, 660.0, None, None), n_steps=80,
    )
    # Runs end-to-end and produces the animation trajectory.
    assert len(res.steps) == 80
    assert isinstance(res.within_envelope, bool)
