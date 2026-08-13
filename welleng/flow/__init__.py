"""welleng.flow — reservoir-fluid (black-oil PVT) reference oracle.

Open-core, scalar, pure-float reference implementations of the classic
black-oil PVT correlations (solution GOR, bubble point, formation volume
factors, densities, viscosities, gas Z-factor). Every correlation is the
**published classic as-authored** in its original field units, with an SI
seam at the interface — this module is the *correctness oracle* that the
vectorised / GPU forms in the commercial layer parity-gate against.

See :mod:`welleng.flow.pvt` for the correlations.
"""
from __future__ import annotations

from . import pvt

__all__ = ["pvt"]
