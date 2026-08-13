"""welleng.flow — reservoir-fluid reference oracle (black-oil PVT + IPR).

Open-core, scalar, pure-float reference implementations of the classic
reservoir-fluid forms: the black-oil PVT correlations (solution GOR, bubble
point, formation volume factors, densities, viscosities, gas Z-factor, gas/
liquid surface tension) in :mod:`welleng.flow.pvt`, the inflow-performance
relationships (Darcy PI, Vogel, Standing composite, Fetkovich, Jones) in
:mod:`welleng.flow.ipr`, and the multiphase vertical-lift-performance (VLP)
local pressure-gradient correlations (Colebrook friction, Zuber-Findlay
drift-flux, Hasan-Kabir mechanistic, Beggs-Brill) in :mod:`welleng.flow.vlp`.
Every form is the **published classic as-authored** — PVT carries an SI seam
over its oilfield-unit correlations, while IPR and VLP are strict-SI throughout
— and this module is the *correctness oracle* that the vectorised / GPU forms
in the commercial layer parity-gate against.

See :mod:`welleng.flow.pvt`, :mod:`welleng.flow.ipr` and
:mod:`welleng.flow.vlp` for the forms.
"""
from __future__ import annotations

from . import ipr, pvt, vlp

__all__ = ["ipr", "pvt", "vlp"]
