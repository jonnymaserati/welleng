"""welleng.flow — reservoir-fluid reference oracle (black-oil PVT + IPR).

Open-core, scalar, pure-float reference implementations of the classic
reservoir-fluid forms: the black-oil PVT correlations (solution GOR, bubble
point, formation volume factors, densities, viscosities, gas Z-factor) in
:mod:`welleng.flow.pvt`, and the inflow-performance relationships (Darcy PI,
Vogel, Standing composite, Fetkovich, Jones) in :mod:`welleng.flow.ipr`. Every
form is the **published classic as-authored** — PVT carries an SI seam over its
oilfield-unit correlations, while IPR is strict-SI throughout — and this module
is the *correctness oracle* that the vectorised / GPU forms in the commercial
layer parity-gate against.

See :mod:`welleng.flow.pvt` and :mod:`welleng.flow.ipr` for the forms.
"""
from __future__ import annotations

from . import ipr, pvt

__all__ = ["ipr", "pvt"]
