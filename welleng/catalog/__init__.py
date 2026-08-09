"""welleng.catalog
------------------

OSDU-interfaced casing + tubing dimensional catalogue (API Spec 5CT) plus a
resolver that fills derived tubular dimensions (ID, wall, drift, yield) from
``(OD, weight[, grade])``.

Data lives in ``data/{casing,tubing}.json`` - factual, cited-by-number API 5CT
values in the same ``_meta`` + rows layout as the companion drilling-mechanics
catalogues. The loader converts to SI via each file's ``to_SI`` factors, so a
resolved :class:`TubularSpec` carries both imperial and SI values.

Example::

    from welleng.catalog import resolve
    spec = resolve(9.625, 47, grade="L80", kind="casing")
    spec.id_in     # 8.681
    spec.drift_in  # 8.525
    spec.yield_psi # 80000
"""
from __future__ import annotations

from .catalog import (
    Catalog,
    CatalogError,
    ConnectionSpec,
    CouplingCatalog,
    CouplingSpec,
    OSDU_ALIASES,
    TubularSpec,
    buttress_coupling_internal_yield_psi,
    buttress_coupling_thread_strength_klb,
    buttress_joint_strength_klb,
    buttress_pipe_thread_strength_klb,
    collapse_pressure_psi,
    coupling_connections,
    grades,
    internal_yield_pressure_psi,
    list_sizes,
    pipe_body_yield_klb,
    plain_end_weight_ppf,
    resolve,
    resolve_coupling,
    round_thread_joint_strength_klb,
    round_thread_pipe_fracture_strength_klb,
    round_thread_pullout_strength_klb,
)

__all__ = [
    "Catalog",
    "CatalogError",
    "TubularSpec",
    "OSDU_ALIASES",
    "resolve",
    "list_sizes",
    "grades",
    # couplings / connections
    "CouplingCatalog",
    "CouplingSpec",
    "ConnectionSpec",
    "resolve_coupling",
    "coupling_connections",
    # body performance (API TR 5C3)
    "plain_end_weight_ppf",
    "pipe_body_yield_klb",
    "internal_yield_pressure_psi",
    "collapse_pressure_psi",
    # connection performance (API TR 5C3 Sec 9 + 10.2)
    "buttress_joint_strength_klb",
    "buttress_pipe_thread_strength_klb",
    "buttress_coupling_thread_strength_klb",
    "buttress_coupling_internal_yield_psi",
    "round_thread_joint_strength_klb",
    "round_thread_pipe_fracture_strength_klb",
    "round_thread_pullout_strength_klb",
]
