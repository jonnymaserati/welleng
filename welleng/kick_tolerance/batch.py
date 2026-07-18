"""Batch / sweep entry points for kick tolerance (Phase 1).

Runs many kick-tolerance cases through the existing solvers with **per-case error
isolation** and **shared real-gas table amortization**, so scripted sweeps (LOT
grids, depth matrices, design curves) don't pay per-call setup N times.

Design (TA1 steer, 2026-07-18):

* **Serial** loops in core -- deterministic, `batch[i]` is bit-identical to the
  single call for case ``i`` (same function, same inputs, no cross-case state).
  Core does **not** spawn processes/threads: concurrency belongs to the caller's
  worker pool (the solvers are stateless for reads and a built :class:`ZTable`
  is read-only, so sharing across threads is safe).
* **Amortization**: pass one prebuilt ``fluid_table`` (:class:`ZTable`) and it is
  shared across every analytical case -- the CoolProp grid is built once, not per
  case. Common well geometry/profile is parsed per case by the solver; hand it the
  same objects to reuse them.
* **Isolation**: one bad case never fails the batch -- each result carries either a
  value or an error string, in input order.

The API layer (welleng-api) exposes these as a paid batch/sweep endpoint; core owns
only the loop + amortization + isolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence

from .analytical import AnalyticalKickTolerance, analytical_kick_tolerance
from .core import KickInputs, KickResult, drill_kick, swab_kick


@dataclass
class BatchCaseResult:
    """One case's outcome in a batch. Exactly one of ``result``/``error`` is set.

    Attributes
    ----------
    index : int
        Position of this case in the input sequence (results are returned in
        input order, so ``index`` also equals the list position).
    result : object or None
        The solver result (:class:`KickResult` or
        :class:`AnalyticalKickTolerance`) if the case succeeded, else ``None``.
    error : str or None
        ``"<ExceptionType>: <message>"`` if the case raised, else ``None``.
    """
    index: int
    result: Optional[Any] = None
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        """True if the case produced a result (no error)."""
        return self.error is None


_CLOSED_FORM: dict[str, Callable[[KickInputs], KickResult]] = {
    "drill": drill_kick,
    "swab": swab_kick,
}


def solve_batch(
    inputs: Sequence[KickInputs], *, kind: str = "drill"
) -> list[BatchCaseResult]:
    """Closed-form kick tolerance over a batch of cases (``drill`` or ``swab``).

    Each case runs :func:`~welleng.kick_tolerance.core.drill_kick` (or
    ``swab_kick``) with per-case error isolation, in input order.

    Parameters
    ----------
    inputs : sequence of KickInputs
        The cases to solve.
    kind : {"drill", "swab"}
        Which closed-form solver to apply to every case.

    Returns
    -------
    list of BatchCaseResult
        One entry per input, in order; each holds a :class:`KickResult` or an error.
    """
    if kind not in _CLOSED_FORM:
        raise ValueError(f"kind must be one of {sorted(_CLOSED_FORM)}, got {kind!r}")
    fn = _CLOSED_FORM[kind]
    out: list[BatchCaseResult] = []
    for i, inp in enumerate(inputs):
        try:
            out.append(BatchCaseResult(i, result=fn(inp)))
        except Exception as exc:  # per-case isolation: one bad case never fails the batch
            out.append(BatchCaseResult(i, error=f"{type(exc).__name__}: {exc}"))
    return out


def batch_analytical_kick_tolerance(
    cases: Sequence[Mapping[str, Any]], *, fluid_table: Any = None
) -> list[BatchCaseResult]:
    """Analytical kick tolerance over a batch of cases, sharing one real-gas table.

    Each case is a mapping of keyword arguments for
    :func:`~welleng.kick_tolerance.analytical.analytical_kick_tolerance` (``sections``,
    ``pp``, ``fp``, ``bhp_psi``, ``rho_mud_ppg``, ``gas_bh_state``, ...). Cases run
    serially in input order with per-case error isolation.

    Parameters
    ----------
    cases : sequence of mapping
        Per-case keyword arguments for ``analytical_kick_tolerance``.
    fluid_table : ZTable, optional
        A prebuilt real-gas table shared across every case (the amortization: the
        CoolProp grid is built once, not per case). If a case already specifies its
        own ``fluid_table`` it is left untouched.

    Returns
    -------
    list of BatchCaseResult
        One entry per case, in order; each holds an
        :class:`AnalyticalKickTolerance` or an error.
    """
    out: list[BatchCaseResult] = []
    for i, case in enumerate(cases):
        try:
            kwargs = dict(case)
            if fluid_table is not None and kwargs.get("fluid_table") is None:
                kwargs["fluid_table"] = fluid_table
            result: AnalyticalKickTolerance = analytical_kick_tolerance(**kwargs)
            out.append(BatchCaseResult(i, result=result))
        except Exception as exc:  # per-case isolation
            out.append(BatchCaseResult(i, error=f"{type(exc).__name__}: {exc}"))
    return out


def sweep_analytical_kick_tolerance(
    base: Mapping[str, Any], param: str, values: Sequence[Any], *, fluid_table: Any = None
) -> list[BatchCaseResult]:
    """Analytical kick tolerance sweeping one parameter of a base case over a grid.

    Convenience over :func:`batch_analytical_kick_tolerance`: expands ``base`` into
    one case per value of ``param`` (e.g. sweeping ``bhp_psi`` or ``fp``), sharing the
    ``fluid_table`` across the whole sweep. This is the design-curve / paid-sweep
    primitive -- one base well parsed against one shared real-gas table.

    Parameters
    ----------
    base : mapping
        Keyword arguments for ``analytical_kick_tolerance`` common to every point.
    param : str
        The keyword to vary.
    values : sequence
        The grid of values for ``param``; one case is solved per value, in order.
    fluid_table : ZTable, optional
        Shared real-gas table (see :func:`batch_analytical_kick_tolerance`).

    Returns
    -------
    list of BatchCaseResult
        One entry per value in ``values``, in order.
    """
    cases = [{**base, param: v} for v in values]
    return batch_analytical_kick_tolerance(cases, fluid_table=fluid_table)
