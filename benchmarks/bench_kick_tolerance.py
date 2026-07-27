"""Micro-benchmark for the welleng kick-tolerance engine.

Times the operations that matter for the (cloud) API so regressions in latency /
run-cost are visible. Run:

    python benchmarks/bench_kick_tolerance.py

Operations
----------
* ``drill_kick`` -- the basic single-shoe closed form (the free/basic API tier).
* ``migrate``    -- one gas-migration sweep.
* ``max_influx_circulated`` -- the inverse solve (bisection over migrations); the
  advanced-tier API hot path and the dominant cost.

Recorded baselines (Python 3.12, one dev machine, 2026-07-14) -- indicative only,
compare RELATIVE change on the same host:

    operation                 original   after opt   speedup
    drill_kick (H-Y auto Z)    45.3 us    22.6 us     ~2.0x
    migrate (n_steps=100)       573 ms     293 ms     ~2.0x
    max_influx_circulated      7491 ms    3758 ms     ~2.0x

The optimisation (2026-07-14) was pure speed, results unchanged (the kick-tolerance
validation suite is the guardrail): the Hall-Yarborough Newton solve is the ~95%
hot path -- its residual/derivative were inlined with the temperature coefficients
precomputed once per solve, the Newton is warm-started from the previous nearby
sub-step, and the isothermal temperature callable no longer allocates an array per
call. The gas-column integration is forward-Euler (sequential in pressure) so the
Newton cannot be vectorised across sub-steps; these were the available safe wins.
"""
from __future__ import annotations

import time

import numpy as np

from welleng.kick_tolerance import (
    KickInputs,
    WellSection,
    analytical_kick_tolerance,
    drill_kick,
    max_influx_circulated,
    migrate,
)


def _time(fn, repeat):
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) / repeat


def _closed_form_case(pp: float = 11.5):
    v_dpa = (6.125 ** 2 - 4.0 ** 2) / 1029.4
    return KickInputs(
        rho_mud=11.9, PP=pp, kick_intensity=1.1, P_lot=16.0, P_apl=210.0,
        D_td=10500.0, D_lot=6500.0, T_s=212.0, T_td=302.0, V_dpa=v_dpa,
    )


def _distinct_closed_form_cases(n: int):
    """DISTINCT inputs, which is what a sweep or a batch endpoint actually does.

    Timing one case repeatedly measures the memo, not the engine. That nearly
    hid a 245x regression on 2026-07-27: the harness reported 12.9 us where the
    honest figure was 0.321 ms, because `_bubble_state` is memoised and every
    call after the first was a cache hit. Perturb the input so each call is real
    work.
    """
    return [_closed_form_case(11.5 + i * 1e-4) for i in range(n)]


def _migration_case():
    sections = [
        WellSection(0.0, 6500.0, 0.066, is_open_hole=False),
        WellSection(6500.0, 10500.0, 0.046, is_open_hole=True),
    ]
    pp = (np.array([0.0, 10500.0]), np.array([10.5, 11.0]))
    fp = (np.array([0.0, 10500.0]), np.array([14.0, 14.0]))
    gas_bh_state = (None, 660.0, None, None)   # H-Y fills Z / density
    return sections, pp, fp, gas_bh_state


def main() -> None:
    inp = _closed_form_case()
    sections, pp, fp, gas_bh_state = _migration_case()
    common = dict(
        bhp_psi=6402.0, rho_mud_ppg=12.0, gas_bh_state=gas_bh_state, n_steps=100,
    )

    # repeated-input figure, kept for continuity with earlier entries
    dk_memo = _time(lambda: drill_kick(inp), 2000) * 1e6
    # and the honest one: every call a distinct case
    _cases = _distinct_closed_form_cases(200)
    _i = iter(_cases)
    def _next_case():
        nonlocal _i
        try:
            return drill_kick(next(_i))
        except StopIteration:
            _i = iter(_cases)
            return drill_kick(next(_i))
    dk = _time(_next_case, 200) * 1e6
    mg = _time(
        lambda: migrate(sections, pp, fp, influx_bbl_bh=25.0, **common), 30
    ) * 1e3
    mx = _time(
        lambda: max_influx_circulated(sections, pp, fp, mode="thorough", **common), 10
    ) * 1e3
    mxf = _time(
        lambda: max_influx_circulated(sections, pp, fp, mode="fast", **common), 10
    ) * 1e3
    ana_common = dict(bhp_psi=6402.0, rho_mud_ppg=12.0, gas_bh_state=gas_bh_state)
    analytical_kick_tolerance(sections, pp, fp, **ana_common)          # warm Z cache
    anc = _time(
        lambda: analytical_kick_tolerance(
            sections, pp, fp, gas_density_mode="conservative", **ana_common), 50
    ) * 1e3
    ane = _time(
        lambda: analytical_kick_tolerance(
            sections, pp, fp, gas_density_mode="exact", **ana_common), 50
    ) * 1e3

    print(f"{'operation':<38}{'time':>12}")
    print(f"{'-' * 50}")
    print(f"{'drill_kick, DISTINCT inputs':<38}{dk:>9.1f} us   <- quote this")
    print(f"{'drill_kick, repeated input (memo)':<38}{dk_memo:>9.1f} us")
    print(f"{'migrate (n_steps=100)':<38}{mg:>9.1f} ms")
    print(f"{'max_influx_circulated [thorough]':<38}{mx:>9.1f} ms")
    print(f"{'max_influx_circulated [fast]':<38}{mxf:>9.1f} ms")
    print(f"{'analytical_kick_tolerance [conserv]':<38}{anc:>9.1f} ms")
    print(f"{'analytical_kick_tolerance [exact]':<38}{ane:>9.1f} ms   <- API/GUI path")


if __name__ == "__main__":
    main()
