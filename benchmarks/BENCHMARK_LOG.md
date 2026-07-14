# Benchmark log

Append-only record of benchmark runs, newest first. This log is the **evidence for
the pre-merge benchmark gate**: performance-sensitive engine code must be benchmarked
+ profiled and logged here BEFORE it is merged (see
[`docs/dev/BENCHMARK_GATE.md`](../docs/dev/BENCHMARK_GATE.md)). Timings are per-host
and indicative — compare RELATIVE change on the same machine, not absolute numbers.

Each entry: date · branch/commit · host · the numbers · what changed + the profile
finding that motivated it.

---

## 2026-07-14 · `feat/kick-tolerance` · Python 3.12, dev machine

`python benchmarks/bench_kick_tolerance.py`

| operation | before | after | speedup |
|---|---|---|---|
| `drill_kick` (H-Y auto Z) | 45.3 µs | **22.6 µs** | 2.0× |
| `migrate` (n_steps=100) | 573 ms | **293 ms** | 2.0× |
| `max_influx_circulated` (inverse solve) | 7491 ms | **3758 ms** | 2.0× |

**Profile finding (cProfile on `max_influx_circulated`):** ~95 % of the time was the
Hall-Yarborough Z-factor Newton solve — 12.5 M `reduced_density` calls, 65 M residual
+ 53 M derivative evaluations — invoked scalar-sequentially inside a forward-Euler
gas-column integration.

**Change (pure speed, results unchanged — the 39-test kick validation suite is the
guardrail):**
- inlined `_hy_residual` / `_hy_residual_derivative` into `reduced_density` and
  precomputed the temperature-coefficient groups once per solve (kills ~118 M Python
  function calls + repeated power math);
- warm-started the Newton from the previous nearby sub-step's `y`
  (`hall_yarborough_z_and_y` / `_z_and_y_at`) — ~2 iterations instead of ~5, same
  converged root to 1e-12;
- isothermal temperature callable returns the scalar instead of allocating a fresh
  `np.full` array per call (~11 M allocations removed).

**Not done (blocked):** vectorizing the Newton across sub-steps — the integration is
forward-Euler (each pressure depends on the previous), so the solves are sequential.

Validation: `tests/test_migration.py test_nogepa.py test_spe208788_worked_example.py
test_spe202426_fig12.py test_spe140113_santos.py test_kick_tolerance.py test_gas_z.py`
= 39 passed, 1 skipped (identical results before/after).

## 2026-07-14 (later) · `feat/kick-tolerance` · fast/thorough mode

Added a `mode="fast"|"thorough"` switch to `migrate` / `max_influx_circulated`
(thorough is the default; the fine 51-pts/section grid + n_steps march = unchanged
validated behaviour). **fast** anchors the check depths and the bubble march to the
INTERFACES — section boundaries (BHA / shoe / hole change) + PP/FP breakpoints, where
the binding constraint can turn — plus a light per-section fill (5 pts/section, 16
march steps). The envelope is smooth between interfaces (JJ), so this defines it
without the fine grid.

| `max_influx_circulated` | thorough | fast |
|---|---|---|
| result | 57.84 bbl | 57.57 bbl (**0.47 %**, same binding depth) |
| time (this host) | 3821 ms | **648 ms** |

**~4× on top of the H-Y 2×** → ~11.5× vs the original 7491 ms; **under 1 s** with
headroom for a slower cloud instance. Guardrails: the 39-test suite runs in thorough
(unchanged); `test_fast_mode_matches_thorough_within_tolerance` locks fast within 2 %
of thorough. Survey note: the engine consumes piecewise-constant TVD `WellSection`s,
so interfaces stay discrete even for a deviated `from_survey` build; per-section
counts bound a survey that spawns many fine sections.

## 2026-07-14 (later) · `feat/kick-tolerance` · march-init fix (safety, not perf)

**Correctness fix** to the migration gas-top march init, surfaced by profiling +
a resolution sweep (`work/kick_tolerance/bug_resolution_sweep.py`). Old init
`gas_top_start = bottom - influx/cap_bottom_section` lengthened the bubble using the
BOTTOM (tight/BHA) section's capacity only; a bubble spilling into the wider open hole
above was over-lengthened → the march started ABOVE the binding interfaces → the
min-margin was taken over safe positions only → `max_influx_circulated` returned
**170 bbl** on a 3-section tight-BHA case when the true first-fracture value is ~55.
**Resolution-invariant** (identical n=16..900) ⇒ NOT discretization; a logic bug.
Fixed by `_fill_up` (mirror of `_fill_down`; spills interval-to-interval, general).

| tight-BHA case | before | after |
|---|---|---|
| `max_influx_circulated` | 170 bbl (non-conservative) | **55.03 bbl** (fine-scan truth <55; conservative) |
| margin vs influx | non-monotone (−4 then +16) | monotone decreasing |

Perf unchanged (init-only, O(sections)): thorough 3821→3790 ms, fast 648→652 ms
(within noise). Validation: 44 passed, 1 skipped (standard-case results identical;
the fix only moves the tight-BHA answer). Residual (narrow gas-bottom breakpoint
under-sampling) → the analytical solver with the complete breakpoint set
(`docs/dev/KICK_ANALYTICAL_PLAN.md`).

## 2026-07-14 (later) · `feat/kick-analytical` · analytical KT solver + optimisation

New `analytical_kick_tolerance` (`welleng/kick_tolerance/analytical.py`): the
migration-form KT evaluated ONLY at the breakpoints of P(gas position) (gas-top-
and gas-bottom-at-boundary + deepest + PP/FP breaks), not a fine march. Exact worst
position (no under-sampling), conservative or exact density mode. Validated vs the
migration (thorough, n=300): base −0.16%, weak +0.05%, sloped +0.07%, tight-BHA
−1.26% (all safe-side / exact); CI = `tests/test_kick_analytical.py` (9 tests).

Optimisation (profile-led; results unchanged across the sweep):
1. profile → `_top_for_bottom` = ~80%, driven by the Hall-Yarborough Newton.
2. family-1 (gas-top-at-boundary) inner solve: 40-iter BISECTION → SECANT (~6 iters,
   monotone) — the dominant call multiplier.
3. trimmed over-set caps: influx bisection 60→44, conservative fixed point 50→30.
4. Z-cache: `lru_cache` on H-Y Z(P,T) rounded to (1 psi, 0.1 degR) — same validated
   backend, ~1000× fewer Newton solves (the same (P,T) recur across the bisection
   and candidates). Migration path unchanged (still un-cached).

| operation | time |
|---|---|
| `analytical_kick_tolerance` [conservative] | ~75 ms |
| `analytical_kick_tolerance` [exact] | ~72 ms  (closed-form column, no fixed point) |
| `max_influx_circulated` [fast] (the march) | ~656 ms |
| `max_influx_circulated` [thorough] | ~3780 ms |

611 → ~75 ms over the optimisation (~8×); **~9× faster than the fast march** and
EXACT/conservative (the march can under-sample a narrow breakpoint). Under the
API/GUI 1 s target with cloud headroom. Backend note: uses H-Y; the CoolProp
real-EOS backend (API default for mixtures/CCUS) is NOT yet wired into the column
integration for either engine — a follow-up needing a precomputed Z(P,T) table.
