# Benchmark log

Append-only record of benchmark runs, newest first. This log is the **evidence for
the pre-merge benchmark gate**: performance-sensitive engine code must be benchmarked
+ profiled and logged here BEFORE it is merged (see
the project's local benchmark-gate policy). Timings are per-host
and indicative — compare RELATIVE change on the same machine, not absolute numbers.

Each entry: date · branch/commit · host · the numbers · what changed + the profile
finding that motivated it.

---

## 2026-07-18 · `feat/survey-lazy-canonical` · Python 3.12, dev machine

Survey construction after increment C (lazy method-model): toolface/build-turn
rates + vertical section are deferred out of `__init__` and computed on first
access via `__getattr__`.

| stations | original | after B | after C | vs original |
|---|---|---|---|---|
| 100 | 0.349 ms | 0.334 ms | **0.189 ms** | ~1.8× |
| 1,000 | 0.872 ms | 0.837 ms | **0.448 ms** | ~1.9× |
| 5,000 | 3.324 ms | 3.226 ms | **1.840 ms** | ~1.8× |

Profile finding: after the needed `_min_curve` geometry (~47%),
`_get_toolface_and_rates` was ~45% of construction (a SplitSurvey + toolface
trig + build/turn rates + plane normals) and is frequently unused. Deferred it
(lazy via `__getattr__`) -> ~**1.8×** faster construction for the common case
that never touches it. First access computes the whole group once (`__getattr__`
populates `self.__dict__`, so subsequent reads are direct); values are identical
to the eager computation (`test_survey_lazy`), and pickle / deepcopy are safe (a
`dls` guard prevents firing before geometry is built).

`_get_vertical_section` (~4%) is kept EAGER on purpose (it uses only n/e, so it
doesn't re-trigger the deferred toolface). Separately, the azimuth at vertical
stations (inc == 0, where azi is undefined) is canonicalised to 0 in
`_make_angles` -- a clear "vertical / not yet steered" sentinel rather than the
arbitrary input azimuth; inc == 0 means no N/E displacement, so it never changes
geometry. Regression-pinned in `test_survey_lazy`. This is the "method model"
perf lever -- the big gain B set up.

## 2026-07-18 · `feat/datum-header-transform` · Python 3.12, dev machine

Survey construction after increment B (datum→header, `azi_reference` default→grid,
positions via the explicit header transform `poss @ Aᵀ + b`):

| stations | before B | after B |
|---|---|---|
| 100 | 0.349 ms | 0.334 ms |
| 1,000 | 0.872 ms | 0.837 ms |
| 5,000 | 3.324 ms | 3.226 ms |

Modest (~3%): the matrix transform is marginally cheaper than the prior
`get_nev * scale + start_nev` scatter, but Survey construction is dominated by the
min-curve kernel + the per-station azi-reference trig + the eager toolface/rates/errors/
vertical-section work. **The big gains land in increment C** — canonical internals + lazy
(method-model) derivation, so the per-station azi trig and the eager error/vertical-section
passes only run when asked. B is primarily architecture (header = georef truth + the
explicit transform); C is where the perf shows.

## 2026-07-18 · `feat/units-converter` · Python 3.12, dev machine

`welleng.units.Units` — the generic, cross-module boundary converter (canonical-SI
internals; user I/O at the edge; performance-critical callers bypass it entirely):

| operation | Units | pint `Quantity.to()` | speedup |
|---|---|---|---|
| scalar convert (ft→m) | **0.24 µs** | 12.1 µs | ~50× |

Design: pint computes the affine `(factor, offset)` per unit pair ONCE at first use
and caches it; conversion is then pure numpy arithmetic (`value * factor (+ offset)`),
scalar or array, with **no pint on the hot path**. For 1M-element arrays pint's own
`.to()` is already numpy-backed (~3 ms, comparable), so the win is on the common
scalar / small-input boundary case (a header datum, a single DLS) — 50×. Affine units
(temperature) carry the offset; multiplicative units skip it. Same-unit convert is an
identity no-op.

## 2026-07-18 · `feat/mincurve-phase1` · Python 3.12, dev machine

`MinCurve` construction (it becomes the foundational geometry kernel — Survey
inherits it, and downstream consumers use it directly, so it must be hyper-performant):

| stations | before | after | speedup |
|---|---|---|---|
| 1,000 | 0.645 ms | **0.175 ms** | 3.7× |
| 10,000 | 6.24 ms | **1.47 ms** | 4.2× |
| 100,000 | 70.7 ms | **19.6 ms** | 3.6× |

~0.18 µs/station (was ~0.64). Profile finding: the dominant cost was
`np.vstack(np.cumsum(...))` for `poss` — the `vstack` wrapper triggered
`atleast_2d` + a per-row stack dispatcher (~2M `list.append` at 100k stations),
pure overhead since `cumsum` already returns the (n,3) array. Replaced with
`np.cumsum(np.column_stack(...))`. Remaining cost is the irreducible trig in
`min_curve_step` + the Haversine `get_dogleg` (kept Haversine for small-dogleg
numerical stability; reworking its trig for marginal gain isn't worth the
precision risk, and a real survey is <2000 stations = <0.4 ms). Heavy coverage
added in `tests/test_mincurve.py`.

## 2026-07-18 · `feat/kt-batch` · Python 3.12, dev machine

Batch / sweep KT (Phase 1). `sweep_analytical_kick_tolerance` over a 30-point
design curve (methane / Hall-Yarborough, no CoolProp table):

| operation | result |
|---|---|
| 30-case analytical sweep | **199 ms total (6.6 ms/case)**, 30/30 ok |

This is exactly a design-curve builder (30 FP-offset re-solves) that a downstream
caller would otherwise hand-roll with a **3 s** time budget — now ~0.2 s in core, so the
budget/truncation can go. Serial by design (no core multiprocessing; the caller owns any worker pool). The real amortization for CoolProp mixtures is the
**shared `fluid_table`**: the ZTable (~seconds to build) is built ONCE for the batch
instead of per case — pass a prebuilt `fluid_table` to `batch_analytical_kick_tolerance`
/ `sweep_analytical_kick_tolerance`. Per-case error isolation adds no measurable
overhead (a try/except per case).

## 2026-07-18 · `fix/kt-unconstrained-closedform` · Python 3.12, dev machine

`analytical_kick_tolerance` on the `open_hole_unconstrained` regime (TWO_SECTION,
STRONG_FP), 5-call mean:

| operation | before | after | speedup |
|---|---|---|---|
| `analytical_kick_tolerance` (unconstrained regime) | ~5026 ms | **18.2 ms** | ~276× |

Profile finding (from a downstream regression report): the regime ran a 40-iteration bisect
where **each iteration executed a full `thorough` migration** (`n_steps=100`, all gas-top
positions) to find the influx whose bubble length fills the open hole. But the bubble is
longest at a single governing position (gas top at surface), so the detection and the
capacity bisect only need that ONE position. Replaced the per-iteration full march with a
single-position evaluation that mirrors the migration's per-step math exactly (same P_rep
seed, Boyle expansion, `_fill_down`, and damped fixed point via `pressure_at_depth`) —
identical numbers (golden `test_bubble_length_limit_is_casing_burst_regime`: analytical ≈
march, abs=0.5 bbl), ~276× faster. Fracture-limited (non-regime) calls unchanged.

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
(local analytical-solver plan).

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

## 2026-07-14 (later) · `feat/kick-analytical` · closed-form gas-top breach (fast path)

Replaced the outer influx bisection with a per-candidate DIRECT breach-influx solve:
gas-top families (1/4) are now CLOSED FORM (conservative = linear constant-density
column; exact = exponential column, a Lambert-W-form root by a 3-iter Newton, no
scipy dep) -- validated standalone vs the previous solver to ±0.02%; gas-bottom
families (2/3, interior/tight-BHA) keep a per-candidate secant on influx. Results
identical to the bisection solver (base -0.16%, weak +0.05%, sloped +0.06%, tight-BHA
-1.26% vs the march; 9-test CI unchanged).

| analytical_kick_tolerance | before (bisection) | after (closed-form gas-top) |
|---|---|---|
| conservative | ~75 ms | **~44 ms** |
| exact | ~72 ms | **~42 ms** |

611 -> ~42 ms cumulative (~14.5x); ~15x faster than the fast march. The closed form
also cuts Z-backend calls sharply -- the lever that matters for the slower CoolProp
backend (still a follow-up). Derivation: SymPy (`FP*exp(k*L)=A+b*L` -> LambertW),
implemented as a dependency-free Newton.

## 2026-07-14 (later) · `feat/kick-analytical` · gas-bottom secant (profiling close-out)

Profiling the closed-form solver showed the residual cost was the gas-bottom
families' 44-iter influx BISECTION (each step a pressure_at_depth + geometry solve).
Swapped it for a bracketed SECANT with an early break (0.25 psi / 0.001 bbl) -- the
margin is monotone in influx so it converges in ~5 iters, not 44. Results identical
(base -0.16%, weak +0.05%, sloped +0.06%, tight-BHA -1.26%; 9-test CI unchanged).

| analytical_kick_tolerance | closed-form (prev) | + gas-bottom secant |
|---|---|---|
| conservative | ~44 ms | **~4.2 ms** |
| exact | ~42 ms | **~4.1 ms** |

611 -> ~4.1 ms cumulative over the analytical work (~150x); **~155x faster than the
fast march** (655 ms). Close-out: the remaining cost is the irreducible Hall-Yarborough
Z kernel (`reduced_density`) + the shared migration `pressure_at_depth` in the
gas-bottom margin eval. Further gains need caching pressure_at_depth's internal H-Y or
an analytic conservative margin -- higher-risk edits to shared code for marginal benefit
far below the 1 s API/GUI target, so STOPPED here. CoolProp backend remains the real
next perf item (a precomputed Z(P,T) table), tracked separately.

## 2026-07-18 · `perf/cache-error-model-defs` · Python 3.12, dev machine

Cache the ISCWSA tool-model definition parses. Profiling `Survey(error_model=...)
.cov_nev` (5000 stn) showed **~45% of the error path was repeated file parsing**:
YAML `safe_load` of the model (~34%) + a JSON name-resolution walk that re-read
~100 files (~11%) — both re-run on EVERY Survey. The parsed dicts are read-only
(ToolError never mutates them; verified). Fix: `@lru_cache` on `_load_yaml_model`
/ `_load_json_model` / `_resolve_json_model`, plus a lazy reorder so the JSON walk
runs ONLY when the YAML is absent (the MWD default no longer walks the JSON tree
at all). `clear_error_model_cache()` for the mid-process model-file-swap edge.

Cold (per-survey re-parse = old) vs warm (cached = new steady state):

| stations | cold (old) | warm (new) | speedup |
|---|---|---|---|
| 100 | 9.52 ms | **3.08 ms** | **3.09×** |
| 500 | 11.80 ms | **6.64 ms** | 1.78× |
| 5,000 | 50.98 ms | **30.41 ms** | 1.68× |

The saving is a ~fixed parse cost (~6–20 ms), so it dominates SMALL/typical
surveys most (3.1× at 100 stn) and compounds across a batch (the same MWD model
was re-parsed N times). `cov_nev` is **bit-identical** cold vs warm
(`test_error_model_cache::test_cached_cov_identical_to_cold`). OPEN + non-breaking
(fixes repeated I/O, not a commercial capability); feeds the lazy error-class
redesign (parse model defs once, reuse per section). Regression:
`tests/test_error_model_cache.py` (4).

## 2026-07-19 · `feat/arc-inc-azi-extrema` · Python 3.12, dev machine

`welleng.utils.arc_inc_azi_extrema` — exact inclination/azimuth extrema over
min-curvature arcs. Replaces a 96-sample/arc envelope test with a closed form.

| operation | result |
|---|---|
| vectorized over 100k arcs | 72.8 ms (**1.4M arcs/s**) |

Exactness (vs 4000-sample/arc reference over 3000 random arcs incl. reflex):
inclination extrema max err 2.8e-5 rad; azimuth swing max err 1.8e-15 (azimuth is
strictly monotonic along a circular arc -> extrema at endpoints, signed swing via
an analytic branch-crossing count). `tests/test_arc_extrema.py` (6).

## 2026-07-19 · `perf/mahalanobis-solve` · Python 3.12, dev machine

MahalanobisClearance combined-metric quadratic form dp^T S^{-1} dp: was a full
eigendecomposition per station (`np.linalg.eigh`, ~85% of the 4.1 s/pair path at
N=2000). When sigma_pa > 0 (operational default 0.5), S = (PSD covariances) +
sigma_pa^2 I is strictly positive-definite, so the form is a direct batched
`np.linalg.solve` — identical to floating tolerance, ~3x cheaper. The eigh path
is kept only for sigma_pa == 0 (a zero-variance direction must read +inf =
"clear"); a shared `_quad_form_inv` branches on sigma_pa.

| N (station pair) | eigh (old) | solve (new) | speedup |
|---|---|---|---|
| 800 | 754.1 ms | **247.1 ms** | **3.05×** |

Parity (tests/test_clearance_mahalanobis_solve, 5): the solve form equals the
eigh form for random SPD S (max rel err 5e-15); end-to-end SF is unchanged vs a
forced-eigh reference (max abs err 1.8e-15, +inf clear-direction mask identical);
the sigma_pa == 0 fallback preserves the degenerate +inf semantics. The remaining
clearance lever (the O(n^2) broadphase + the IscwsaClearance per-station scipy
closest-point search) is tracked separately.
## 2026-07-19 · `feat/iscwsa-analytic-closest-point` · Python 3.12, dev machine

IscwsaClearance found the closest point on the offset well per reference station
by a scipy Powell search over MD (two segments each), re-interpolating the
min-curvature arc many times -- ~85% of the ~5 s/pair path at N=2000. A
min-curvature segment is a planar circular arc, so the closest point is
closed-form (`_closest_x_on_arc`): maximise (Q-C)·(sin th * t0 - cos th * b) over
the arc, th* = atan2(qa, -qb) with an explicit endpoint pick when the optimum is
off-arc (a naive clamp lands on the wrong end when it wraps past +/-pi).

| N (station pair) | scipy (old) | analytic (new) | speedup |
|---|---|---|---|
| 2,000 (minimize_sf=False) | 5310 ms | **974 ms** | **5.5×** |
| 2,000 (minimize_sf=True)  | 5337 ms | **972 ms** | 5.5× |
| 3-case ISCWSA-like battery | 1180 ms | 367 ms | 3.2× |

Correctness: the ISCWSA reference-value validation (tests/test_clearance_iscwsa,
all 10 ACR wells, both minimize_sf, incl. the datum-shifted well 10) passes
unchanged; the analytic point is the true minimum so its achieved distance is
never worse than scipy's (which under-converges on flat minima).

Two subtleties that had to be right (pinned in tests/test_clearance_closest_point):
- FRAME: build the arc in the local (n, e, tvd) frame that `_get_nevs` /
  `_interpolate_pos_nev` use -- NOT `pos_nev`/`vec_nev`, which the header transform
  datum/grid-shifts (they coincide only for a default header; well 10's `pos_nev`
  tvd was 900 m off the local tvd).
- ENDPOINT WRAP: off-arc optima need an explicit f(0) vs f(dogleg) pick.

BOTH inner closest-point searches are now analytic -- `_get_closest_points` AND
the `get_sf_mins` interpolated-minimum refinement. (An earlier cut left the latter
on scipy after it "diverged from the ISCWSA reference"; that divergence was the
SAME pos_nev-vs-local FRAME BUG, not a real difference -- the analytic point is
the true minimum, so once re-applied in the local frame it matches the reference
exactly.) Only the OUTER `get_sf_mins` optimisation stays scipy: it minimises the
*separation factor* over reference MD (dist/EOU with a covariance projection), a
genuine 1-D optimisation with no clean closed form, and fires only at local SF
minima. The O(n^2) broadphase remains a separately-tracked lever.

## 2026-07-19 · `feat/survey-composition-0.19` · Python 3.12, dev machine

`SurveyComposition` — tie a wellbore's ordered survey sections (per-section
tool/error model) into one Survey with per-component-correct covariance carry
(systematic stays correlated within a tool run, restarts at a tool change;
global geomag carries per share-mode; random correlates per error model).

| operation | result |
|---|---|
| compose 3-tool / 202-station wellbore | **16.4 ms** |
| single-survey error model, same geometry | 3.8 ms |

~4x a single survey — the compose builds each error component as its correlated
run (multiple Survey+error evaluations; model-definition parses are cached from
the 0.19 cache). Absolute cost is interactive-fine for the single-wellbore path;
errors stay lazy at the Survey level. Structural validation: same-tool sections
compose EXACTLY equal to the single continuous survey (the tie/leg formulation
of the ISCWSA error-model definition, §4.7), tool-change carry + systematic
independence + sidetrack ties pinned in tests/test_survey_composition.py (16).

## 2026-07-19 · `feat/edm-ipm-parser` · Python 3.12, dev machine

EDM/COMPASS IPM import: DP_TOOL_TERM -> live error models (interpreter path)
+ per-section IPM dicts in SurveyComposition.

| operation | result |
|---|---|
| `parse_edm_ipm` (Volve.xml, 211 MB, streaming regex) | 2.0 s cold / 0.2 s warm |
| standard `ISCWSA MWD Rev5.11`, 100 stations (regression check) | 3.2 ms/survey (baseline 3.8 ms — no regression) |
| EDM IPM dict model (23 grouped terms, interpreter), 100 stations | 7.1 ms/survey |

The dict-model path costs ~2x the native hand-coded MWD path (every term goes
through the formula interpreter + intermediates) — new functionality, not a
regression of an existing path; the standard-model hot path is unchanged. The
lowercase-vocabulary bindings additions to `_call_interpreter` are dict inserts,
not measurable at this scale.

## 2026-07-20 — branch fix/ipm-units-gating (deg.nT unit fix + IPM inc-gating + Rot6Axis model)
- ISCWSA MWD Rev5.11, 167 stations, 20 reps: **3.6 ms/survey** (in line with prior baseline;
  the changes add one dict-lookup wrapper per term evaluation and a converter-side key rename —
  no hot-path regression). Machine: miner5, .venv312.

## 2026-07-26 — branch feat/clearance-interior-cov (A-5 influx-temperature correction, 0.26.0rc10)

The corrected revision `"column-mean-2026"` (now the default) solves a FIXED POINT for the
influx-column height before it can evaluate the gas state — the column height is an output of
A-2 and the density depends on the temperature at the column's mid-depth. ~7 gas-property
solves per evaluation, and the closed form asks for the same station three times per case
(`constant_A`, its `resolve_gas_properties`, and the result's own reporting).

| `drill_kick`, SPE-208788 Table-1 case | ms/call |
|---|---|
| `spe-208788` (legacy path, unchanged) | 0.025 |
| `column-mean-2026`, un-memoised | 0.140 |
| `column-mean-2026`, memoised — repeat inputs | 0.026 |
| `column-mean-2026`, memoised — 300 DISTINCT inputs (worst case, no reuse) | 0.066 |

Un-memoised was **5.6x** the legacy path. `_influx_temperature_impl` is `lru_cache`d on the
scalars the fixed point actually depends on, which collapses the 3 calls/case to 1 and makes an
envelope sweep over a repeated station free. Worst case — every call a distinct input, so the
cache never hits — is 2.6x the legacy path at 0.066 ms/call, which is well inside the envelope
and monotonicity sweeps' existing budget. Machine: this dev box, .venv312.

## 2026-07-26 — `ErrorModel.cov_nev_at` source-stacked evaluation (0.26.0rc12)

a downstream consumer profiled `MahalanobisClearance` in `mode="exact"`: one check on a real path
(124 candidate stations x 12 Volve offsets) spent **2.98 s of 3.67 s — 81% — inside 8906 SCALAR
`cov_nev_at` calls**, issuing 219k `np.outer` products. It is the binding constraint of every
`plan()` call that reports an oracle SF.

They asked for a batched `cov_nev_at(md_array)`. **Refused — that is a vectorised public entry
point, which is a batch consumer's / outside this module's scope -- the public entry point is scalar by design.** What core
CAN do is make the SCALAR call faster, and almost all of the win was available there: a single
interior evaluation touches all 35 sources of the default MWD model, and at 3-vector sizes numpy's
per-call overhead dominated the arithmetic.

`_interior_stacks()` stacks the per-source arrays once per model (model-invariant — no per-leg or
per-query cache, deliberately), so one query is evaluated with a handful of einsums over the SOURCE
axis instead of a 35-iteration Python loop with ~37 `np.outer` products.

| `cov_nev_at`, 100-station MWD Rev5.11 survey, 35 sources | ms/call |
|---|---|
| per-source loop (rc11) | 0.540 |
| source-stacked (rc12) | **0.097** |

**5.6x on the scalar path.** Public API unchanged and still strictly scalar — one measured depth in,
one (3, 3) out.

Numerical parity vs the per-source loop: **max 2.7e-16 relative (~1 ulp)** across 137 queries on
build / vertical / horizontal surveys. NOT bit-identical, and the reason is only that einsum
accumulates the sum in a different order than a sequential `+=`. Pinned by
`test_source_stacked_form_matches_a_per_source_reference`, which re-implements the replaced loop
verbatim inside the test and asserts < 1e-14 relative.

Projected on pathfinder's profile: 2.98 s -> ~0.53 s, so the check goes 3.67 s -> ~1.2 s, **~3x**
against the ~3.6x ceiling they estimated for a 10x — i.e. most of the win, without a batched public
API. Machine: this dev box, .venv312.

## 2026-07-26 — `SurveyComposition` propagation sharing (0.26.0rc13)

a downstream consumer profiled their programme setup and found `SurveyComposition` was **93% of it**
(28.71 ms of 30.95 ms), running **8 full `ErrorModel` propagations for a 2-section compose**.

Cause: `_compose_component` is called once per covariance component (global / systematic / random /
well), and each re-built the run's `Survey` — but `attr` only selects which covariance to READ off
a build. The only thing that changes the build is the severed-run `_tmd_datum` override, so a run
needs at most TWO propagations, not one per component. The multi-model `_run_component_per_term`
path had the same shape: one `Survey` per group, rebuilt for each of the three components that
reach it. Three quarters of every result was discarded.

Fixed by caching the run builds on `(run, severed, start_nev)`, scoped to a single `survey()` call
so a mutated composition can never read a stale run.

| 60-station, 2 groups, tie at station 30 | propagations | ms |
|---|---|---|
| single-model, before | 5 | 13.9 |
| single-model, after | **3** | **8.6** |
| multi-model (Rev4 + Rev5.11), before | 8 | 19.7 |
| multi-model, after | **5** | **13.4** |

**BIT-IDENTICAL** on `cov_nev`, `cov_nev_global`, `cov_nev_systematic`, `cov_nev_random`,
`cov_nev_well` — required, not merely observed: this is an MC-gated path that probcol's
`covariances_at(programme=)` anchors on, so a change that merely agreed closely would be a parity
failure. Pinned by `test_composition_does_not_re_propagate_per_covariance_component`.

Not "chasing performance" — computing the same thing four times is a defect (project-owner ruling, 2026-07-26).
Machine: this dev box, .venv312.

## 2026-07-27 — kick tolerance: a 245x regression caught by this gate, and a 4x win (0.27.0rc1)

**The gate did its job.** Two changes were about to ship together: the analytical solver got
genuinely faster, and `drill_kick` got 245x SLOWER. Only the first was noticed until the
benchmark was run, because no test measures time.

| operation | 0.26.0 | branch, before fix | after fix |
|---|---|---|---|
| `drill_kick` (H-Y auto Z), harness (repeated inputs) | 49.3 us | 23851.9 us | **12.9 us** |
| `drill_kick`, DISTINCT inputs (batch/sweep reality) | 0.049 ms | 24.2 ms | **0.321 ms** |
| `analytical_kick_tolerance` [exact] | 5.2 ms | — | **1.2 ms** |
| `analytical_kick_tolerance` [conservative] | 5.4 ms | — | **1.4 ms** |
| `migrate`, `max_influx_circulated` | unchanged | | |

**Note the harness overstates the win.** It calls `drill_kick` with the SAME inputs 20 times,
so the new memo hits across calls and reports 12.9 us. Distinct inputs -- what a sweep or a
batch endpoint actually does -- give 0.321 ms, still 6.5x slower than 0.26.0. Quote the
distinct-input number.

**Cause of the regression.** The `bubble-state` revision replaced a lumped influx state with
the bubble's OWN state, which meant integrating the gas column. `_bubble_state` marched 256
points x up to 64 outer iterations, calling Hall-Yarborough at every point: profiling showed
**51,300 H-Y solves for 20 `drill_kick` calls, 2,565 per call**. It is also called TWICE per
case -- once by `constant_A` for the buoyancy deficit, once by `influx_column` for the
reported state.

**Two fixes.**

1. *Memo, bit-identical.* `_bubble_state` is asked for the same bubble twice per case.
   Keyed on `_inputs_key`, built from EVERY field by reflection rather than a hand-listed
   subset -- a hand-listed key is precisely how a memo silently breaks a frozen revision
   (omit `model_revision` and a result computed under one revision is served for another).
   2x, and free.
2. *Closed-form column.* The column is exponential, so integrate it in closed form instead of
   marching it. With `rho ~ c.P`, `dP/dh = g.rho` gives `P(h) = P_top.exp(k.h)`, and

       INT rho dh = dP / g   =>   rho_bar = (P_bottom - P_top) / (g.H)

   which is EXACT for any internal distribution -- the identity already asserted by
   `test_the_bulk_density_is_pinned_by_the_endpoint_pressures`. `c` is taken at the column's
   mean pressure and mid-height temperature, the treatment `analytical.py` already uses on
   the same column. 37x.

**Accuracy cost: 0.04%**, and downward (28.6606 -> 28.6492 bbl on the Table-1 case), i.e. the
conservative side. Permitted because `bubble-state` is UNRELEASED -- the freeze binds on the
existence of a published artefact, not on elapsed time.

**Every validation anchor held**, checked explicitly rather than inferred from a green suite:

| anchor | before | after |
|---|---|---|
| NOGEPA-50 Sec 3.2 | identity, 1e-12 | identity, 1e-12 |
| independent worked case (unpublished) | -0.76% | -0.76% |
| SPE-208788 worked example | -2.72% | -2.72% |
| SPE-202426 Nassab, stated ratio | +2.41% | +2.42% |

NOGEPA stays EXACT because it assumes Z = 1 and isothermal, so `c` is constant and the
exponential column is not an approximation there at all.

**Harness fixed in the same change.** It now times DISTINCT inputs and reports both figures,
so a memo can never again flatter a regression into invisibility:

```
drill_kick, DISTINCT inputs               319.6 us   <- quote this
drill_kick, repeated input (memo)          12.5 us
analytical_kick_tolerance [exact]           1.2 ms
analytical_kick_tolerance [conservative]    1.3 ms
migrate (n_steps=100)                     325.3 ms
max_influx_circulated [thorough]         4206.1 ms
max_influx_circulated [fast]              722.6 ms
```

Machine: this dev box, .venv312.

---

## 2026-07-27 — rc3 defect fixes (Findings A + D). No perf change; harness un-broke.

Three additive changes, none on a hot path: two `bool` flags on `KickResult`
(`bubble_length_limited`, `capacity_negative`) and an already-fractured guard in
`analytical_kick_tolerance` that only runs in the `not isfinite(v_star)` branch — i.e.
only when the breach search found no candidate at all, which is not the normal path.

```
drill_kick, DISTINCT inputs               160.3 us   <- quote this
drill_kick, repeated input (memo)          13.0 us
analytical_kick_tolerance [exact]           1.2 ms   <- API/GUI path
analytical_kick_tolerance [conservative]    1.4 ms
migrate (n_steps=100)                     328.5 ms
_max_influx_circulated [thorough]        4240.4 ms
_max_influx_circulated [fast]             731.1 ms
```

Against the previous entry: `drill_kick` 319.6 -> 160.3 us is the profiling work already
logged above, not these fixes; the analytical path is flat at 1.2/1.4 ms. Nothing here
moved a number.

**The harness itself was broken and silently so.** It still imported
`max_influx_circulated`, which this cycle renamed to `_max_influx_circulated` when the
marching oracle went private — so the gate's own script died on import. It was not run
between that rename and now. Fixed in this change. A blocking gate that cannot start is
indistinguishable from a gate that passes if nobody reads the output, which is the second
time this cycle the benchmark gate has caught something only because it was actually run.

Machine: this dev box, .venv312.

---

## 2026-07-27 — MAASP added (rc4). Flat.

`maasp()` is a handful of `np.interp` evaluations over the exposed-depth set the
analytical solver has already built (`_env_d` is passed straight in as `check_depths`,
so the candidate set is not recomputed). It runs once per `analytical_kick_tolerance`
call; on `drill_kick` it is two multiplications.

```
drill_kick, DISTINCT inputs               165.6 us   (was 160.3 -- noise)
drill_kick, repeated input (memo)          13.8 us
analytical_kick_tolerance [exact]           1.3 ms   (was 1.2 -- at the rounding edge)
analytical_kick_tolerance [conservative]    1.4 ms   (was 1.4)
migrate (n_steps=100)                     334.2 ms
_max_influx_circulated [thorough]        4333.5 ms
_max_influx_circulated [fast]             745.2 ms
```

Not claiming these as improvements or regressions: the harness prints one decimal at
1.2-1.3 ms, so the analytical move is at the resolution limit. Nothing here is a real
change and nothing was optimised.

Machine: this dev box, .venv312.

---

## 2026-07-28 — migrate() env+SICP merged; bit-identical, 1.10x. 0.27.1rc1

Profiling flagged `migrate()` as redoing work each step. At `n_steps=100` —
1790 `pressure_at_depth` calls, 45,170 Hall-Yarborough Z solves:

```
  loop   1582 calls  (88.4%)   15.8 per step   <- damped fixed point on P_rep
  env     100 calls  ( 5.6%)    1.0 per step
  sicp    108 calls  ( 6.0%)    1.1 per step
```

**Fixed here (the safe 6%):** the envelope check and the SICP value had identical gas
geometry and identical kwargs, and `pressure_at_depth` is vectorised over depth — so
calling it twice integrated the same gas column twice. Now one call with the surface
depth prepended.

**Parity against the RELEASED 0.27.0 from PyPI, same case, same process:**

```
0.27.0 (PyPI)   307.5 ms
branch          279.6 ms      1.10x

sicp   bit-identical: True      pbind  bit-identical: True
sidp   bit-identical: True      minfp  bit-identical: True
```

Bit-identical was verified, not asserted — same function, same inputs, batched.

**NOT fixed here, deliberately:** the 88% damped fixed point (`P = 0.5*(P + P_new)`,
~16 iterations, each a fresh 20-substep integration). Replacing it with an actual
solve is the ~70%, but it moves a value converged to 1e-4, so it goes in with all
five validation anchors re-measured — not in the same change as a bit-identical
refactor. See [[feedback-solve-dont-search]].

Machine: this dev box, .venv312.
