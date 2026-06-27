# Mesh vs pedal-curve anti-collision — working notes (running capture)

Capture for the planned paper. Local-only dev note. Started 2026-06-27.

## VALIDATION (done first — the model is correct)
welleng's `IscwsaClearance` (pedal / separation-factor) reproduces the **published
ISCWSA standard-set separation factors** to **≤0.5% relative** (worst Well 10
0.50%, most ≤0.26%; decision-critical min-SFs match to 3 dp: 03 0.456 vs 0.457,
06 1.029 vs 1.029, 11 0.226 vs 0.226, 10 −0.607 vs −0.607). Data:
`tests/test_data/clearance_iscwsa_well_data.json` (Rev4, has published `SF`);
Well 10 needs `kop_depth=900`. **So the pedal/SF collision model is sound — we
will not "look silly".** The mesh method is welleng's OWN (no published reference).

## The authoritative source: SPE-187073-MS (Sawaryn et al.)
"Well Collision Avoidance — Separation Rule" (Sawaryn, Wilson, Bang, Nyrnes,
Sentance, Poedjono, Lowdon, Mitchell, et al.). In `Reference/Wellbore Collision/`.
- SF = adjusted centre-to-centre distance D / a function of the **relative
  positional uncertainty** → build the **COMBINED ellipsoid** (covariance
  Σ_ref + Σ_off — i.e. RSS) and take the **pedal curve of that combined ellipsoid**.
- Constants: **k = 3.5**, surface margin Sm = 0.3 m, σ_pa = 0.5 m. MASD = the D
  at which SF=1; ADP = allowable deviation from plan.
- PRD (pedal radial distance) construction in **Appendix C**; pedal radius along a
  unit direction u = √(uᵀ Σ u) (the support function of the ellipsoid).
- Other refs we have: SPE-184644-PA, SPE-116155-PA, SPE-121040-MS,
  Definition_of_the_ISCWSA_Error_Model.pdf, Stockhausen ISCWSA-43.

## PIVOT (2026-06-27, validated): mesh value = SPEED; problem = CONSERVATISM
The original "pedal misses collisions" thesis is DEAD — validated false. Across the
ISCWSA set the statistically-correct collision measure (Mahalanobis of the offset
within the kσ **combined** ellipsoid `√(dᵀΣ⁻¹d)`, = probcol's chi²) is **≥ the
pedal SF everywhere** → the pedal rule is CONSERVATIVE (over-states risk) by design
(it uses the support function `√(uᵀΣu)` ≥ the true ray-boundary). welleng's mesh
(two independent kσ ellipsoids, LINEAR `k(σr+σo)`) is MORE conservative still.
Ordering: **linear-mesh < pedal < Mahalanobis(truth)**.

Quantified over the ISCWSA set: current mesh over-demands standoff by **1.4–2.5×**
vs truth (flat **√2** from linear-vs-RSS combination + another ~1–1.8× from
support-function-vs-true-surface). The user uses the mesh in AWTP because it is
**fast for binary go/no-go** (fcl `in_collision`, no SF recreation) — that speed is
the real value.

**New thesis: validate the mesh as a fast collision checker and REDUCE its
conservatism.** Fix = build ONE ellipsoid with the **combined covariance**
Σ_ref+Σ_off (other well = centerline + radius). The mesh tests the true ellipsoid
SURFACE (= the exact Mahalanobis kσ boundary, not the support-function
approximation) → less conservative than even pedal, keeps fcl speed. Target =
SF_maha (computed; reduces standoff 1.4–2.5× vs current mesh).

**Scope of THIS paper — kσ geometric boundary only (user, 2026-06-27):**
- This paper lives in the **kσ-boundary / SF domain** (matching SPE-187073). The
  **Mahalanobis distance** (Mahalanobis 1936, "On the generalised distance in
  statistics") is the *exact* metric the SF approximates; pedal uses the support
  function `√(uᵀΣu)`, the mesh uses the true ellipsoid surface = exact Mahalanobis.
  So `SF_maha` is the geometrically-exact **kσ boundary** (less conservative than
  pedal), NOT a probability.
- **Reference to weave in:** Mahalanobis (1936); the chi²(3 DOF) ellipsoid
  probability content for k selection. Standard statistics — fine to cite.

## FOLLOW-UP PAPER — probcol — DO NOT REFERENCE IN THIS PAPER
- **ASSUME probcol DOES NOT EXIST when writing this paper. Do not mention it,
  probability-of-collision integration, or intercept/relief wells.**
- probcol is the user's SEPARATE side project: it **integrates the probability
  density over the collision region** (chi², 3 DOF) to an actual *probability of
  collision*, used to **MAXIMISE collision probability** (e.g. planning an
  **intercept / relief well** — the opposite goal). It captures the probability
  volume a kσ boundary excludes.
- It is a **follow-up paper that will cite THIS one**, not vice-versa. Keep the
  dependency one-way. At most, this paper may give a single generic "a full
  probabilistic-collision treatment is future work" nod WITHOUT naming probcol or
  developing it — or omit entirely.

## (superseded) earlier thesis — kept for the record
The "pedal misses collisions" framing below is RETRACTED (see PIVOT above).

NOT "pedal is wrong" — it reproduces ISCWSA. Rather: **the mesh method quantifies
and removes the separation rule's OWN DOCUMENTED geometrical limitations**
(SPE-187073, "Geometrical Limitations"):
- **Limitation A** — the planar wrong-side boundary: a *curved* offset well lies
  closer than the plane, so the rule **UNDER-estimates** the probability of being
  on the wrong side of the (curved) offset well. A documented false-negative.
- **Limitation B** — the closest-approach / travelling-cylinder point is NOT the
  point of highest collision probability for **flattened (eccentric) ellipsoids**;
  severity grows with ellipsoid flattening. Exact only for SPHERICAL confidence
  regions.
The mesh (true 3-D closest-surface distance + overlap) is exact for arbitrary
geometry/orientation, so it resolves A and B. That is the contribution.

## CRITICAL correctness issue (must fix before any comparison/figure)
welleng's `MeshClearance` builds **two independent kσ ellipsoids** and tests
surface overlap → overlap at `k·(σ_ref + σ_off)` (LINEAR sum). The separation
rule uses the **combined ellipsoid** `k·√(σ_ref² + σ_off²)` (RSS / quadrature —
statistically correct for the relative position, Var(P_off−P_ref)=Σ_ref+Σ_off).
Linear > RSS (√2× in the symmetric case), so the mesh-as-built is **~√2 over-
conservative** vs the validated rule. The earlier "Well-06 +10 m false negative"
(pedal 1.27 vs mesh 0.86; band = the 37.9–53.6 m window between RSS and linear
thresholds) is **THIS ARTIFACT, not a geometric miss** — discard it.
**Fix:** mesh the COMBINED-covariance ellipsoid (Σ_ref+Σ_off) at one well vs the
other as a point + hole radius (RSS-consistent). THEN any residual mesh-vs-pedal
difference is the genuine limitation-A/B geometry. (probcol already does proper
probabilistic collision via chi2 shells — cross-check against it.)

## Plan
1. [done] Validate pedal SF vs ISCWSA published (≤0.5%).
2. Rebuild the mesh comparison RSS-correct (combined ellipsoid).
3. Re-hunt a TRUE geometric false negative in the limitation-A (curved offset) or
   limitation-B (flattened-ellipsoid) regime — using the ISCWSA wells (+ minimal
   realistic shift, surface elevation preserved) where possible.
4. Cross-check against probcol's probabilistic collision.
5. Figures (publication, user wants close-ups): wellbores AT collision; the
   **pedal curve of the combined ellipsoid** drawn (PRD per SPE-187073 App. C —
   pedal radius = √(uᵀΣu)); true closest-surface distance; limitation A (curved
   offset vs plane) and B (flattened ellipsoid) illustrated. Reuse probcol
   `polygon_fit_explainer.png` for the inscribe/circumscribe/true-ellipse point.
6. Explain (briefly) how to draw the pedal curve + cite SPE-187073 (and the
   others). k=3.5, Sm=0.3, σ_pa=0.5.

## Combined-cov mesh — VALIDATED, ready to implement
- AWTP meshes BOTH wells' uncertainty (user) → two-ellipsoid LINEAR overlap → √2+ over-conservative. Confirmed: at σ=2.445, Well 06 shifted +5…+15 m the current linear mesh says HIT while the combined-cov mesh AND the Mahalanobis truth say CLEAR (5.9–15.4 m of real margin). Linear mesh = **false collision alarms** → AWTP rejects safe trajectories.
- **Combined-cov mesh == Mahalanobis kσ boundary** (hit/clear identical at every shift). Prototype: `scratchpad/combined_mesh_proto.py`.
- **Design (frame-clean, NEV):** for each ref station, combined cov `Σ_ref[i] + Σ_off[nearest]` (NEV); build the kσ ellipsoid mesh directly from the 3×3 NEV cov (eigendecomp → surface → `trimesh.convex_hull`, inflate by `R_ref+R_off+Sm`) — bypasses the HLA machinery; test vs the OTHER well's **centreline** (zero-cov). Put ALL combined uncertainty on ONE mesh (two separate ellipsoids can only ever overlap at the linear sum, never RSS).
- **API options:** (A) `MeshClearance(combined_cov=True)` mode (pairwise); (B) a `combined_cov_mesh(ref, off, k)` builder returning trimesh objects for AWTP's `CollisionManager`; keep the analytic Mahalanobis SF as the validation oracle/test. For AWTP this means per-pair meshing (the combination is pairwise) — document the cost.
- Validation oracle: `scratchpad/rss_correct.py` (`SF_maha`), already matches.

## DELIVERED — the conservatism fix (branch `perf/clearance-mesh`)
- **`MahalanobisClearance(IscwsaClearance)`** (commit 4b88b4f) — exact kσ boundary
  of the combined ellipsoid, `SF=√(d'ᵀ(Σr+Σo)⁻¹d')/k`. Analytic, vectorized, no
  mesh/fcl → the fast AWTP go/no-go. Reuses the validated closest-approach
  machinery; only swaps support-function → Mahalanobis. SF_maha ≥ SF_pedal on the
  whole ISCWSA set (less conservative; up to ~1.9×). Test:
  `test_mahalanobis_less_conservative_than_pedal`.
- **`combined_cov_mesh(survey, other, k, ...)`** (commit e25cbbf) — builds ONE
  trimesh tube carrying Σ_survey+Σ_other + combined radii, for the AWTP
  `CollisionManager` (test vs the other well's centreline). Fixes the two-ellipsoid
  linear over-conservatism. Validated: ISCWSA Well 06 nudged 10 m → linear mesh
  HIT (false alarm), combined mesh clears (matches Mahalanobis). Tests: returns a
  mesh; never more conservative than linear; clears the linear false alarm.

## CORRECTNESS milestone (2026-06-27) + the value proposition
- `MahalanobisClearance` now matches the validated pedal rule on EVERY ISCWSA
  hit/clear verdict (03,04,09,10,11 hit incl. the Well-10 sidetrack; 01,02,05,06,
  07,08 clear) while being **>= pedal at the governing point** (less conservative
  on the margin). Two real bugs were found and fixed (a same-point discrepancy
  led to them): (1) it used the Euclidean closest-approach point not the
  min-Mahalanobis point (SPE-187073 limitation B) and missed between-station
  crossings → **interpolate BOTH wells + search min-Mahalanobis**; (2) it used
  `pos_nev` (recomputed) not the authoritative `n/e/tvd` the rule uses. Plus the
  `sigma_pa` project-ahead floor for degenerate (sidetrack) covariance.
- **THE VALUE = reduced conservatism (primary):** the rule is conservative by
  design; the exact combined-ellipsoid boundary lets you drill closer SAFELY →
  a bigger **decision space**. This is what interests White Space (more feasible
  trajectories to optimise over), as well as operators (tighter, still-safe
  anti-collision). **Speed = secondary** (binary go/no-go at batch/AI-planning
  scale — moot for one-off checks, but where the field is heading; also of
  interest to White Space). Position the paper around CONSERVATISM/accuracy.
- **Position as THE method to use** (not merely a diagnostic): present the
  correct combined-covariance, min-Mahalanobis k-sigma boundary as the
  recommended replacement for the support-function pedal rule. Reproducible via
  the open welleng code (don't need to ship diagnostic datasets).
- Well 10: with correct positions it IS a genuine hit (the earlier "separated
  below KOP" was the pos_nev bug). The "don't scan a sidetrack until confirmed
  separated" convention is a `kop_depth`/scan-region matter, separate from the
  metric.

## QUOTE BANK (verbatim from the literature — verify page nos. when writing)
The literature already makes our case — lead with these:
- **SPE-116155:** "Unplanned collisions between oil wells can have catastrophic
  results, but **overconservative separation rules also have a cost**." Also:
  "all of these have limitations that may not be clearly understood … many
  computations currently in use fail when the two wells are parallel."
- **SPE-121040:** "…provide an adequate margin of error without being **too
  conservative and placing unnecessary restrictions on well-design options**."
  And: "This is usually a conservative assumption that **overestimates the size
  of the EOU of the relative displacement vector**." And: "modified to
  accommodate real-world geophysical and **economical considerations**."
- **SPE-184644:** "**More advanced methods that overcome such limitations are
  impractical for general application because of high conceptual or
  computational complexity**." ← our fast (35 ms), open, validated method
  REFUTES this. Also: "Is conservative for situations in which the method is not
  completely valid."
- **SPE-187073 (Sawaryn et al.):** "These limitations may result in **significant
  under, or over-estimation** of the probability of being on the wrong side of
  the offset well." "The higher the value, the more conservative is the rule."
  (limitations A & B; pedal/support-function basis.)

## REFINED PLAN (user, 2026-06-27)
1. **Economic argument is STRONGER than "decision space"** — over-conservatism
   makes operators either (a) **walk away from wells they could safely drill**,
   or (b) **do extra / more-sensitive surveying (gyro, infill stations)** which
   **costs more AND pushes out first production → serious negative well
   economics**. Ground it in the SPE-116155 / SPE-121040 quotes above.
2. **Fully reproducible — "do what we wish SPE papers did":** ship the diagnostic
   data with the paper (the public ISCWSA AC standard set + per-station SF
   outputs for pedal AND Mahalanobis + the worked Well-06 numbers), so anyone
   can reproduce every figure/number. (Contrast SPE-194099's proprietary wells.)
   This paper IS reproducible (probcol follow-up is the separate one).
3. **Figures: only what's relevant to the CORRECTED thesis** — drop the old
   "false negative" framing entirely. Include BOTH **figurative/schematic**
   (support function vs true ray-boundary; combined-ellipsoid; why pedal is
   conservative) AND **factual/actual** (the conservatism plot from the ISCWSA
   data; Well-06 worked example) figures showing the DIFFERENCE.

## PAPER CONTENT (capture — what the paper argues)
1. **Validate first:** welleng's `IscwsaClearance` reproduces published ISCWSA
   separation factors ≤0.5% (SPE-187073 separation rule; k=3.5, Sm=0.3, σpa=0.5).
2. **The pedal rule is conservative by design** (support function `√(uᵀΣu)` ≥ true
   ray-boundary) — it does NOT miss collisions. The "mesh catches missed
   collisions" idea was tested and is FALSE; dropped.
3. **welleng's two-ellipsoid mesh is over-conservative** (LINEAR `k(σr+σo)` vs RSS
   `k√(σr²+σo²)`) → false collision alarms → AWTP rejects safe trajectories.
   Quantified 1.4–2.5× excess standoff on the ISCWSA set.
4. **Fix:** combined-covariance (one ellipsoid, RSS) — exact via Mahalanobis kσ
   boundary; less conservative than even the pedal rule; keeps mesh binary-check
   speed. `MahalanobisClearance` (analytic) + `combined_cov_mesh` (trimesh).
5. **Mesh's real value = speed** for binary go/no-go in automated planning (no SF
   recreation, which the perf work showed is the slow part).
6. Figures: surface construction (inscribe/circumscribe/true ellipse — probcol
   `polygon_fit_explainer.png`); how fcl computes triangle-mesh distance; the
   linear-vs-combined conservatism (Well-06 sweep); validation table vs ISCWSA.
7. **References:** SPE-187073 (Sawaryn et al., separation rule + pedal curve +
   geometrical limitations A/B); Mahalanobis (1936, generalised distance);
   chi²(3 DOF) for k; SPE-184644-PA / 116155-PA / 121040-MS as available;
   ISCWSA error-model definition. **NOT probcol (see fence above).**
8. Scope = kσ geometric boundary; full probability-of-collision is the probcol
   follow-up (do not reference).

## Engineering done this session (branch `perf/clearance-mesh`, off main/0.12.1)
- `7f09ec7` perf: MeshClearance closest-point uses `_interpolate_pos_nev` (no
  per-call Survey) → ~5× (9.3→1.7 s), SF identical, tests pass.
- `17a2ccd` feat: `polygon_fit` (default **circumscribed**) on WellMesh +
  MeshClearance; `tests/test_mesh_clearance.py`. NOTE: circumscribe fixes the
  POLYGON under-count (2nd-order); it does NOT fix the RSS-vs-linear issue (1st-order).
- Profiling: fcl negligible; cost was the SF recreation. Recreating fcl = packaging
  win only.

## Open
- [ ] Rebuild RSS-correct mesh comparison; re-hunt limitation-A/B false negatives.
- [ ] Polished close-up figures (pedal curve of combined ellipsoid; limitations A & B).
- [ ] Cross-check vs probcol probabilistic collision.
- [ ] Write paper md; cite SPE-187073 + others; brief pedal-curve construction.
- [ ] Push/PR `perf/clearance-mesh` (after `git fetch`; base main/0.12.1).
