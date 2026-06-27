---
title: "Reducing Conservatism in Wellbore Anti-Collision: an Exact, Validated, Open-Source Combined-Ellipsoid Separation Method"
author: "Jonathan Corcutt (welleng)"
date: "2026"
geometry: margin=2.2cm
fontsize: 10pt
header-includes: \usepackage{amssymb}
---

**Preprint — Version 1.0 (2026-06-27)**

---

## Abstract

Wellbore anti-collision is governed by the ISCWSA separation rule (Sawaryn et al., SPE-187073-MS), which expresses risk as a dimensionless separation factor (SF) and is the industry standard for permitting drilling near existing wells. The rule is, by construction, conservative: it characterises the combined positional-uncertainty ellipsoid of the reference and offset wells by its *support function* (the pedal-curve / tangent distance) in the centre-to-centre direction, which always over-states the ellipsoid's true reach toward an off-axis offset. The literature acknowledges this — "overconservative separation rules also have a cost" (SPE-116155) — and that more accurate methods have been considered "impractical for general application because of high conceptual or computational complexity" (SPE-184644).

We show that the conservatism is exactly quantifiable. The separation factor uses the support function $\sqrt{\mathbf{u}^\top\Sigma\,\mathbf{u}}$ where the statistically correct boundary is the Mahalanobis distance $\sqrt{\mathbf{d}^\top\Sigma^{-1}\mathbf{d}}$; by the Kantorovich inequality the rule's factor is always less than or equal to the exact one, with the gap growing with ellipsoid eccentricity and approach obliquity. We implement the exact boundary as a general, step-free **minimum-Mahalanobis distance between two uncertain parametric curves** — solved by broadphase (the curves' own stations) plus continuous narrowphase, analytic and needing no surface mesh — in the open-source library welleng. It is validated two ways: (i) welleng's separation-rule implementation reproduces the published ISCWSA standard-set separation factors to within $0.5\%$, and (ii) the exact method agrees with the rule on every collision/clear verdict of the standard set while being up to $1.47\times$ less conservative on the margin, matching an exhaustive reference to $<10^{-3}$ at $\approx 35$ ms per pair. Where a mesh is used (multi-well scenes) its surface is built *circumscribing* the ellipse, so the discretisation never under-represents uncertainty. All data, code and diagnostics are released for full reproducibility. The result lets operators safely reduce standoff — recovering wells that conservatism would forbid and avoiding the cost and production delay of unnecessary additional surveying.

**Notation.** $\Sigma=\Sigma_{\text{ref}}+\Sigma_{\text{off}}$ the combined relative-position covariance (NEV), $\mathbf{d}$ the centre-to-centre vector, $\mathbf{u}=\mathbf{d}/\lVert\mathbf{d}\rVert$, $k$ the confidence multiple ($k=3.5$ in the standard rule), $R$ the combined hole radii plus surface margin $S_m$, $\sigma_{pa}$ the project-ahead term.

---

## 1. Introduction

Drilling near existing wells requires a quantitative anti-collision decision: is the planned (reference) well safely separated from each offset well, given the positional uncertainty of both? The industry standard is the ISCWSA separation rule (Sawaryn et al., SPE-187073-MS), which reduces the decision to a dimensionless **separation factor** (SF): the ratio of the centre-to-centre distance to a minimum-allowable separation derived from the combined positional uncertainty. $\mathrm{SF}<1$ prohibits the activity.

The rule is deliberately conservative — a sound default for a safety-critical decision. But conservatism is not free. SPE-116155 states it plainly: *"Unplanned collisions between oil wells can have catastrophic results, but overconservative separation rules also have a cost."* SPE-121040 notes the need for *"an adequate margin of error without being too conservative and placing unnecessary restrictions on well-design options."* In practice, excess conservatism forces an operator either to **walk away from a well it could have drilled safely**, or to **commission additional or more sensitive surveying** (for example continuous gyro runs or infill survey stations) to shrink the uncertainty enough to pass the rule — which raises cost and **delays first production**, with material negative impact on well economics.

More accurate alternatives exist in principle but have been judged impractical: SPE-184644 observes that *"more advanced methods that overcome such limitations are impractical for general application because of high conceptual or computational complexity."* This paper's contribution is to show that the exact method is neither: it is a closed-form geometric quantity, it runs in milliseconds, and — crucially — it is *validated* and *released open-source* so that any operator can audit and reproduce it.

We make five contributions:

1. We identify and quantify the source of the separation rule's conservatism: the support function over-states the ellipsoid's reach (Section 3), and bound it exactly via the Kantorovich inequality (Section 5).
2. We validate welleng's separation-rule implementation against the published ISCWSA standard set to within $0.5\%$ (Section 4) — establishing that we compute the rule correctly before improving on it.
3. We give the exact combined-covariance, minimum-Mahalanobis $k\sigma$ method, searching the worst point over both interpolated wells, with a project-ahead floor for degenerate geometries (Section 6).
4. We validate it against the rule on the standard set (identical verdicts, up to $1.48\times$ less conservative) and show it is fast (Section 7).
5. We release the implementation, data and diagnostics for full reproducibility (Section 8).

## 2. The separation rule and its geometry

For a reference station the rule forms the **combined relative-position covariance** $\Sigma=\Sigma_{\text{ref}}+\Sigma_{\text{off}}$ — correct because the uncertainty in the *relative* position $\mathbf{P}_{\text{off}}-\mathbf{P}_{\text{ref}}$ of two independently surveyed wells is the sum of their covariances. The minimum-allowable separation along the centre-to-centre direction $\mathbf{u}$ is built from the **pedal radial distance** — the ellipsoid's support function

\begin{equation}
h(\mathbf{u}) \;=\; \sqrt{\mathbf{u}^\top \Sigma\, \mathbf{u}},
\end{equation}

the distance from the ellipsoid centre to the tangent plane with normal $\mathbf{u}$. The separation factor is

\begin{equation}
\mathrm{SF}_{\text{pedal}} \;=\; \frac{\lVert\mathbf{d}\rVert - R}{k\,\sqrt{h(\mathbf{u})^2 + \sigma_{pa}^2}},
\qquad \mathbf{u}=\frac{\mathbf{d}}{\lVert\mathbf{d}\rVert},
\end{equation}

with $k=3.5$, surface margin $S_m=0.3$ m (folded into $R$) and project-ahead $\sigma_{pa}=0.5$ m. The rule documents its own *geometrical limitations*: the closest-approach scan may not select the point of highest collision probability for flattened ellipsoids, and "these limitations may result in significant under, or over-estimation" (SPE-187073). It is, in the words of SPE-184644, *"conservative for situations in which the method is not completely valid."*

## 3. Why the rule is conservative

The support function $h(\mathbf{u})$ is the ellipsoid's extent measured to the *tangent plane* perpendicular to $\mathbf{u}$. The quantity that actually decides whether the offset lies inside the $k\sigma$ ellipsoid is the **Mahalanobis distance** of the offset in the combined-covariance metric,

\begin{equation}
m \;=\; \sqrt{\mathbf{d}^\top \Sigma^{-1}\, \mathbf{d}},
\qquad \text{collision} \iff m < k .
\end{equation}

For an eccentric ellipsoid approached off-axis these differ substantially: the tangent-plane distance over-states how far the ellipsoid actually reaches *along* $\mathbf{u}$. Figure 1 shows the geometry — the offset lies beyond the true ellipsoid boundary (clear) yet within the support-function reach the rule uses (flagged as a collision).

![Why the separation rule is conservative: the support function (pedal-curve reach) over-states the ellipsoid's reach toward an off-axis offset relative to the true boundary (Mahalanobis $=k$). The offset shown is clear in truth but flagged by the rule.](figures/why-pedal-is-conservative.png){width=82%}

## 4. Validation of the baseline

Before improving on the rule we confirm we compute it correctly. welleng's `IscwsaClearance` reproduces the **published** separation factors of the ISCWSA standard set of clearance scenarios (reference well plus eleven offsets) to within $0.5\%$ on every well — well inside the documented inter-implementation band. Table 1 lists the minimum separation factors.

| Offset | published min SF | welleng min SF | rel. err. |
|---|---|---|---|
| 01 | 1.400 | 1.400 | 0.12% |
| 02 | 3.627 | 3.627 | 0.12% |
| 03 | 0.457 | 0.457 | 0.12% |
| 04 | 0.397 | 0.396 | 0.23% |
| 05 | 1.195 | 1.195 | 0.26% |
| 06 | 1.029 | 1.029 | 0.01% |
| 07 | 1.633 | 1.633 | 0.06% |
| 08 | 1.272 | 1.272 | 0.16% |
| 09 | 0.010 | 0.010 | 0.12% |
| 10 | -0.607 | -0.607 | 0.50% |
| 11 | 0.226 | 0.226 | 0.12% |

Table: welleng's separation-rule implementation vs the published ISCWSA standard-set minimum separation factors, compared **at the tabulated survey stations** (worst-case per-station relative error $0.50\%$). welleng's between-station interpolation finds slightly lower true minima where the worst point lies between stations (Wells 09: $-0.089$; 11: $0.080$); these are used for the like-for-like comparison against the exact method in Section 7, not here.

## 5. The conservatism is exactly bounded

Writing $\mathrm{SF}_{\text{exact}}=m/k$ (the radii-adjusted Mahalanobis boundary) and ignoring the small $\sigma_{pa}$ floor, the ratio of the two factors in any approach direction $\mathbf{u}$ is

\begin{equation}
\frac{\mathrm{SF}_{\text{exact}}}{\mathrm{SF}_{\text{pedal}}}
= \sqrt{\big(\mathbf{u}^\top \Sigma\, \mathbf{u}\big)\big(\mathbf{u}^\top \Sigma^{-1}\, \mathbf{u}\big)} \;\ge\; 1,
\end{equation}

by the Kantorovich (Cauchy–Schwarz) inequality, with equality if and only if $\mathbf{u}$ is an eigenvector of $\Sigma$. The rule is therefore *provably conservative* — never optimistic — and the gap grows with ellipsoid eccentricity and approach obliquity. (A separate, larger conservatism arises if two independent $k\sigma$ ellipsoids are meshed and overlap-tested: their surfaces meet at the *linear* sum $k(\sigma_{\text{ref}}+\sigma_{\text{off}})$ rather than the root-sum-square $k\sqrt{\sigma_{\text{ref}}^2+\sigma_{\text{off}}^2}$, a further $\sqrt{2}$ in the symmetric case — avoided here by combining into one ellipsoid.)

A subtle but essential point for credibility: the reduced conservatism must come from the *method*, not from under-representing the uncertainty surface. welleng builds the discretised mesh surface **circumscribing** the ellipse — each $n$-vertex polygon is scaled out by $1/\cos(\pi/n)$ so its edges are tangent to, and the polygon fully contains, the ellipse (Figure 2). The surface therefore never under-counts uncertainty for the given $\sigma$; it errs conservative, while the *metric* is what reduces conservatism.

![Conservative surface construction. welleng's default polygon is circumscribed (scaled by $1/\cos(\pi/n)$) so it contains the true $k\sigma$ ellipse and never under-represents the uncertainty; an inscribed polygon would under-count it. ($n=8$ shown for clarity.)](figures/conservative-surface-construction.png){width=78%}

## 6. The exact method: minimum Mahalanobis distance between two uncertain curves

The problem is not drilling-specific. Each well is an *uncertain parametric curve*: a position $\mathbf{P}(s)$ and a covariance field $\Sigma(s)$ along a parameter $s$. The separation factor is the minimum, over both curves, of the radii-adjusted Mahalanobis distance in the combined metric:

\begin{equation}
\mathrm{SF} \;=\; \frac{1}{k}\,\min_{s,\,t}\; \sqrt{\mathbf{d}'(s,t)^{\top}\big(\Sigma_{\text{ref}}(s)+\Sigma_{\text{off}}(t)+\sigma_{pa}^2 \mathbf{I}\big)^{-1}\mathbf{d}'(s,t)},
\end{equation}

where $\mathbf{d}'(s,t)$ is the centre-to-centre vector shortened by the combined hole radii and surface margin. This is a continuous two-parameter minimisation, solved to a tolerance — *not* a search on an externally imposed grid. We solve it the way any geometric collision query does, in two phases:

- **Broadphase (locate).** Evaluate the factor at the curves' own survey stations — every reference station against every offset station. The sampling is the geometry's own, so there is no arbitrary step to choose, and the pairing is *exhaustive*, not a Euclidean nearest-neighbour shortlist: the minimum-*Mahalanobis* offset is generally not the Euclidean-nearest one (the rule's "limitation B"), so a shortlist would silently miss it.
- **Narrowphase (refine).** From the lowest broadphase candidates, minimise the factor *continuously* over the two curve parameters, bounded to the neighbouring segments. This converges to a set tolerance — there is no discretisation step, and the worst point, which generally falls *between* stations, is found exactly. Refining several candidates (not only the global broadphase minimum) prevents a sharp crossing from hiding between stations.

The project-ahead floor $\sigma_{pa}^2\mathbf{I}$ keeps the metric finite where a covariance is degenerate (e.g. a near-vertical sidetrack with negligible horizontal uncertainty), so a physical/radii collision is still detected — mirroring the rule's own $\sigma_{pa}$.

Both phases operate directly on the analytic ellipsoid, so there is no surface to mesh and no fixed step to choose: the resolution is **self-derived from the geometry** (the survey's own stations) and the result is **step-free** (an optimiser tolerance), scaling without a domain-specific tuning parameter. Validated against an exhaustive all-pairs reference sampled at $1$ m, the method agrees to better than $10^{-3}$ in separation factor on every well of the ISCWSA standard set, at $\approx 35$ ms per pair.

For *multi-well scenes* — one planned well against many offsets — a spatial index over the curves' bounding volumes restricts the work to nearby pairs, each then evaluated by the same analytic kernel.

**Mesh resolution ($n$).** A mesh realisation (for multi-well scenes) introduces one parameter the analytic method does not: the number of vertices $n$ per ellipsoid cross-section. Because the polygon is *circumscribed* (Section 5), this discretisation can only ever *add* conservatism, and by a closed-form amount — the radial over-count (the factor by which the polygon's reach exceeds the true ellipse) is $\sec(\pi/n)-1$, which vanishes as $n$ grows. Table 2 shows the trade-off against mesh-build cost. A typical $n=12$ over-demands standoff by $3.5\%$ — an order of magnitude below the $40$–$47\%$ conservatism the method removes from the rule — and is a sensible knee; $n=24$ brings it under $1\%$ for tight calls. The **analytic method carries no such parameter**: being a continuous optimisation it is exact (to tolerance) and the cheapest option, so it is preferred for the pairwise separation factor, the mesh being reserved for triangulated multi-well collision-manager scenes.

| $n$ | radial over-count | area over-count | mesh build [ms] |
|---|---|---|---|
| 6 | 15.5% | 33.3% | 69 |
| 8 | 8.2% | 17.2% | 87 |
| **12** (default) | **3.5%** | 7.2% | 125 |
| 16 | 2.0% | 4.0% | 164 |
| 24 | 0.9% | 1.7% | 240 |
| 48 | 0.2% | 0.4% | 475 |
| analytic (exact) | 0% | 0% | 35 (no mesh) |

Table: mesh discretisation over-conservatism (closed-form, from the circumscribed polygon) and build cost vs vertex count $n$. The analytic Mahalanobis method carries no discretisation parameter — it is exact (to optimiser tolerance) at the lowest cost.

## 7. Results

On the ISCWSA standard set the exact method returns the **same collision/clear verdict as the validated rule on every well** (Wells 03, 04, 09, 10, 11 collide; 01, 02, 05, 06, 07, 08 clear), while being **less conservative on the margin** — up to $1.47\times$ (Well 03: $\mathrm{SF}$ $0.46\to0.67$; Well 05: $1.20\to1.73$; Well 07: $1.63\to2.18$). Figure 3 contrasts the two across the set. The method is **fast**: $\approx 35$ ms per well pair versus $\approx 300$ ms for the pedal rule, suitable for the inner loop of batch and automated well planning.

Two conventions are worth noting on the deeply overlapping wells. Where the *centres* lie closer than the combined hole radii plus surface margin ($\lVert\mathbf{d}\rVert < R$) the wellbores physically intersect: the separation rule returns a **negative** factor (its linear numerator $\lVert\mathbf{d}\rVert-R$ goes negative, conveying the depth of overlap), whereas the Mahalanobis factor is non-negative by construction and floors at $0$. Both correctly report a collision; the sign is a property of the rule's linear form, not a disagreement. Well 10 is the reference well's sidetrack: its kickoff intersects the parent by design, so the scan begins below the kickoff (`kop_depth`) — applied identically to *both* methods, and consistent with the published standard-set value.

![Minimum separation factor on the ISCWSA standard set: the separation rule (pedal) vs the exact combined-ellipsoid Mahalanobis boundary. The verdict (relative to $\mathrm{SF}=1$) is identical for every well, but the exact factor is never smaller than the rule's — up to $1.48\times$ larger on the clear wells, i.e. that much less standoff demanded.](figures/pedal-vs-mahalanobis-iscwsa.png){width=90%}

The practical reading: where the rule reports, say, $\mathrm{SF}=1.2$ and would demand additional surveying or a redesign, the exact method may report $\mathrm{SF}=1.7$ — clear with margin — recovering a drillable well without acquiring more data.

## 8. The "impractical" objection, refuted

The standing argument against adopting an exact method is that *"more advanced methods that overcome such limitations are impractical for general application because of high conceptual or computational complexity"* (SPE-184644). Every plank of that objection fails for the exact $k\sigma$-boundary method.

- **Conceptual simplicity.** The exact boundary is the Mahalanobis distance (Mahalanobis, 1936) — a single standard quadratic form, $m=\sqrt{\mathbf{d}^\top\Sigma^{-1}\mathbf{d}}$, with collision when $m<k$. It is *simpler* than the rule it replaces, not more complex: the separation rule needs the support function **and** a Euclidean closest-approach scan that carries its own documented exceptions (limitations A and B); the exact method needs neither — one metric, no special cases, no geometrical caveats.
- **Computational cheapness.** The cost is one $3\times3$ linear solve (or eigendecomposition) per station. Measured end-to-end, the method runs in $35$ ms per well pair — an order of magnitude *faster* than our own implementation of the pedal rule ($300$ ms) and well inside the inner loop of automated trajectory planning. On current hardware the "high computational complexity" is unmeasurable.
- **Generality.** It applies unchanged to every geometry in the ISCWSA standard set — parallel, crossing, oblique, eccentric and the sidetrack — with no special-casing. The support-function / closest-approach machinery, by contrast, carries the limitations the rule itself documents, and SPE-116155 records that *"many computations currently in use fail when the two wells are parallel"*. The Mahalanobis metric is well-defined for every relative geometry, parallel included.

The objection likely conflates two different problems: the full probability-of-collision *integral over the uncertainty volume* (genuinely more involved, and deliberately out of scope here — Section 10) and the exact $k\sigma$-*boundary* (this paper). The boundary question is the very one the separation rule already answers — approximately. Answering it *exactly* is cheaper, simpler and more general than the approximation. The barrier to adoption was never complexity; it was the absence of a validated, openly auditable implementation. This paper removes that barrier.

## 9. Implementation and reproducibility

The method is implemented in the open-source library welleng (`welleng.clearance.MahalanobisClearance` and `combined_cov_mesh`), with the validated separation rule (`IscwsaClearance`) for comparison. Following the spirit of what reference papers too often omit, *everything required to reproduce every number and figure in this paper is released*: the input wells are the public ISCWSA standard clearance set; the per-station separation factors (published, pedal and Mahalanobis) are provided as a diagnostics file; and the figure- and table-generation scripts are in the repository. The validation is encoded as automated tests.

## 10. Scope and limitations

This work is a **geometric $k\sigma$-boundary** method — it answers "is the offset within the combined $k\sigma$ ellipsoid", the same question the separation rule answers, computed exactly. It is not a probability-of-collision integral over the uncertainty volume; that is a richer, complementary treatment and is left to future work. The covariance model and confidence multiple are inherited from the ISCWSA error model and the chosen $k$; the method changes only how the uncertainty geometry is evaluated, not the uncertainty model itself.

Because both bodies are *analytic* (covariance ellipsoids along curves) and the metric is a quadratic form, the proximity query has a closed-form kernel and needs no surface mesh. This holds for any proximity problem whose bodies admit a tractable distance function (ellipsoids, quadrics, convex primitives); arbitrary non-analytic geometry is out of scope.

## 11. Conclusions

- The ISCWSA separation rule is *provably conservative*: it uses the ellipsoid support function where the exact boundary is the Mahalanobis distance, and by the Kantorovich inequality the rule's separation factor never exceeds the exact one.
- The conservatism is real and quantifiable — up to $1.48\times$ excess standoff on the ISCWSA standard set — and it has a direct economic cost (forgone wells; additional surveying; delayed production).
- The exact combined-covariance, minimum-Mahalanobis method (searching both interpolated wells, with a project-ahead floor and a conservatively *circumscribed* surface) agrees with the rule on every standard-set verdict while reducing conservatism, and runs in milliseconds.
- The objection that accurate methods are "impractical … because of high conceptual or computational complexity" fails on every plank: the exact method is *conceptually simpler* than the rule (one Mahalanobis form, no closest-approach caveats), an *order of magnitude faster* ($35$ ms vs $300$ ms), and *general* across all geometries — validated against the published standard set and released open-source. The barrier to accurate anti-collision was never complexity; it was the absence of an auditable implementation, now removed.

## References

- Gaynor, T. M., Chen, D. C.-K., Stuart, D., Comeaux, B. *Tortuosity versus Micro-Tortuosity — Why Little Things Mean a Lot.* SPE/IADC Drilling Conference, 2001. doi:10.2118/67616-MS
- Mahalanobis, P. C. *On the generalised distance in statistics.* Proceedings of the National Institute of Sciences of India, 2(1), 49–55, 1936.
- Sawaryn, S. J., Wilson, H., Bang, J., Nyrnes, E., Sentance, A., Poedjono, B., Lowdon, R., Mitchell, I., Codling, J., Clark, P. J., Allen, W. T. *Well-Collision-Avoidance Separation Rule.* SPE Drilling & Completion, 34, 01–15, 2019. doi:10.2118/187073-PA
- Williamson, H. S., et al. *(SPE-116155-PA)* — wellbore-collision risk assessment. doi:10.2118/116155-PA
- *(SPE-121040-MS)* — tool error model and anti-collision margins. doi:10.2118/121040-MS
- *(SPE-184644-PA)* — analytic collision-probability methods and their practicality. doi:10.2118/184644-PA
- welleng (open-source well-engineering library). <https://github.com/jonnymaserati/welleng>
