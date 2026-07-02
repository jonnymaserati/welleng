---
title: "An Open, Vectorized Closed-Form Solver for the 3D Curve-Hold-Curve Point-to-Target Problem"
author: |
  Jonathan Corcutt\
  Corcutt Beheer B.V., Wassenaar, Netherlands\
  ORCID 0009-0008-1953-7760
date: "2026"
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
linkcolor: black
urlcolor: blue
citecolor: blue
header-includes: |
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{caption}
  \captionsetup{font=small,labelfont=bf,textfont=it}
  \usepackage{float}
  \usepackage{placeins}
  \usepackage{needspace}
  \floatplacement{figure}{tbp}
  \floatplacement{table}{tbp}
---

**Preprint · DOI: [10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979)**

---

## Abstract

The curve-hold-curve point-to-target problem — connect a start station (position and direction) to a target station (position and direction) with two circular arcs joined by a straight tangent (a *circle-line-circle*, CLC), under a dogleg-severity limit — is the core construction in directional-drilling trajectory planning. The operational standard solves it iteratively; iterative schemes can fail to converge when the tangent section is short or vanishes, and offer no guarantee of finding every solution. An explicit closed-form solution **is not new**: Sawaryn (2021, SPE-204111-PA) characterised the general 3D case as a tenth-order polynomial in the tangent length whose real positive roots parameterise every solution, and equivalent constructions exist for the related 3D Dubins problem (Baez et al., 2024; Xu et al., 2025). The closed form has nonetheless remained largely unused in practice. **Our contribution is to make it usable and open:** (i) we report that the *printed* eliminated polynomial (Eq. 15 of SPE-204111-PA) does not reproduce the paper's own worked Example-2 roots under length normalisation, and we recover the correct coefficients by replicating the paper's surd-elimination; (ii) we give an open, vectorized implementation that returns *every* CLC solution per station pair and selects the minimum-measured-depth path by default, using one batched companion-matrix eigenvalue solve rather than a per-instance loop; and (iii) we integrate it into the open-source library welleng as the curve-hold-curve solver of its trajectory `Connector`, replacing the inherited iterative scheme; and (iv) building on the batched solver, we add a radius sweep — over the design radius, or independent $R_1,R_2$ — that maps the feasible-radius trade-off in one solve, recovers the maximum usable radius, and surfaces an easily-missed operational hazard: near the critical radius a *larger* arc radius can remove the drillable path entirely. The solver reproduces Example 2 of SPE-204111-PA to the printed precision, recovers constructed paths to machine precision in a continuous-integration round-trip, and runs at the sub-millisecond-per-path scale needed for whole-field planning. All code, tests and data are released for reproducibility.

**Notation.** $p_1,t_1$ and $p_4,t_4$ are the start and target positions and unit tangents (NEV); $R_1,R_2$ the first/second arc radii; $\beta$ the straight tangent (hold) length; $\alpha_1,\alpha_2$ the two arc doglegs (subtended angles). Invariants: $\Delta p = p_4-p_1$, $\mu = t_1\!\cdot t_4$, $\psi^2=\lVert\Delta p\rVert^2$, $\eta_1=\Delta p\!\cdot t_1$, $\eta_4=\Delta p\!\cdot t_4$, $\eta_{14}=\Delta p\!\cdot(t_1\times t_4)/\sqrt{1-\mu^2}$.

---

## 1. Introduction

A directional well plan is a chain of stations, each a position and a tangent direction, that a trajectory must interpolate under a maximum dogleg severity (DLS, the curvature limit a bottom-hole assembly can build). The fundamental sub-problem is the **point-to-target** connection: given a start pose $(p_1,t_1)$ and a target pose $(p_4,t_4)$, find the curve-hold-curve trajectory — a circular arc of radius $R_1$, a straight tangent (the *hold*), and a second arc of radius $R_2$ — that joins them; geometrically a **circle-line-circle (CLC)** path. The radii follow from the design DLS; the unknowns are the tangent length $\beta$ and the two arc doglegs $\alpha_1,\alpha_2$.

The minimum-curvature method (Taylor & Mason, 1972; Zaremba, 1973) is the accepted model for the arcs, and most planning packages solve the CLC connection iteratively (Sawaryn & Thorogood, 2005, SPE-84246-PA). Iteration is robust in the common case but, as Sawaryn (2021) notes, "convergence issues have been reported in cases where the intermediate tangent section is either small or vanishes," and the conditions guaranteeing convergence or the existence of a solution had not been published. The iterative scheme inherited by most drilling planners (Sawaryn & Thorogood, 2005) returns a single (principal) root, with no assurance that other — possibly shorter — valid trajectories have not been missed. (The robotics literature on the equivalent 3D curvature-constrained path does enumerate the several CSC solution types and select the optimum — e.g. Hota & Ghose, 2010, 2014 — but those methods have not displaced the single-root iterative scheme in drilling practice.) These limitations were anticipated by the originators of the drilling construction: Sawaryn & Thorogood (2005, SPE-84246-PA) cautioned that with an iterative scheme "even if the scheme converges, there is no guarantee that it converges to the correct solution," that "there are cases for which no solution exists and extra code is needed to trap this condition," and that "explicit expressions are more predictable and usually confer advantages in speed and maintainability of the computer code." The present work answers each point directly: an all-roots explicit solver, a connector that raises on the no-solution case rather than masking it, and a vectorized batch path.

An explicit, all-roots solution removes both limitations, and one exists. Sawaryn (2021, SPE-204111-PA) showed that the general 3D point-to-target trajectory lies on a tenth-order self-intersecting surface and derived an eliminated polynomial of degree ten in $\beta$ whose real positive roots parameterise *every* CLC solution, valid for asymmetric radii and reducing through a hierarchy of degenerate forms (biarc, planar, landing). Independently, the equivalent 3D Dubins (CSC) problem was solved analytically via an inverse-kinematics (RRPRR-manipulator) formulation (Baez, Navkar & Becker, 2024) and reparametrised to a two-variable numerical search (Xu, Baryshnikov & Sung, 2025). Despite this, the iterative method remains the operational standard — even Sawaryn, having derived the polynomial, recommended falling back to iteration to obtain the principal root in practice.

This paper closes the gap between the existence of a closed form and its routine use, with four contributions:

1. **A correction to the printed eliminated polynomial.** As printed, Eq. 15 of SPE-204111-PA is not scale-covariant: it does not reproduce the paper's own worked Example-2 roots when the inputs are length-normalised. We recover the correct degree-ten coefficients by replicating the paper's surd-elimination of the half-angle tangents (Appendix B of SPE-204111-PA), and verify the result against Example 2 (§5).
2. **An open, vectorized solver** that returns every CLC solution per station pair and selects the minimum-measured-depth path by default. The roots of the (corrected) polynomial are obtained for a whole batch of station pairs in a single companion-matrix eigenvalue solve, so a field-scale sweep is one array operation rather than a Python loop over instances (§4).
3. **Integration into welleng**, an open-source directional-drilling library, as the curve-hold-curve solver of its trajectory `Connector` — replacing the inherited iterative scheme (§7).
4. **A batched radius sweep** (§8). Because the solver is vectorized, sweeping the design radius — or independent $R_1$ and $R_2$ — is one further batched solve. It maps the feasible-radius trade-off, recovers the maximum usable radius, and exposes an easily-overlooked hazard: near the critical radius a *larger* radius can remove the drillable path. An engineering tool, not new mathematics.

Everything is released open-source with its test suite and reference data.

## 2. The curve-hold-curve construction

A CLC trajectory leaves $p_1$ along $t_1$, turns through the first arc (radius $R_1$, dogleg $\alpha_1$) to a tangent point, holds straight for a length $\beta$, then turns through the second arc (radius $R_2$, dogleg $\alpha_2$) to arrive at $p_4$ along $t_4$. With $T_1=\tan(\alpha_1/2)$, $T_2=\tan(\alpha_2/2)$ and $t_2$ the (unit) hold direction, the closure condition is

\begin{equation}\Delta p \;=\; R_1 T_1\,(t_1+t_2)\;+\;\beta\,t_2\;+\;R_2 T_2\,(t_2+t_4). \end{equation}

The five dot-product invariants $\psi^2,\eta_1,\eta_4,\eta_{14},\mu$ (Notation) are rotation- and translation-invariant, so the problem is frame-free: no canonical-frame transform is required, and the solution depends only on these scalars and the radii.

## 3. The closed-form solution

**Forward constraints.** Sawaryn's forward equations express the invariants as functions of the path parameters (his Eqs. 11–13):

\begin{equation}\eta_1 = R_1\sin\alpha_1 + \beta\cos\alpha_1 + R_2 T_2(\mu+\cos\alpha_1), \end{equation}
\begin{equation}\eta_4 = R_1 T_1(\mu+\cos\alpha_2) + \beta\cos\alpha_2 + R_2\sin\alpha_2, \end{equation}
\begin{equation}\eta_{14} = \big(R_1 T_1 + \beta + R_2 T_2\big)\sqrt{\tfrac{1-\mu^2-\cos^2\alpha_1-\cos^2\alpha_2+2\mu\cos\alpha_1\cos\alpha_2}{1-\mu^2}}. \end{equation}

These reproduce Example 2 of SPE-204111-PA exactly and are the trustworthy core of the method.

**The eliminated polynomial, and the printed-form trap.** Eliminating $\alpha_1,\alpha_2$ from (2)–(4) gives a single polynomial of degree ten in $\beta$,

\begin{equation}P(\beta)=\sum_{k=0}^{10} c_k(\psi^2,\eta_1,\eta_4,\mu,R_1,R_2)\,\beta^{k}, \end{equation}

whose real positive roots are the tangent lengths of all CLC solutions. The coefficients are large (the paper notes the expansion runs to thousands of terms). The polynomial as **printed** in SPE-204111-PA does not reproduce the paper's worked Example-2 roots once the invariants and radii are length-normalised — i.e. it carries a transcription error. We therefore do not use the printed coefficients; we recover the correct $c_k$ by replicating the paper's own derivation — the half-angle quadratics and the bilinear constraint of Appendix B of SPE-204111-PA — and eliminating the surds symbolically. A regression test pins the distinction: the printed form fails to vanish at the Example-2 roots, the re-derived coefficients vanish to numerical zero (§5).

**Reconstruction.** For each root $\beta$, the arc doglegs follow from the half-angle quadratics, and the hold direction and waypoints are recovered in closed form and frame-free:

\begin{equation}t_2 = \frac{\Delta p - R_1 T_1\,t_1 - R_2 T_2\,t_4}{R_1 T_1 + \beta + R_2 T_2}, \quad p_2 = p_1 + R_1 T_1\,(t_1+t_2), \quad p_3 = p_2 + \beta\,t_2, \end{equation}

with $p_2,p_3$ the ends of the first arc and the hold. The reconstructed endpoint matches $p_4$ to machine precision.

**Degenerate cases.** When the tangents are parallel or antiparallel ($\lvert\mu\rvert=1$) the general form's $1/(1-\mu^2)$ is singular, and when the geometry is planar $\eta_{14}=0$; both are covered by the biquadratic 2D reduction of SPE-204111-PA, to which the solver dispatches automatically.

## 4. The vectorized solver

**All-roots, minimum-depth-by-default.** The solver returns every valid CLC solution for a station pair and, by default, the one of minimum measured depth $\mathrm{MD}=R_1\alpha_1+\beta+R_2\alpha_2$. The subtended angles are reported as the true dogleg in $[0,2\pi)$: the back-substitution yields $\alpha$ as $2\operatorname{atan2}(\cdot)\in(-2\pi,2\pi]$, and a co-terminal representation of the same physical path (identical $\tan(\alpha/2)$) would otherwise inflate $\lvert\alpha\rvert$ and mis-rank the measured depth; normalising $\alpha \bmod 2\pi$ recovers the true arc length and the correct ordering.

**Batched root-finding.** The cost is dominated by finding the roots of (5). Rather than loop over station pairs, we assemble the degree-ten companion matrix for each pair and obtain all roots for the whole batch with a single `numpy.linalg.eigvals` call over an $(N,10,10)$ array. This is the step that vectorizes: invariants by einsum, coefficients evaluated array-wise, roots by batched eigenvalues, then array-wise forward-verification to discard spurious roots from the elimination. The same code path runs unchanged on a GPU array library.

## 5. Validation

**Example 2 (absolute).** The solver reproduces the four published tangent lengths of SPE-204111-PA Example 2 ($\beta \approx 1072.6,\,1630.2,\,1789.95,\,2356.9$) to the printed precision, and the per-instance exact-rational resultant cross-check agrees (Figure 1).

![SPE-204111-PA Example 2, principal (minimum-MD) solution: arc 1 ($R_1=1250$ ft, dogleg $13.953^\circ$) — hold ($1072.6$ ft) — arc 2 ($R_2=1750$ ft, dogleg $13.109^\circ$), reproducing the published values (Table 1); measured depth $1777.4$ ft. Start pose green, target red; black dots mark the arc$\to$hold$\to$arc joints. The three further (large-arc) solutions are tabulated, not drawn.](../figures/example2_view.png){width=82%}

**The printed-form trap.** A test (`test_eq15_is_trapped`) asserts that the printed Eq. 15 does *not* vanish at the Example-2 roots, while the re-derived coefficients (`_eq15_coeffs`) do — pinning the correction.

**Round-trip recovery (continuous integration).** For random asymmetric constructions, the solver recovers the generating path: a known CLC is constructed, the solver is run on its endpoints, and a returned solution must align with the generator's arc/line waypoints, with the reconstructed endpoint reaching the target to $\sim10^{-13}$ and the selected minimum-MD path never exceeding the generator's. This runs in CI over many random cases; an offline sweep of $1000$ constructions recovers $1000/1000$ at machine precision (Figure 2).

![Round-trip recovery on a random asymmetric construction. The solver's minimum-measured-depth solution (green) overlays the constructed path (silver) to machine precision; the black dots mark the arc$\to$hold$\to$arc joints. The other CLC solutions (coloured) are returned simultaneously — the two faint surfaces are the horn tori the arcs inhabit.](../figures/clc-recovery.png){width=95%}

**Comparison with the worked examples.** welleng reproduces all four worked examples of SPE-204111-PA to the printed precision (Table 1). Example 2 returns all four tangent lengths (with the published subtended angles for the principal, minimum-MD root); Example 3's biarc ($\beta=0$) is recovered at the paper's biarc radius; and Example 4's *landing* — a target on a line, $p_4=p_0+k\,t_4$ — is solved for the along-line distance $k$ by the dedicated landing routine (Section 8), matching both published roots and the resulting biarc angles. (Example 1 is a feasibility check: the paper finds its second section infeasible at the motor's curvature limit.)

Table: Validation against the worked examples of Sawaryn (2021, SPE-204111-PA). Differences are at the printed precision of the source.

| Example | Radii (ft) | Quantity | Sawaryn (2021) | welleng |
|---|---|---|---|---|
| 2 (CLC) | $R_1{=}1250,\ R_2{=}1750$ | principal $\beta$ (ft) | 1072.6 | 1072.60 |
|  |  | $\alpha_1,\ \alpha_2$ (deg) | 13.953, 13.109 | 13.953, 13.109 |
|  |  | all four $\beta$ (ft) | 1072.6 / 1630.2 / 1789.95 / 2356.9 | 1072.60 / 1630.20 / 1789.95 / 2356.90 |
| 3 (biarc) | $R_1{=}R_2{=}2748.29$ | $\beta$ (ft) | 0 | 0.01 |
|  |  | $\alpha_1,\ \alpha_2$ (deg) | 19.552, 17.669 | 19.552, 17.669 |
| 4 (landing) | $R_1{=}R_2{=}469.94$ | $k$ roots (ft) | 37.27 / 567.52 | 37.27 / 567.52 |
|  |  | $\alpha_1,\ \alpha_2$ at $k{=}567.52$ (deg) | 30.774, 16.385 | 30.774, 16.385 |

## 6. Performance

The batched solver finds all roots for $N$ station pairs in one eigenvalue call. On a single AMD Ryzen 9 5950X (NumPy, double precision), the vectorized `solve_clc_batch` runs at **0.023 ms/path** at $N=2000$ — returning *every* CLC solution per pair — and the per-instance `solve_clc` at **1.3 ms/path** (Table 2). The exact-rational resultant cross-check (`solve_clc_resultant`, via python-flint) is far slower at $\sim\!10^2$ ms/path; it is a deterministic completeness check, not the production path. For external context, Baez et al. (2024) report $\sim\!0.025$ s to enumerate all solutions of a single configuration with a computer-algebra implementation on an Apple M2 Pro — a different metric (all-solutions-per-configuration rather than batched per-path throughput), so not a like-for-like comparison, but it places the analytic approach in the sub-second regime and welleng's batched path in the sub-millisecond.

Table: Solver throughput on an AMD Ryzen 9 5950X (single process, double precision).

| Method | Throughput | Returns |
|---|---|---|
| `solve_clc_batch` (vectorized, $N{=}2000$) | 0.023 ms/path | all solutions per pair |
| `solve_clc` (per-instance) | 1.3 ms/path | all solutions per pair |
| `solve_clc_resultant` (exact-rational) | $\sim\!10^2$ ms/path | completeness cross-check |

## 7. Integration in welleng

welleng's trajectory `Connector` resolves a connection between two nodes by classifying the inputs (hold, min-curve, curve-hold, curve-hold-curve, …) and solving each case. The curve-hold-curve case previously used the inherited iterative scheme; it now calls the closed-form solver directly, populating the connection's public state (the two arcs, the hold, measured depths, and the intermediate poses) from the minimum-MD solution at the design radii, using welleng's own minimum-curvature primitives so the rendered geometry is consistent to machine precision. The closed-form solver covers the two-arc case with both radii positive; the lower-order degenerate forms Sawaryn tabulates — build-and-hold ($R_2=0$) and the straight hold ($R_1=R_2=0$) — are the connector's other cases (`curve_hold`, `hold`), following Sawaryn & Thorogood (2005), while the planar ($\eta_{14}=0$) and parallel/antiparallel ($|\mu|=1$) degeneracies route to the biquadratic form of Section 3.

When no CLC exists at the design radii — the target requires tighter curvature than the design DLS — the connector by default raises a `ValueError` rather than silently tightening below the caller's stated DLS, leaving the radius decision to the caller (relax the limit, or sweep the radius). Opt-in `on_infeasible='max_radius'` instead returns the gentlest feasible curve — the analytic maximum-radius biarc (Section 8) — emitting a warning that the design DLS is exceeded. Either way the design constraint is never breached silently: a deliberate change from the iterative scheme, which quietly reduced the radius to reach an otherwise-infeasible target.

## 8. Extensions: feasibility and radius sweeps

The design-radius solver is complemented by two analytic extensions built on the same Eq. 15 machinery. The **maximum-radius** solve returns the gentlest feasible curve — the largest radius for which a valid CLC exists with both arc doglegs $\le\pi$ (a $>\pi$ arc being a non-physical loop) — obtained as the $\beta=0$ (biarc) boundary, i.e. the largest root of the constant coefficient $c_0$. It is the principled, closed-form replacement for the iterative auto-tighten: when the target is infeasible at the design DLS, it gives the best-fit gentlest curve to return in place of an error, and being $\beta=0$ it is also the minimum-tangent (biarc) trajectory. (For parallel tangents, $\lvert\mu\rvert=1$, where the general form is singular, the maximum radius is obtained from the planar form of Section 3.) The **landing** solve handles a target on a line, $p_4=p_0+k\,t_4$ (Appendix C of SPE-204111-PA): substituting the line-parametrised invariants ( — $\eta_1=\varepsilon_1+\mu k$, $\eta_4=\varepsilon_4+k$, $\eta_{14}=\varepsilon_{14}$, $\psi^2=\psi_0^2+2\varepsilon_4 k+k^2$) into $c_0$ yields a degree-ten polynomial in $k$ (reducing to Sawaryn's quartic in the symmetric-planar Example 4), solved for the along-line landing distance. (Sawaryn & Tulceanu (2009) address the complementary class of indirectly-defined targets — toolface-based steering, and landing onto a formation *plane* as a single-arc build-and-hold — distinct from the two-arc line landing here.) Both reproduce their worked examples exactly (Table 1). The maximum-radius result is wired into the connector as an opt-in fallback ($\texttt{on\_infeasible='max\_radius'}$) that returns the gentlest feasible curve in place of raising when the target is infeasible at the design DLS.

A third extension is a **radius sweep**. Because only the radii change between solves, the minimum-MD CLC over an entire range of design radii is obtained in a single batched eigenvalue solve — the fixed-radius solver of Section 4 broadcast over a range of radii ($\texttt{solve\_clc\_r\_sweep}$; the design radii $R_1,R_2$ are scaled together over a default $\pm25\%$ spread, with the design radius itself always included). This is an engineering convenience rather than new mathematics, but it makes the radius–measured-depth trade-off explicit (Figure 3): measured depth increases monotonically with radius — a gentler curve is a longer path — the feasible range is bounded above by the maximum-radius result, and at $R=0$ the arcs collapse to instantaneous turns so the path degenerates to the straight tangent $p_1\to p_4$ with measured depth equal to the chord length. It lets a caller trade dogleg severity against measured depth, or recover a feasible radius when the design DLS fails, without re-solving in a Python loop.

![Radius sweep for the SPE-204111-PA Example 2 geometry: measured depth of the minimum-MD CLC against design radius ($R_1=R_2=R$), computed in one batched solve. Measured depth rises monotonically with radius from the chord length at $R=0$ (the straight-tangent limit, arcs collapsed to instantaneous turns) to the maximum-radius bound, beyond which no CLC exists with both arc doglegs $\le\pi$ (shaded).](../figures/clc-r-sweep.png){width=75%}

The sweep also surfaces a counter-intuitive, easily-overlooked hazard: **the vertical-S radius trap** (Figure 4). Consider a planar build–hold–drop "S" whose two arcs each turn $90^\circ$ at the design radius — a deliberately contrived example chosen to expose the effect clearly, not a typical profile. Instinct says a *larger* radius gives a gentler, more drillable curve. Near the critical radius the sweep shows the opposite: as the radius is raised the hold shrinks, and $\sim8\%$ above the design radius it reaches zero — the two arcs meet in a biarc (the largest radius at which a drillable S exists). Beyond it no such S remains; the only CLC solutions carry a $>\pi$ arc (a physical loop) at $\sim2.4\times$ the measured depth. *Reducing* the radius instead lengthens the hold and lowers the arc doglegs — a geometrically simpler path. The feasibility boundary here is the parallel-tangent ($|\mu|=1$) degenerate case, solved by the planar biquadratic of Section 3; the batched sweep routes such stations to it automatically. A single design-radius solve gives no hint of the cliff — the sweep makes it explicit.

![The vertical-S radius trap for a planar $90^\circ/90^\circ$ build–hold–drop S. **Left:** measured depth of the drillable S (both arcs $\le\pi$) against arc radius; above the critical radius ($\sim1.08\times$ design, shaded) no drillable S exists and the shortest remaining path is a $>\pi$ loop at $\sim2.4\times$ the measured depth. **Right:** the hold length $\beta$ — raising the radius shrinks it to zero (the arcs meet in a biarc) at the critical radius, beyond which no drillable S exists; reducing the radius lengthens the hold (a simpler S). Both panels are one batched sweep.](../figures/clc-s-trap.png){width=95%}

The swept paths make the effect concrete (Figure 5): reducing the radius keeps a tidy S, whereas the raised-radius solution winds through a full loop. The practical point is a workflow one — **a trajectory should not be judged on its design-radius solution alone.** A single solve reports one path; the sweep, being a single batched call, lets an engineer *visually* check whether a tighter curve yields a more drillable — or even the only feasible — trajectory before committing to a radius. Assessing only the design dogleg can hide both the cliff above and the better path below it.

![The vertical-S sweep in 3D, viewed in the N–V plane. **Green:** the drillable S at $R = 0.70 / 0.85 / 1.00$ (design) — a tighter radius gives a simpler S. **Red:** at $R = 1.15$ (above the critical radius) the only remaining solution is a $>\pi$ loop, at more than twice the measured depth. Start pose green, target red.](../figures/clc-s-sweep-3d.png){width=70%}

The sweep generalises to *independent* radii ($\texttt{solve\_clc\_r\_grid}$): $R_1$ and $R_2$ vary on separate axes, giving the full reachability region in one batched solve (Figure 6). For the vertical S this is instructive. The coupled sweep runs along the diagonal $R_1=R_2$ and leaves the drillable region just above the design radius; but *asymmetric* radii remain drillable well beyond it — shortening one arc buys back the feasibility a uniform increase loses. The drillable set is bounded, yet far larger than the diagonal alone reveals; a single-radius or coupled view cannot show it. Its boundary is exactly the maximum-usable-radius locus: where it meets the diagonal is the symmetric maximum radius ($\approx1.08\times$ design here), the asymmetric generalisation of the maximum-radius solve above. The vertical S is a parallel-tangent ($|\mu|=1$) case — a degenerate form for which Sawaryn's general 3D expression is restricted ($\mu\neq\pm1$) and which he handles with the planar equation (Eq. 34 of SPE-204111-PA). welleng routes such stations to that planar form throughout — the batch solve, the sweep, and the maximum-radius calculation — so the drillable region and its boundary follow with no special handling by the caller.

![Independent $R_1\times R_2$ sweep of the vertical-S geometry: measured depth of the minimum-MD drillable S over the radius grid (grey = infeasible, no drillable S). The coupled sweep is the dashed diagonal $R_1=R_2$; it leaves the drillable region just above the design point (red), while asymmetric radii stay feasible far beyond. One batched solve.](../figures/clc-r-grid.png){width=62%}

\FloatBarrier

## 9. Conclusion

The closed-form CLC point-to-target solution is sound but, as printed, carries a transcription error in its eliminated polynomial and has lacked an open, vectorized implementation. We supply both: the corrected coefficients, and a batched all-roots solver that selects the minimum-measured-depth path, validated against the originating paper and integrated into an open-source planner. The result makes the explicit solution a practical default, with the all-roots guarantee the iterative method cannot offer.

## Data and code availability

The solver, its test suite, and the reference data are in welleng (`welleng/sawaryn_analytical.py`, `tests/test_sawaryn_analytical.py`); the welleng software concept DOI is [10.5281/zenodo.20968887](https://doi.org/10.5281/zenodo.20968887). This paper: DOI [10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979).

**Citation (required).** This work is released openly on the condition that it is cited. Any use of the corrected polynomial coefficients, the solver, the maximum-radius / landing / radius-sweep tooling, or the results and figures of this paper — whether in a publication, in software, or in a commercial product — must cite **this paper** (DOI [10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979)) **and** welleng (software concept DOI [10.5281/zenodo.20968887](https://doi.org/10.5281/zenodo.20968887)). The underlying closed form is due to Sawaryn (2021, SPE-204111-PA) and must be cited alongside it.

\Needspace*{22\baselineskip}

## References

- Sawaryn, S. J. (2021). A Generalized Solution to the Point-to-Target Problem Using the Minimum Curvature Method. *SPE Drilling & Completion*. DOI: [10.2118/204111-PA](https://doi.org/10.2118/204111-PA).
- Sawaryn, S. J., & Thorogood, J. L. (2005). A Compendium of Directional Calculations Based on the Minimum Curvature Method. *SPE Drilling & Completion* 20(1): 24–36. DOI: [10.2118/84246-PA](https://doi.org/10.2118/84246-PA).
- Sawaryn, S. J., & Tulceanu, M. A. (2009). A Compendium of Directional Calculations Based on the Minimum-Curvature Method—Part 2: Extension to Steering and Landing Applications. *SPE Drilling & Completion* 24(2): 311. DOI: [10.2118/110014-PA](https://doi.org/10.2118/110014-PA).
- Baez, V. M., Navkar, N., & Becker, A. T. (2024). An Analytic Solution to the 3D CSC Dubins Path Problem. arXiv:2405.08710 [cs.RO].
- Xu, L., Baryshnikov, Y., & Sung, C. (2025). Reparametrization of 3D CSC Dubins Paths Enabling 2D Search. arXiv:2503.11560 [cs.RO].
- Hota, S., & Ghose, D. (2010). Optimal Geometrical Path in 3D with Curvature Constraint. *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Taipei, 113–118. DOI: [10.1109/IROS.2010.5653663](https://doi.org/10.1109/IROS.2010.5653663).
- Hota, S., & Ghose, D. (2014). Optimal Trajectory Planning for Unmanned Aerial Vehicles in Three-Dimensional Space. *Journal of Aircraft* 51(2), 681–688. DOI: [10.2514/1.C032245](https://doi.org/10.2514/1.C032245).
- Taylor, H. L., & Mason, C. M. (1972). A Systematic Approach to Well Surveying Calculations. *SPE Journal* 12(6): 474. DOI: [10.2118/3362-PA](https://doi.org/10.2118/3362-PA).
- Zaremba, W. A. (1973). Directional Survey by the Circular Arc Method. *SPE Journal* 13(1): 5–11. DOI: [10.2118/3664-PA](https://doi.org/10.2118/3664-PA).
