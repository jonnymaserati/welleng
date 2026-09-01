---
title: "An Exact Continuous Interior Ellipse of Uncertainty for Minimum-Curvature Wellbore Surveys"
author: |
  Jonathan Corcutt\
  Corcutt Beheer B.V., Wassenaar, Netherlands\
  ORCID 0009-0008-1953-7760
date: "2026"
papersize: a4
geometry: margin=2.2cm
fontsize: 10pt
header-includes: |
  \usepackage{amsmath}
  \usepackage{amssymb}
  \usepackage{caption}
  \captionsetup{labelformat=empty}
  \emergencystretch=3em
---

**Version 1.0 — 2026-09-01** · DOI: [10.5281/zenodo.22232546](https://doi.org/10.5281/zenodo.22232546) · companion to *Combining and Forward-Carrying Overlapping Wellbore Surveys* (concept DOI [10.5281/zenodo.22080773](https://doi.org/10.5281/zenodo.22080773)).

## Abstract

The ISCWSA/OWSG error model propagates a survey tool's error sources into a
position covariance **at the survey stations**. Between stations — where
anti-collision scrutiny is often highest — practice either linearly interpolates
the assembled covariance or treats the sub-station geometry as an added
statistical uncertainty. Linear interpolation is not the covariance of the
resolved arc: blending two station covariances inflates and re-orients the
ellipse near doglegs. The error is survey-interval dependent — negligible at the
ISCWSA-assumed ~30 m spacing, growing on coarse or high-dogleg surveys — and
shifts the separation factor either way, depending on the approach geometry. We present an **exact, deterministic** continuous
interior covariance for the resolved minimum-curvature trajectory: each error
source is propagated onto the minimum-curvature arc at an arbitrary measured
depth through the **partial-leg propagation Jacobian**, giving the covariance as
a closed-form function of arc position that is exact at the survey stations and
continuous between them. The construction is validated against an
interpolated-position Monte-Carlo reference to about 0.2%. It removes the
interpolation error at any interior point, and — because it is continuous in
measured depth — it makes a separation-factor constraint first-order smooth
along the trajectory, which benefits gradient-based well planning. An open, reproducible implementation
is provided in welleng.

## 1. Introduction

The ISCWSA/OWSG standard (Williamson, 2000; ISCWSA Rev 5) defines the position
covariance at the depths a survey is *taken*. A wellbore, however, is a
continuous object, and many downstream calculations need the uncertainty at an
arbitrary measured depth: anti-collision separation factors are evaluated at the
point of closest approach between two wells, which almost never lands on a survey
station of either; automated trajectory planners evaluate clearance continuously
along a candidate path.

Three treatments of the between-station covariance are in current use, and none
is the exact covariance of the resolved trajectory:

1. **Linear interpolation** of the assembled station covariances. Simple, but it
   is not the covariance of the resolved arc: blending two station covariances
   inflates and re-orients the ellipse near doglegs (a mixture/swelling effect).
   The resulting error is survey-interval dependent — negligible at the
   ISCWSA-assumed ~30 m spacing, growing on coarse or high-dogleg surveys — and
   shifts the separation factor either way, depending on the approach geometry.
2. **Eigen-decomposition interpolation** (Brooks, 2010, SPE-116155): spherical-
   linear interpolation of the eigenvector rotation with interpolated
   eigenvalues, reassembled positive-definite. This is smoother and
   positive-definite by construction, but it interpolates the *result* between
   two station covariances — it never propagates the error sources to the interior
   point — and so cannot be exact.
3. **A statistical interpolation-uncertainty term.** A joint-covariance treatment
   can add a Monte-Carlo "survey-frequency" covariance (the Stockhausen effect —
   unresolved micro-doglegs between stations) to the station-level propagation.
   This models the *unresolved* sub-station geometry as extra uncertainty; it is
   not the covariance of the *resolved* minimum-curvature arc.

This paper gives the missing object: the exact deterministic covariance of the
resolved minimum-curvature trajectory at any interior measured depth. It is not a
new error model — it uses the standard ISCWSA/OWSG station-defined sources — and
it is not a new correlation framework; it is the exact interior evaluation that
the station-defined model already implies once the trajectory between stations is
taken to be the minimum-curvature arc that the survey itself defines.

**Why it matters:** the interior covariance is the input to two operations of
direct engineering value. The first is survey combination and fusion, where the
overlap uncertainty is evaluated and reduced continuously along measured depth
(the companion paper quantifies that benefit on public field data). The second,
and the sharper case, is **anti-collision**: the exact Mahalanobis separation
factor and the probability of collision ($P_\text{col}$) between two wells
(Corcutt, 2026) are evaluated at the point of closest approach, which almost
never coincides with a survey station of either well. There, the interior
covariance *is* the collision-risk input — the number the go/no-go decision
rests on. Interpolating it rather than propagating it introduces an error that
depends on both survey interval and geometry: at the ISCWSA-assumed ~30 m spacing
it is small at the operative point (Section 3), but it grows on coarse, legacy or
gappy surveys and near strong doglegs, and its sign is set by the approach
geometry — over-conservative on a build approach, under-conservative on a
near-direct crossing. That a between-station treatment can under-estimate
collision risk is independently documented, by a different mechanism: Diao et al.
(2025) show that evaluating the uncertainty only at designated points, neglecting
the envelope between them, "may underestimate the risk of collision". An exact
interior covariance removes the interpolation error at whatever depth the closest
approach falls, in either regime.

## 2. The exact interior covariance

Let a leg run between survey stations $i$ and $i{+}1$, and let $q$ be an interior
point at arc-fraction $f\in[0,1]$ with measured depth $md_q$ and partial length
$L_q = md_q - md_i$. Its inclination and azimuth $(\theta_q,\phi_q)$ are the
minimum-curvature interpolation of the two stations' angles; the unit tangent at
any station $k$ (and at $q$) is

\begin{equation}
\mathbf{u}_k = \big(\sin\theta_k\cos\phi_k,\ \sin\theta_k\sin\phi_k,\ \cos\theta_k\big)^{\mathsf T}.
\tag{1}
\end{equation}

**Partial-leg Jacobian:** each error source is defined in the tool's
depth–inclination–azimuth (DIA) frame. The sensitivity of the partial leg
$[i,q]$'s NEV displacement to a DIA error is the minimum-curvature Jacobian
evaluated to $L_q$ rather than to the full leg length,

\begin{equation}
\mathbf{J}_{i\to q} =
\big[\ \mathbf{c}_D\quad \mathbf{c}_\theta\quad \mathbf{c}_\phi\ \big],\qquad
\mathbf{c}_D=\tfrac{1}{2}\big(\mathbf{u}_i+\mathbf{u}_q\big),\quad
\mathbf{c}_\theta=\tfrac{L_q}{2}\,\mathbf{t}^{\theta}_q,\quad
\mathbf{c}_\phi=\tfrac{L_q}{2}\,\mathbf{t}^{\phi}_q,
\tag{2}
\end{equation}

the three columns being the depth, inclination and azimuth sensitivities, where
the interior inclination and azimuth direction vectors are

\begin{align}
\mathbf{t}^{\theta}_q &= (\cos\theta_q\cos\phi_q,\ \cos\theta_q\sin\phi_q,\ -\sin\theta_q)^{\mathsf T}, \tag{3a}\\
\mathbf{t}^{\phi}_q &= (-\sin\theta_q\sin\phi_q,\ \sin\theta_q\cos\phi_q,\ 0)^{\mathsf T}. \tag{3b}
\end{align}

A second Jacobian
$\mathbf{J}'_{i\to q}$ is the out-leg coupling of station $i$ into $[i,q]$;
carrying it is what makes the construction exact at *both* ends of the leg (a
form keeping only $\mathbf{J}_{i\to q}$ is exact at $f\to1$ but biases $f\to0$ by
one leg).

**Interior propagation:** write $\mathbf{e}^{\mathrm{DIA}}_k$ for a source's
DIA-space error at station $k$, and let $\mathbf{p}_i=\sum_{j\le i}\mathbf{e}^{\mathrm{NEV}}_j$
be that source's NEV running sum frozen at station $i$. For a **systematic**
(correlated) source the interior NEV error is the arc-fraction blend of the two
bounding stations plus the coupling and the frozen prefix,

\begin{equation}
\mathbf{s}_q(f) = (1{-}f)\,\mathbf{J}_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_i
+ f\,\mathbf{J}_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_{i+1}
+ \mathbf{J}'_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_i
+ \mathbf{p}_i,
\tag{4}
\end{equation}

contributing the outer product $\mathbf{s}_q\mathbf{s}_q^{\mathsf T}$. A
**random** (independent) source instead contributes two independent outer
products — the interior point splits its own weight $(1{-}f)/f$ between the two
bounding stations —

\begin{equation}
\mathbf{g}_i = \mathbf{e}^{\mathrm{NEV*}}_i + \mathbf{J}'_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_i + (1{-}f)\,\mathbf{J}_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_i,
\qquad
\mathbf{g}_j = f\,\mathbf{J}_{i\to q}\,\mathbf{e}^{\mathrm{DIA}}_{i+1},
\tag{5}
\end{equation}

giving $\mathbf{g}_i\mathbf{g}_i^{\mathsf T}+\mathbf{g}_j\mathbf{g}_j^{\mathsf T}$
plus the frozen random prefix $\mathbf{R}_i=\sum_{j\le i}\mathbf{e}^{\mathrm{NEV}}_j\mathbf{e}^{\mathrm{NEV}\mathsf T}_j$.
Summing over all sources gives the interior covariance in closed form,

\begin{equation}
\mathbf{C}(f) = \sum_{\text{systematic}} \mathbf{s}_q\mathbf{s}_q^{\mathsf T}
\;+\; \sum_{\text{random}} \big(\mathbf{g}_i\mathbf{g}_i^{\mathsf T}+\mathbf{g}_j\mathbf{g}_j^{\mathsf T}+\mathbf{R}_i\big).
\tag{6}
\end{equation}

Two properties hold by construction and are the correctness anchors:

\begin{equation}
\mathbf{C}(0)=\boldsymbol{\Sigma}_i,\qquad \mathbf{C}(1)=\boldsymbol{\Sigma}_{i+1}
\quad\text{(exact at the stations),}
\tag{7}
\end{equation}

reproducing the conventional per-station ISCWSA covariance to machine precision,
while $\mathbf{C}(f)$ is continuous in $f$ along the leg — so the covariance, and
any quantity derived from it such as a separation factor, is continuous in
measured depth. Unlike interpolation of the assembled covariance, $\mathbf{C}(f)$
propagates the error *sources* to the interior point; unlike a statistical
interpolation-uncertainty term (Section 4), it is the exact covariance of the
*resolved* arc.

![**Figure 1.** The interior covariance $\mathbf{C}(f)$ is a *continuous*
ellipse-of-uncertainty surface along the leg — exact at the survey stations
($\mathbf{C}(0)=\boldsymbol{\Sigma}_i$, $\mathbf{C}(1)=\boldsymbol{\Sigma}_{i+1}$,
equation 7) and evaluable at any interior point $q$ at arc-fraction $f$. The
faint ellipses trace the surface between the stations; a query lands at an
arbitrary $q$ — the point of closest approach in anti-collision, or a swept
clearance depth in planning — which almost never coincides with a survey
station.](../figures/interior-eou-surface.png)

**A worked construction:** take one leg from $md_i=1000$ m ($\theta_i=30^\circ$,
$\phi_i=45^\circ$) to $md_{i+1}=1030$ m ($33^\circ$, $48^\circ$), and the leg
midpoint $f=0.5$. The dogleg is $3.384^\circ$; the minimum-curvature interior
angles from equation (1)'s tangent interpolation are $\theta_q=31.49^\circ$,
$\phi_q=46.56^\circ$, at partial length $L_q=15$ m. The bounding and interior
tangents (1) are $\mathbf{u}_i=(0.3536, 0.3536, 0.8660)$ and
$\mathbf{u}_q=(0.3592, 0.3793, 0.8527)$, so the partial-leg Jacobian (2) is

$$
\mathbf{J}_{i\to q}=\begin{bmatrix}
0.356 & 4.40 & -2.85\\
0.366 & 4.64 & 2.69\\
0.859 & -3.92 & 0
\end{bmatrix},
$$

its first column the depth sensitivity $\tfrac12(\mathbf{u}_i+\mathbf{u}_q)$ (a
half-tangent, dimensionless), the second and third the inc/azi sensitivities
scaled by $L_q/2 = 7.5$ m. A source acts through this matrix on its DIA error:
an azimuth-reference systematic of $e_A$ radians, for instance, contributes
$e_A\,(-2.85, 2.69, 0)^{\mathsf T}$ metres of horizontal displacement at $q$
through (4) (plus its frozen prefix from the stations above). Evaluating (4)–(6)
over all sources gives $\mathbf{C}(0.5)$; setting $f=0$ or $f=1$ collapses the
blend to the bounding station and recovers $\boldsymbol{\Sigma}_i$ or
$\boldsymbol{\Sigma}_{i+1}$ exactly, as in (7).

## 3. Validation

**Monte-Carlo oracle:** the independent ground truth for an interior point's
position uncertainty is an interpolated-position Monte Carlo: perturb every
station's error sources by their modelled statistics, reconstruct the perturbed
minimum-curvature trajectory, read the interior point's position from the
perturbed arc, and accumulate its covariance. This makes no analytical
assumption about the interior propagation and converges to the true covariance as
the sample size grows.

On a deviated leg (inclination 45→60°, interior point $f=0.5$) the exact
covariance and the Monte-Carlo estimate coincide (Figure 2a): at $N=10^6$ the
relative difference is 0.09%.

It should be said plainly that the analytical form is a closed-form evaluation at
constant cost per query — a single sub-millisecond call, independent of any sample
count — so evaluating it at every point of closest approach, or continuously along
a planning candidate, is negligible against the station covariance already in
hand. It *replaces* the Monte Carlo entirely in use; one never runs a Monte Carlo
to obtain an interior covariance. The large sample
counts here serve one purpose only: to *prove* the analytical form is exact. The
decisive test of exactness is not any single sample size but the **behaviour as
$N$ grows**. An exact analytical form differs from the Monte-Carlo estimate only
by the latter's sampling noise, which falls as $1/\sqrt N$; a form carrying a
residual bias would instead plateau at that bias no matter how large $N$ becomes.
So we push the Monte Carlo to several million samples not because the method
needs it, but to drive the sampling noise below any bias that might be hiding.
Across $N=10^5$ to $4\times10^6$ the difference falls monotonically — 0.54%,
0.12%, 0.11%, 0.06% — tracking the $1/\sqrt N$ noise floor with no plateau
(Figure 2b). There is no bias floor to find: the analytical form is exact to the
Monte-Carlo noise at every sample size. For contrast, a first-order
boundary-anchored interior form does *not* converge to zero this way — it
over-states at low inclination (about 19% at 18°, where the $1/\sin(\text{inc})$
azimuth weights are ill-conditioned), a bias that persists as $N$ grows.

![**Figure 2.** Monte-Carlo validation of the exact interior covariance at a
$45{\to}60^\circ$ leg midpoint. (a) The analytical $1\sigma$/$2\sigma$ ellipses
(solid) sit on the Monte-Carlo cloud and its ellipses (dashed) in the horizontal
plane. (b) The exact-vs-Monte-Carlo relative difference falls as $1/\sqrt N$ with
no plateau — the signature of an exact form (a biased form would flatten). The
comparison isolates the smooth measurement terms (see the course-length note
below).](../figures/interior-mc-validation.png)

**Course-length (XCL) terms:** the Monte-Carlo comparison above isolates the
smooth measurement terms. The course-length terms (XCLA/XCLH; Codling, 2017,
SPE-187249) are included in the interior covariance, but by one of two routes.
Under the default representation they use a stated **partial-course-length
convention** — station-exact at $f=0,1$ — which is *not* independently
Monte-Carlo-validated at a fractional point, because a course-length error has no
clean measurement-space Monte-Carlo ground truth there. Under the own-only angle-error recast (Corcutt, 2026) of the
Codling (2017) course-length term, XCL is expressed as an effective
inclination/azimuth angle error and then propagates through equations (1)–(6)
exactly like any smooth source — at which point it *is* Monte-Carlo-validatable.
Either way the terms are carried; the distinction is only whether the interior
form is exact-and-validated or a stated convention.

**Separation-factor error:** replacing linear covariance interpolation with the
exact interior covariance removes the interior error at any depth; its *size* is
survey-interval dependent. On the ISCWSA standard collision-test wells, at their
~30 m spacing, the linearly-interpolated separation factor at the governing
closest approach differs from the exact by under 0.2% (e.g. well 10, SF 0.202
either way) — negligible at standard density. The error grows on coarse or
high-dogleg surveys — about 33% at a 30°/30 m dogleg on a coarse leg. Because
linear interpolation inflates and re-orients the ellipse (a mixture/swelling
effect, $\mathbf{C}_\text{lin}\succeq\mathbf{C}(f)$ for a systematic source by
convexity), the sign of the separation-factor error follows the approach
geometry — over-stated on a build approach, under-stated on a near-direct
crossing. The exact form is correct in every case.

**Smoothness:** because the covariance is continuous in measured depth, a
separation-factor constraint built on it is first-order smooth along the
trajectory. Minimum curvature is $C^1$ and curvature still jumps at stations, so
the smoothness is of the constraint *gradient*, not higher; but station-wise
interpolation makes the constraint non-smooth *at the stations*, and those kinks
degrade the convergence of the quasi-Newton and SQP solvers used in gradient-based
well planning. The exact continuous covariance removes them.

## 4. Relation to prior work

The exact interior covariance sits alongside, and is distinct from, the three
treatments of Section 1. It differs from **Brooks (2010)** in kind: Brooks
interpolates the two station covariances; this propagates the sources to the
interior point. It differs from a **statistical survey-frequency term** in what it
models: that term is the uncertainty of the *unresolved* micro-dogleg geometry,
whereas this is the exact covariance of the *resolved* minimum-curvature arc — the
two are complementary, not competing (a full uncertainty budget can carry both).
And it differs from a **first-order boundary-anchored** interior form in accuracy:
the same conceptual target, evaluated exactly rather than to first order, which is
what removes the low-inclination bias.

The nearest framework is the covariance EOU (Bulychenkov, 2026), whose joint
covariance connects stations, runs and wells and carries a distinct
between-station term, $\mathbf{C}_\text{sf}$, for the positioning uncertainty the
survey does not resolve. That term is complementary to, not the same as, the
object here: $\mathbf{C}_\text{sf}$ concerns the *unresolved* sub-station
geometry, whereas $\mathbf{C}(f)$ is the exact covariance of the *resolved*
minimum-curvature arc — a full uncertainty budget can carry both. (We thank K.
Bulychenkov for the distinction.)

**On novelty:** the mechanism is not new outside drilling. Propagating a
covariance to an intermediate point of a trajectory through the state-transition
(partial-leg) Jacobian, $\mathbf{P}=\boldsymbol{\Phi}\mathbf{P}\boldsymbol{\Phi}^{\mathsf T}$,
is the standard error-propagation of aided inertial navigation, Kalman smoothing
and geodetic least-squares network adjustment, where the covariance at any epoch
or derived point is routine. What is new is not the mathematics but its **exact
application to the ISCWSA/OWSG wellbore model at an interior point of the
minimum-curvature arc**, in an open, validated form — directional surveying has
used covariance interpolation or first-order/statistical interior forms in place
of the exact propagation the station-defined model already implies. As with the
companion combination and anti-collision work, the contribution is exactness,
openness and validation, not new machinery.

## 5. Open implementation

The exact interior covariance is implemented in welleng as
`cov_nev_at(md)` on `welleng.error.ErrorModel`: a single call returns the
$3\times3$ NEV covariance at any measured depth, exact at the stations and
continuous between. Its regression gates are station-exactness (equation 7, which needs no
oracle) and parity against the analytical exact reference, itself validated
against the Monte-Carlo simulation of Section 3. As with the survey-combination
work, the implementation is open and reproducible, so a claimed interior
covariance can be checked against an independent reference rather than trusted on
assertion.

Because the construction propagates each error *source* by its propagation mode
(equations 4–6) and never hard-codes a particular term, it is **not tied to a
specific tool or to the MWD standard**: any tool error model expressed in the
ISCWSA/OWSG station-source architecture — MWD, a gyroscopic tool, a combined or
sag-corrected model, or an operator's own instrument-performance model (IPM) —
yields its exact interior covariance the same way, with no per-tool derivation.
That generality is the practical point: the barrier to obtaining a continuous
uncertainty for, say, a gyro survey is removed, not merely lowered.

This generality extends to the **small-angle exceptions** that several ISCWSA
terms carry. Weight functions with a $1/\sin(\text{inc})$ dependence (the
cross-axial and misalignment terms) are singular toward vertical, and the
standard defines a substitution below a vertical-inclination limit — a
position-space singular weight the imported term declares alongside its ordinary
weight. The interior covariance must honour these at the *interior* inclination,
not only at the stations: it re-evaluates each term at the interior angle through
the same interpreter that applies the term's own declared exception, so the
substitution is read from the term definition rather than hard-coded per term.
A term that interpolated its covariance between stations instead — bypassing its
own exception — over-states the interior most exactly where these terms dominate,
at low inclination; honouring the exception is what keeps the interior faithful
there.

The substitution is stated in position space for a physical reason worth making
explicit, because it is also why interpolating the angle-weight through vertical
is not merely inaccurate but qualitatively wrong. A transverse-sensor error does
not map to a *signed* angle error at exactly vertical: the inclination it induces,
$\delta\text{inc}\approx|b_{xy}|/G$, is the magnitude of a two-component
transverse perturbation (Williamson, 2000), so it is one-sided — Rayleigh-folded
rather than signed-Gaussian — with variance $(2-\pi/2)(\sigma/G)^2\approx
0.43\,(\sigma/G)^2$. The linear angle-weight presumes a signed-Gaussian angle
error; at vertical that presumption fails not only because $1/\sin(\text{inc})$
diverges but because the underlying distribution is folded, and the horizontal
position it should produce is a bounded ring rather than a diverging lateral
error. The position-space substitution represents that bounded, folded spread
directly — which is why the interior form must read it rather than
inter/extrapolate the weight through vertical. The fold is confined to a
razor-thin neighbourhood of exactly vertical (the crossover is at
$\text{inc}\sim b/G$, a few hundredths of a degree), so this fixes the correct
*limiting form* at the singular station; it is not a correction spread over the
near-vertical interval.

## 6. Conclusion

The ISCWSA/OWSG model defines the position covariance at survey stations; the
covariance *between* stations has been supplied by interpolation or by a
statistical uncertainty term, neither of which is the exact covariance of the
resolved trajectory. Propagating the standard error sources onto the
minimum-curvature arc through the partial-leg Jacobian gives that exact
covariance in closed form — continuous in measured depth, exact at the stations,
and validated against a Monte-Carlo oracle. It removes the interpolation error at
any interior point — negligible at standard survey density but material on coarse
or high-dogleg surveys, and of a sign set by the approach geometry — and makes
anti-collision constraints smooth for automated planning, and it is provided as
an open, validated reference implementation.

\newpage

## References

- Brooks, A. G. (2010). A New Look at Wellbore-Collision Probability. SPE Drilling &
  Completion, 25(2), 223–232 (presented 2008). DOI:
  [10.2118/116155-PA](https://doi.org/10.2118/116155-PA).
  [eigen-decomposition covariance interpolation, §1.]
- Bulychenkov, K. (2026). Covariance EOU: A Joint-Covariance Framework for Wellbore
  Positioning and Multisensor Data Fusion. Preprint. Concept DOI
  [10.5281/zenodo.22148755](https://doi.org/10.5281/zenodo.22148755)
  (v1 [10.5281/zenodo.22148756](https://doi.org/10.5281/zenodo.22148756)).
- Codling, J. (2017). Extended course-length (XCL) survey-interval error term. DOI:
  [SPE/IADC-187249](https://doi.org/10.2118/187249-MS). [the course-length terms, adopted
  in the ISCWSA error-model supplement.]
- Diao, B. et al. (2025). Journal of Petroleum Exploration and Production Technology.
  DOI: [10.1007/s13202-025-02054-z](https://doi.org/10.1007/s13202-025-02054-z).
  [independent finding, by a different mechanism (error-ellipsoid envelope neglected between
  designated points), that existing methods "may underestimate the risk of collision", §1.]
- Corcutt, J. (2026). Combining and Forward-Carrying Overlapping Wellbore Surveys: An Open
  Implementation Demonstrated on Public Volve Data. Zenodo. DOI:
  [10.5281/zenodo.22080773](https://doi.org/10.5281/zenodo.22080773).
  [companion paper — the combination and forward-carry the interior covariance feeds.]
- Corcutt, J. (2026). Making the Exact Wellbore Anti-Collision Boundary Practical: an
  Efficient, Validated, Open Implementation, and the Cost of the Separation-Rule
  Approximation. Zenodo. DOI:
  [10.5281/zenodo.20976872](https://doi.org/10.5281/zenodo.20976872). [the exact Mahalanobis
  separation factor the interior covariance feeds at the point of closest approach.]
- Corcutt, J. (2026). The ISCWSA Extended-Course-Length Error as an Own-Only Angle Error:
  Reconciling the Position-Direct Form with the Standard Measurement-Error Architecture.
  Zenodo. DOI: [10.5281/zenodo.21901641](https://doi.org/10.5281/zenodo.21901641).
  [the own-only angle-error recast of XCL used here.]
- Williamson, H. S. (2000). Accuracy Prediction for Directional Measurement While
  Drilling. SPE Drilling & Completion, 15(4), 221–233. DOI:
  [10.2118/67616-PA](https://doi.org/10.2118/67616-PA).
- ISCWSA / OWSG error model, Rev 5 (and the tool-code error term definitions).
- Sawaryn, S. J. and Thorogood, J. L. (2005). A Compendium of Directional
  Calculations Based on the Minimum Curvature Method. SPE Drilling & Completion, 20(1),
  24–36. DOI: [10.2118/84246-PA](https://doi.org/10.2118/84246-PA).
