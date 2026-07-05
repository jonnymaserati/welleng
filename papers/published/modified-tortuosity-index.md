---
title: "A Note on the Well Trajectory Tortuosity Index: Non-Independence of the Inclination–Azimuth Combination, and a Dimensionless Form"
author: |
  Jonathan Corcutt\
  Corcutt Beheer B.V., Wassenaar, Netherlands\
  ORCID 0009-0008-1953-7760
date: "2026"
papersize: a4
geometry: margin=2.2cm
fontsize: 10pt
header-includes: \usepackage{amssymb}
---

**Preprint — Version 1.0 (2026-06-26)**

---

## Abstract

The Tortuosity Index (TI) is a widely used geometric quality metric for well trajectories, adapted to drilling by Ashok, Zhou, D'Angelo and co-workers (Zhou et al., SPE-178869-MS, 2016; D'Angelo et al., SPE/IADC-194099-MS, 2019) from the retinal-vessel tortuosity measure of Grisan et al. (2008). The published method computes the index separately in the inclination and azimuth domains and combines the two as the root of the sum of squares, $TI_{3D}=\sqrt{TI_{inc}^{2}+TI_{azm}^{2}}$.

We show that this combination is not well-posed: the arc-length and chord-length terms entering the inclination and azimuth sums are the *full three-dimensional* quantities (the published equations define them in 3D, and only the curve-turn *detection* differs between the two domains), so the two "components" are not independent and any curve turn detected in both domains is counted in both sums. The root-sum-of-squares therefore over-states three-dimensional tortuosity rather than decomposing it.

The correction is to compute the index directly in three dimensions: sectionizing the trajectory by the continuity of the curve-turn *plane* (its normal vector $\mathbf{n}=\mathbf{v}_{1}\times\mathbf{v}_{2}$) measures 3D curvature once per turn, captures torsion, and removes the inclination/azimuth split that causes the double count. A second correction addresses units: the published index scales with the unit of length (a survey in meters returns $0.3048$ times the value of the same survey in feet), where dividing each turn's excess ratio by its arc length and normalizing by total curve length yields a dimensionless *Modified* Tortuosity Index (MTI). Both corrected forms are validated against a closed-form single-arc solution (exact agreement), a set of dimensional and ordering invariants, and regression anchors on the ISCWSA standard wells, all implemented as open-source tests in welleng. A naming collision with the literature is noted (the "MTI" of SPE/IADC-194099-MS is an unrelated *Mapped* Tortuosity Index), and a survey-frequency correction (the maximum-curvature method) is left to a companion paper.

**Notation.** $D$ measured depth, $\mathbf{r}=(N,E,V)$ position (north, east, vertical), $\mathbf{v}$ the unit trajectory (tangent) vector, $\mathbf{n}$ the curve-turn plane normal, $I$ inclination, $A$ azimuth, $L_{cs}$ arc (along-hole) length of a curve turn, $L_{xs}$ chord (straight-line) length of a curve turn, $L_{c}$ total curve length, $\kappa$ a scale constant.

---

## 1. Introduction

Tortuosity quantifies how much a path deviates from being straight or smooth. For a wellbore it is an intuitive, purely geometric measure of trajectory quality, and a natural Key Performance Indicator (KPI) for directional-drilling performance: a smoother well reduces torque and drag, casing-running and completion risk, and the friction "fudge factors" that dominate extended-reach planning.

The Tortuosity Index used in drilling originates in the medical-imaging literature — Grisan et al. (2008) graded the tortuosity of retinal blood vessels — and was adapted for well trajectories by Ashok and co-workers at the University of Texas at Austin RAPID program and presented to the International Association of Directional Drilling (IADD), then developed in a sequence of SPE papers (Zhou et al., 2016; D'Angelo et al., 2019). The 2019 paper (SPE/IADC-194099-MS) additionally separates *planned* from *unplanned* tortuosity, which is not our subject here; we are concerned only with the underlying *wellbore* Tortuosity Index it builds on.

This note does the following:

1. It identifies a mathematical inconsistency in the published inclination/azimuth root-sum-of-squares formulation (Section 2).
2. It sets out the corrected computation — a tortuosity index evaluated directly in three dimensions, sectionizing by curve-turn-plane continuity, which measures each turn once and captures torsion (Section 3).
3. It gives a dimensionless form of the *Modified* Tortuosity Index and notes a naming collision with the literature (Section 4).
4. It reads the along-hole index *profile* as total, remaining and local tortuosity, which communicates well quality where a single scalar cannot (Section 5).
5. It validates the corrected indices analytically and by invariant, with an open-source implementation and test suite (Section 6), and notes the survey-frequency limitation (Section 7).

The corrected 3D computation, the dimensionless form and the non-independence observation below were described by the present author in open-source form (the welleng library) and a 2022 blog series; this note consolidates and validates that material.

## 2. The published Tortuosity Index and its non-independence

The published method (Zhou et al., 2016, eqs 1–4; carried forward by D'Angelo et al., 2019) analyses *curve turns* along the well path. For the $i$-th curve turn it forms an arc-length and a chord-length (Zhou et al. 2016, their eqs 2–3),

\begin{align}
L_{cs,i} &= D_{i+1}-D_{i}, \\[2pt]
L_{xs,i} &= \sqrt{(V_{i+1}-V_{i})^{2}+(N_{i+1}-N_{i})^{2}+(E_{i+1}-E_{i})^{2}},
\end{align}

and scores the turn by the excess of arc over chord, $L_{cs,i}/L_{xs,i}-1$, which is zero for a straight segment and grows with curvature. The scores are summed over the curve turns, weighted by a frequency modifier $n/(n+1)$ and normalized by the total curve length $L_{c}$, giving an index in each of the inclination and azimuth domains:

\begin{align}
TI_{inc} &= \frac{n}{n+1}\,\frac{1}{L_{c}}\sum_{i=1}^{n}\left(\frac{L_{cs,i}}{L_{xs,i}}-1\right), \\[4pt]
TI_{azm} &= \frac{m}{m+1}\,\frac{1}{L_{c}}\sum_{j=1}^{m}\left(\frac{L_{cs,j}}{L_{xs,j}}-1\right).
\end{align}

The curve turns are detected *independently* in the two domains: inflection points are found from the sign change of the second derivative of inclination with respect to depth (yielding $n$ inclination curve turns), and separately from the second derivative of azimuth (yielding $m$ azimuth curve turns). The two indices are then combined into a single three-dimensional value as the root of the sum of squares (Zhou et al. 2016, their Eq. 4, reproduced here),

\begin{equation}
TI_{3D}=\sqrt{TI_{inc}^{2}+TI_{azm}^{2}}.
\end{equation}

Because the raw indices are small ($10^{-7}$ to $10^{-5}$ in magnitude), the published results are scaled by a factor of $10^{7}$ so they fall in a convenient range of roughly 0–100.

**The problem.** Equation (5) treats $TI_{inc}$ and $TI_{azm}$ as orthogonal components whose magnitudes may be combined by Pythagoras. This is only valid if the two are independent. They are not. The terms $L_{cs}$ and $L_{xs}$ in equations (1)–(2) — and hence in both (3) and (4) — are the *full three-dimensional* arc and chord lengths. Nothing in $L_{cs}/L_{xs}-1$ isolates an inclination component or an azimuth component; the *only* thing that differs between the inclination sum and the azimuth sum is the *set of curve turns* over which the (identical, 3D) summand is evaluated. A curve turn in a typical 3D build-and-turn produces an inflection in *both* inclination and azimuth, so its three-dimensional excess-ratio is added to $TI_{inc}$ *and* to $TI_{azm}$, and equation (5) returns up to $\sqrt{2}$ times the single underlying quantity.

![*The inclination/azimuth root-sum-of-squares double-counts a three-dimensional curve turn. A synthetic build-and-turn whose single curve bends in both inclination and azimuth is scored by welleng's corrected once-per-3D-section index (green) and by the published method. Because the arc- and chord-length terms (equations 1–2) are the full 3D quantities, the same curve turn is detected in both domains and its identical 3D excess-ratio enters $TI_{inc}$ and $TI_{azm}$; the root-sum-of-squares (equation 5) then returns $\sqrt{2}$ times the true value — a $41\%$ over-statement.*](../figures/ti-double-count.png){width=72%}

A valid orthogonal decomposition would require $TI_{inc}$ to be built from the trajectory *projected onto a vertical plane* (so that $L_{cs}$, $L_{xs}$ measure only the inclination-plane curvature) and $TI_{azm}$ from the *plan-view projection*. That projection is non-trivial — it requires re-deriving the projected measured depth on each principal plane to a controlled accuracy — and is not what the published equations compute. The root-sum-of-squares of two co-measured 3D quantities is not a 3D magnitude.

## 3. Computing the index directly in three dimensions

Rather than split a three-dimensional path into two coupled two-dimensional views and recombine them, we measure curvature directly in three dimensions.

**Curve turns as planar arcs.** Modern wells are planned and computed almost exclusively with the minimum-curvature method, in which the path between two survey stations is a circular arc lying on a plane in 3D space. The orientation of that plane is given by its unit normal, the cross product of the trajectory's unit tangent vectors at the two stations,

\begin{equation}
\mathbf{n}_{i,j}=\mathbf{v}_{i}\times\mathbf{v}_{j}.
\end{equation}

The normal is independent of the radius of curvature (the dogleg severity); it depends only on the *plane* in which the turn occurs, which is exactly the property by which the published method already defines a curve turn. A continuous curve turn between two points therefore traverses a single spherical surface whose plane normal is constant. For a straight (hold) section the two tangents are parallel and the cross product is undefined ($\mathbf{0}$ / NaN), which usefully marks holds as discrete sections with arc length equal to chord length and zero tortuosity.

**Sectionizing.** We label the path continuous between successive stations while the plane normal is unchanged,

\begin{equation}
\text{section continues while}\quad \mathbf{n}_{k}\approx\mathbf{n}_{k-1},
\end{equation}

tested with a tolerance on the normal-vector components (and, optionally, on the dogleg severity, for surveys where curvature changes within a common plane). The indices of the stations where continuity breaks define the section starts $s$. For every station $k$ we then measure the arc and chord from the start of *its own* section,

\begin{align}
L_{cs,k} &= D_{k}-D_{s(k)}, \\[2pt]
L_{xs,k} &= \lVert\,\mathbf{r}_{k}-\mathbf{r}_{s(k)}\,\rVert,
\end{align}

where $s(k)$ is the start station of the section containing $k$. Evaluating the excess ratio at every station, and accumulating completed sections, gives a tortuosity *profile* along the well rather than a single end-of-well number. Writing $n_{k}$ for the number of curve-turn / hold sections completed by station $k$, the three-dimensional Tortuosity Index is

\begin{equation}
TI_{k}=\frac{n_{k}}{n_{k}+1}\,\frac{\kappa}{L_{c,k}}\sum_{\text{sections}\le k}\left(\frac{L_{cs}}{L_{xs}}-1\right),
\qquad \kappa=10^{7},
\end{equation}

with $L_{c,k}=D_{k}-D_{0}$ the curve length to station $k$. This reduces to the published per-curve-turn sum at section boundaries but, because it sectionizes by the 3D plane normal, captures torsion (a change of turning plane registers as a new section) and never double-counts a turn.

Torsion of the well path has long been studied — computed pointwise from the Frenet frame and folded into wellbore curvature and energy indices. The point here is narrower: using the *continuity of the turning-plane normal* $\mathbf{n}=\mathbf{v}_i\times\mathbf{v}_j$ as the *sectionization criterion* for the tortuosity index registers a change of turning plane as a new section directly, without a separate torsion computation.

## 4. A dimensionless Modified Tortuosity Index

The index of equation (10) is *not dimensionless*: the summand $L_{cs}/L_{xs}-1$ is dimensionless, but the normalizing factor $1/L_{c}$ carries units of inverse length. The published index inherits this — its reference ranges are implicitly per-foot, and the $10^{7}$ scale is calibrated to feet. A survey expressed in meters yields a value $0.3048$ times that of the identical survey in feet, so absolute indices are not comparable across unit systems.

The Modified Tortuosity Index (MTI) restores dimensional consistency. Each section's excess ratio is divided by its arc length, and the total curve length is moved to the numerator,

\begin{equation}
MTI_{k}=\frac{n_{k}}{n_{k}+1}\,\kappa\,L_{c,k}\sum_{\text{sections}\le k}\left[\left(\frac{L_{cs}}{L_{xs}}-1\right)\frac{1}{L_{cs}}\right],
\qquad \kappa=1.
\end{equation}

Now $\left(L_{cs}/L_{xs}-1\right)/L_{cs}$ has units of inverse length, the sum has units of inverse length, and multiplying by $L_{c}$ yields a pure number. The MTI of a survey in feet equals that of the same survey in meters (verified to machine precision in Section 6). The constant $\kappa$ is retained for continuity with the published form (where it is the easily-overlooked $10^{7}$); for the dimensionless index $\kappa=1$ is appropriate.

![*The same well in two unit systems. **Left:** the published TI evaluated with the survey in meters and in feet differs by exactly the length-unit ratio $0.3048$. **Right:** the MTI is identical in the two unit systems (ratio $1.000000$). Both computed from the welleng implementation on the synthetic build-and-turn.*](../figures/ti-unit-invariance.png){width=88%}

The unit-dependence is inherited from the source measure: the SPE adaptation carried it over unremarked. The dimensionless form above removes it. Note that simply dropping the $1/L_c$ factor would collapse the index toward the plain arc-over-chord ratio, discarding the turn-*frequency* sensitivity that distinguishes many small turns from a few large ones; the MTI form removes the units while retaining that sensitivity.

> **A naming caution.** In SPE/IADC-194099-MS the abbreviation "MTI" denotes a *Mapped* Tortuosity Index — the tortuosity of the planned curve turns mapped onto the as-drilled path, used to form the Unplanned Tortuosity Index $UPTI=TI-MTI$. That is a different quantity from the *Modified* Tortuosity Index defined here. We retain "MTI" for the modified index (consistent with the prior welleng releases and posts that introduced it) but flag the collision explicitly to avoid confusion.

**As a quality KPI.** Because the MTI is unit-invariant and consistently defined, it supports a dimensionless wellbore-quality index relative to a reference (planned) trajectory,

\begin{equation}
QI_{tortuosity}=\frac{MTI_{\text{drilled}}}{MTI_{\text{planned}}}-1,
\end{equation}

which is zero for a well drilled exactly to plan and grows with unplanned tortuosity. Used this way, an operator and a directional-drilling contractor can agree an MTI target up front and measure delivery against it.

**Recovering the reference trajectory.** The quality index needs a planned MTI, but the original plan is frequently unavailable — for a legacy well, a sidetrack, or any well whose design was never archived. The same sectionization recovers a usable reference directly from the as-drilled survey. Running the continuity test at a *coarse* tolerance (a relaxed normal-vector tolerance together with a dogleg-continuity tolerance of a few deg/30m) distils the trajectory to a handful of curve-turn and hold control points — the build, hold and turn sections a planner would have specified — which are then re-connected with minimum-curvature arcs to yield an inferred design trajectory.

This exposes a striking property: an as-drilled survey of many stations is *compressible* to a handful of control points that describe it almost exactly. On the Volve field data set (Equinor, released for open use) — whose Landmark EDM export carries, for each wellbore, both the as-drilled definitive survey and the independently archived well plan — the as-drilled survey for well NO 15/9-F-2 has 134 stations. The coarse sectionization distils it to just **eleven** $(D, I, A)$ control points whose minimum-curvature reconstruction reaches the target exactly, staying within 1 m of the as-drilled path throughout (Figure 3): a better-than-12:1 reduction with no material loss of geometry, because the well *is* a handful of build, hold and turn sections and the intervening stations are redundant. The distilled points are, in effect, the design intent — what a planner would have specified — recovered from the as-drilled survey without the plan. The recovered MTI ($0.338$) equals the as-drilled ($0.338$): the control points sit at the section boundaries, so each recovered arc reproduces the as-drilled arc and no curvature is lost — the compression is faithful, not a smoothing. Their concentration in the shallow hole (Figure 3) is not arbitrary but tracks where the design detail lies — the kick-off, a doubling of build rate (from $0.50$ to $1.00\,^\circ/30\,\text{m}$ near $311$ m), and an early azimuth turn are each a genuine section boundary — while the long tangent below needs none. (Naive decimation, cutting into curved sections, *would* lose curvature; this is the survey-frequency effect of Section 7.) The $QI_{tortuosity}$ of equation (12) needs an *independent* baseline: against the archived plan (MTI $0.325$), which the recovery never sees, the as-drilled is $4\%$ more tortuous. Where the plan is lost, the recovery still supplies its geometry — the control points, and the target — but not, being drawn from the same survey, an independent tortuosity baseline for that well.

![*Recovering the design of Volve well NO 15/9-F-2 from its as-drilled survey alone. The as-drilled definitive survey (blue, 134 stations) is distilled by the plane-normal sectionization to eleven control points (green dots) and reconnected with minimum-curvature arcs (green dashed); the field's independently archived plan (silver) is shown for comparison but is not used by the recovery. The reconstruction reaches the target exactly, staying within 1 m of the as-drilled path throughout, and recovers an MTI ($0.338$) within 4% of the archived plan ($0.325$). The plan's divergence from the as-drilled path at the toe (right panel) is a genuine plan-versus-delivery difference — exactly what $QI_{tortuosity}$ quantifies.*](../figures/ti-recovery.png){width=95%}

Two properties make this example a KPI in practice. First, the *consistency*: the as-drilled and its ten-point re-derivation return the same minimum-curvature MTI to three figures, and (Section 6) that value is stable against the section threshold and the station count. Two parties working from the same survey will compute the same number — the reproducibility a contract metric requires. Second, it *discriminates*: this well came in $4\%$ more tortuous than its plan. Read as a delivery KPI agreed up front, that difference is exactly the signal an operator and directional driller would surface in a post-well review — a measured, unambiguous prompt to ask *why* the delivered well was rougher than designed. (This is the minimum-curvature reading; the max-curvature treatment of Section 7 would raise the as-drilled value further, and is left to the companion paper.)

## 5. Interpreting the index: total, remaining and local tortuosity

A single end-of-well number is a poor summary of a property that varies along the hole. Two wells with the same total tortuosity can drill very differently: one may accumulate it gently over its length, while another concentrates it in a short, severe interval that dominates torque, drag and running risk. Conversely a well with high tortuosity confined to an early feature — a sidetrack kick-off, say — may be straightforward to drill below it. A scalar index cannot distinguish these cases.

Because the three-dimensional index of equation (10) is evaluated at every station it is naturally a *profile*, and three readings of it carry distinct engineering meaning:

- **Total tortuosity** — the value at total depth (or any chosen station): the whole-well summary used as a KPI.
- **Remaining tortuosity** — the increment still to be accumulated from the current bit position to a target: what a smoother continuation would have to overcome, useful while drilling ahead.
- **Local tortuosity** — the along-hole gradient of the index over a short interval: this is what flags a single bad section that a total-tortuosity number would hide.

Plotted against depth, the along-hole MTI is monotonic and steps up where the well turns, so its local gradient pinpoints the sections that contribute most (Figure 4).

![*The three readings on the Volve NO 15/9-F-2 as-drilled survey, plotted against measured depth (increasing downward, as a driller would read it). **Left:** the cumulative MTI is flat through the vertical upper hole and accumulates through the lower build-and-turn to its total of $0.338$ at TD; the remaining tortuosity from a nominal bit position to the target is the bracketed increment ($0.310$). **Right:** the local gradient $\mathrm{d}(MTI)/\mathrm{d}(MD)$ is near zero above $\sim2100$ m and pinpoints the most tortuous section at $\sim3000$ m MD — a feature the single total-depth number would hide.*](../figures/ti-views.png){width=82%}

As a purely geometric quality metric it is well suited to scoring and comparing candidate trajectory *designs*: it was used this way in the AI-assisted well-trajectory optimization of Almarzooqi et al. (SPE-211041-MS, 2022), who include tortuosity among the metrics their optimizer can "take this into account and, when required, optimize on," and who acknowledge the present author "for his technical guidance and work on the modified tortuosity index." We make no claim that the index predicts torque and drag: those carry a large weight-driven component that a geometric measure does not contain, and relating tortuosity to friction quantitatively requires a torque-and-drag model, which is out of scope here.

## 6. Validation

The published wells are proprietary and the source papers tabulate no raw survey data, so the index cannot be checked against published numbers directly. Validation is instead by (a) a closed-form solution, (b) dimensional and ordering invariants, and (c) regression anchors on the ISCWSA standard wells. All checks are implemented as open-source tests (`tests/test_tortuosity_index.py` in welleng).

**Closed form — a single arc.** For a single continuous circular arc the path is one section, $n=1$, and equation (10) collapses to

\begin{equation}
TI=\frac{1}{2}\,\frac{\kappa}{L_{c}}\left(\frac{L_{\text{arc}}}{L_{\text{chord}}}-1\right).
\end{equation}

For a $0^\circ\!\to\!60^\circ$ build over $600$ m (arc length $L_{c}=600$ m, $\kappa=10^7$, $L_c$ in feet), the implementation returns $TI=119.88178$, matching equation (13) to a relative tolerance of $10^{-9}$.

\newpage

**Invariants.** Table 1 lists the qualitative properties the indices must satisfy and the observed result.

| Property | Expected | Observed |
|---|---|---|
| Straight hole | $TI=MTI=0$ | exactly $0$ |
| MTI unit-invariance | feet $=$ meters | equal to $10^{-9}$ |
| TI unit-dependence | feet/meters $=0.3048$ | $0.30480$ |
| Max-curvature pre-processing | more tortuous | $MTI$ increases |
| Interpolate-then-max-curvature | more tortuous still | $MTI$ increases further |
| Single arc | closed form (13) | match to $10^{-9}$ |

Table: Tortuosity-index invariants and observed results.

**Regression anchors.** Table 2 records the current end-of-well values on the two ISCWSA standard wells; the test suite locks these to guard against silent change. They are re-baselined against the present implementation: the absolute values published in the original 2022 posts have drifted by roughly 20% as the underlying interpolation and maximum-curvature routines evolved, but the *sectionization* is unchanged and every invariant above holds, so the core method is intact. (Exact reproduction of the 2022 absolutes is a concern for the companion maximum-curvature paper, not for the index definition validated here.)

| Well | Index | Value |
|---|---|---|
| ISCWSA #1 (interp 30 m) | $TI$ | $18.6413$ |
| ISCWSA #1 (interp 30 m) | $MTI$ | $0.6055$ |
| ISCWSA #2 (min-curvature) | $MTI$ | $0.5247$ |
| ISCWSA #2 (max-curvature) | $MTI$ | $0.7308$ |
| ISCWSA #2 (interp 30 m + max-curvature) | $MTI$ | $0.8200$ |

Table: End-of-well regression anchors (welleng, current release).

**Robustness.** On the Volve NO 15/9-F-2 as-drilled survey (Section 4) the minimum-curvature MTI is $0.338$ and moves less than $2\%$ (to $0.342$) as the section-continuity tolerance is tightened across three orders of magnitude, saturating at about ten sections; the eleven-control-point re-derivation (Section 4) returns the same value to three figures. The metric is thus insensitive to the threshold and to station count once the genuine curve turns are resolved — the reproducibility a KPI requires.

**Implementation note.** Two defects were found and fixed in passing while preparing this validation: a scale-factor keyword was read under a misspelt name, so the documented override was silently ignored; and the dogleg-continuity tolerance was not forwarded by one of the convenience methods, leaving that argument inert. Neither affected default outputs; both are now covered by regression tests.

## 7. A note on survey frequency

Because the minimum-curvature method fits the *least* curved path through a pair of survey stations, any tortuosity index computed from the survey alone *under-estimates* the true tortuosity, and does so more severely as the stations become sparser. Two surveys with identical station data but different station spacing return different indices (Table 2 shows the ordering — a maximum-curvature pre-processing that adds dogleg between stations raises the index, and interpolating first raises it further). The MTI is therefore an absolute KPI only where the survey spacing is controlled.

A maximum-curvature treatment that assigns a consistent, frequency-robust tortuosity to sparse and dense surveys alike — and its relationship to the survey-interval uncertainty term (XCLA/XCLH) already carried by the ISCWSA error model — is developed separately.

## 8. Conclusions

- The published Tortuosity Index combines an inclination-domain and an azimuth-domain index by root-sum-of-squares, but the two are built from the same three-dimensional arc and chord lengths and differ only in curve-turn detection; they are not independent, and the combination over-states three-dimensional tortuosity.
- Computing the index directly in three dimensions, sectionizing by the continuity of the curve-turn plane normal, avoids the double count and captures torsion.
- A Modified Tortuosity Index that divides each turn's excess ratio by its arc length and normalizes by total curve length is dimensionless, so values are comparable across unit systems; it supports a relative wellbore-quality KPI. ("MTI" here is the *Modified* index, distinct from the *Mapped* index of the same abbreviation in the literature.)
- The along-hole index *profile* — read as total, remaining and local tortuosity — communicates well quality where a single scalar cannot; it has been used to score candidate trajectories in subsequent well-planning work, and the same sectionization, run coarsely, recovers an inferred design trajectory when the plan is lost.
- Both indices are validated against a closed-form single-arc solution, dimensional and ordering invariants, and regression anchors, with an open-source implementation and test suite.
- Survey-frequency dependence and a maximum-curvature treatment are the subject of a companion paper.

## Reproducibility

The indices, tests, and worked examples are in the open-source library welleng (<https://github.com/jonnymaserati/welleng>): `welleng.survey.tortuosity_index`, `welleng.survey.modified_tortuosity_index`, the total/remaining/local reader `Survey.tortuosity_views`, and `tests/test_tortuosity_index.py`. The development of the method is documented in a series of posts at <https://jonnymaserati.github.io>.

\newpage

## References

- Almarzooqi, L. Z., Fonseca, R. M., Buhindi, H. A., Binabadat, E. K., Al Hamlawi, I., Dolle, N., Meshcheriakova, O. *Transforming Well Planning through an AI-Assisted Well Trajectory Optimization Approach Applied to an Offshore Field in the Middle East.* Abu Dhabi International Petroleum Exhibition and Conference, 2022. DOI: [10.2118/211041-MS](https://doi.org/10.2118/211041-MS)
- Ashok, P., et al. *Measuring Wellbore Tortuosity.* IADD luncheon presentation, 22 February 2018. <https://www.iadd-intl.org/media/files/files/47d68cb4/iadd-luncheon-february-22-2018-v2.pdf>
- D'Angelo, J., Ashok, P., van Oort, E., Shahri, M., Nelson, B., Thetford, T., Behounek, M. *Unplanned Tortuosity Index: Separating Directional Drilling Performance from Planned Well Geometry.* SPE/IADC International Drilling Conference and Exhibition, 2019. DOI: [10.2118/194099-MS](https://doi.org/10.2118/194099-MS)
- Equinor. *Volve field data set.* Released for open use, 2018. <https://www.equinor.com/energy/volve-data-sharing>
- Grisan, E., Foracchia, M., Ruggeri, A. *A Novel Method for the Automatic Grading of Retinal Vessel Tortuosity.* IEEE Transactions on Medical Imaging, 27(3), 310–319, 2008. DOI: [10.1109/TMI.2007.904657](https://doi.org/10.1109/TMI.2007.904657)
- Zhou, Y., Zheng, D., Ashok, P., van Oort, E. *Improved Wellbore Quality Using a Novel Real-Time Tortuosity Index.* SPE/IADC Drilling Conference and Exhibition, 2016. DOI: [10.2118/178869-MS](https://doi.org/10.2118/178869-MS) (the 3D combiner is their Eq. 4)
- Corcutt, J. *A Modified Tortuosity Index.* 2022. <https://jonnymaserati.github.io/2022/05/26/a-modified-tortuosity-index.html>
- Corcutt, J. *Modified Tortuosity Index: Tolerance Sensitivity.* 2022. <https://jonnymaserati.github.io/2022/06/11/modified-tortuosity-index-tolerance-sensitivity.html>
- Corcutt, J. *Modified Tortuosity Index: Maximum Curvature and Survey Frequency.* 2022. <https://jonnymaserati.github.io/2022/06/19/modified-tortuosity-index-survey-frequency.html>
