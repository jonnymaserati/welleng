---
title: "Survey Resolution — Not Survey Frequency or Model Order — Bounds Trajectory-Derived Fidelity"
author: |
  Jonathan Corcutt\
  Corcutt Beheer B.V., Wassenaar, Netherlands\
  ORCID 0009-0008-1953-7760
date: "2026"
papersize: a4
geometry: margin=2.2cm
fontsize: 10pt
colorlinks: true
linkcolor: black
urlcolor: blue
citecolor: blue
header-includes: |
  \usepackage{amssymb}
  \emergencystretch=3em
---

**Version 1.0 — 2026 · DOI: [10.5281/zenodo.21906319](https://doi.org/10.5281/zenodo.21906319) · companion code: welleng (Apache-2.0)**

*A worked, self-correcting note on why denser surveys and higher-order models hit the same physical wall.*

*Cite as:* Corcutt, J. (2026). *Survey Resolution — Not Survey Frequency or Model Order — Bounds Trajectory-Derived Fidelity.* Zenodo. <https://doi.org/10.5281/zenodo.21906319>

## Abstract

It is intuitive that a denser survey, or a higher-order mechanical model, should
give a more accurate picture of a well. This note examines that intuition
directly. Working through the full modelling stack we have been building — the
open-source `welleng` library for the survey side (the ISCWSA extended-course-length
(XCL) survey-interval error term, a maximum-curvature interpolator, the tortuosity index,
and minimum-curvature covariance propagation), together with a tiered torque-and-drag
(T&D) engine in its drilling companion — we arrive at a single conclusion that holds
across both position uncertainty and T&D: **fidelity is
bounded by the resolution of the survey, not by how often you survey below that
resolution, nor by the order of the mechanical model.** The physical cause is common
to both: the stiff near-bit assembly band-limits the wellbore it drills, so geometry
shorter than the assembly's bending wavelength is neither drilled, nor sensed, nor
traversed as extra measured depth. The adopted survey-interval error model already
encodes this floor; a higher-order T&D model cannot recover curvature the survey never
captured. We also record, honestly, that we set out to propose a *new* error term and
had to correct ourselves against the primary literature en route — that correction is
part of the lesson. The practical guidance is simple: match survey frequency to the
physical floor, match model tier to the data and the question, and recognise where the
genuinely open problem still lies.

## 1. An intuition worth examining

Two instincts recur in directional drilling, and both feel obviously correct:

1. *If I survey more often, my position is more certain.*
2. *If I use a stiffer, higher-fidelity mechanical model (soft-string, then stiff-string, then
   full 3D finite-element), my torque-and-drag prediction is more accurate.*

Both are reasonable first thoughts. Both are, past a point, wrong — and they are wrong
for the *same* physical reason. The purpose of this note is not to catch anyone out; it
is to make that shared reason clear, because once it is clear it changes how you spend
effort. The instinct to densify is expensive, and much of the expense buys nothing real.
The survey-interval effect itself is long recognised but, in Jamieson's words, "very hard
to quantify" (Jamieson, *Introduction to Wellbore Positioning*, §8.2); the aim here is to
show why the useful part of it is physically bounded.

## 2. What we did: model it, comprehensively

This is not an argument from the armchair. The conclusion here is the output of modelling
the survey-interval problem across our modelling stack, applied together. Four of the
five components are in the open-source `welleng` library, so the position-side result is
reproducible by any reader:

- the **XCL survey-interval error term** (Codling, SPE/IADC-187249; adopted in the ISCWSA
  MWD error model Rev5), which converts survey spacing into a covariance contribution;
- a **maximum-curvature interpolator** (`Survey.maximum_curvature`), which reconstructs
  the tightest physically-admissible path consistent with the survey stations and
  quantifies the geometry a straight minimum-curvature reconstruction (Taylor & Mason,
  SPE-3362) leaves out;
- the **tortuosity index / modified tortuosity index**, which accumulate curvature — the
  geometric driver of the capstan term in T&D;
- **minimum-curvature covariance propagation**, the position-uncertainty engine.

The fifth is a **tiered torque-and-drag engine** (soft-string, then stiff/beam-column,
then experimental 3D) in `welleng`'s separate drilling companion — not part of the public
library — used here so the effect of *model order* could be measured rather than assumed.
The T&D claims below rest on that modelling together with the published literature cited;
the position claims are reproducible directly in open `welleng`.

We set out, in fact, to *find a missing term* — a signed, directional survey-interval
bias, and a high-frequency floor we believed the XCL term lacked. The stack did not
confirm the hypothesis. It showed, repeatedly and from several independent directions,
that the wall is physical and that the standard already respects it. Sections 3–5 are
that result; Section 6 is the correction we had to make, kept in because it is the most
useful part for anyone tempted down the same path.

## 3. The wall: the drillstring band-limits the wellbore

A wellbore is not free to take any shape. It is cut by a bottom-hole assembly (BHA)
whose near-bit section is stiff, and that stiffness sets a shortest wavelength of
curvature the assembly can physically produce. Below that wavelength three things all
fail at once:

- **It is not drilled.** A stiff collar "cannot possibly conform" to short-wavelength
  undulations; it rides the crests (Gaynor, SPE/IADC-67818). The drilled centreline is
  band-limited at the source.
- **It is not sensed.** The MWD sensor sits inside that same stiff collar, so it measures
  the band-limited tangent — it cannot report curvature the collar cannot follow.
- **It is not traversed.** Measured depth is a pipe tally: a fixed length of steel. Even
  if the string drapes slightly into short-wavelength waviness, the extra path length is
  second-order (well under a millimetre per stand) — MD is, to first order, the smooth
  centreline length regardless.

![A stiff drill collar (6¾-inch) in an 8½-inch hole lies on the low side and follows the smooth planned curvature (here a gentle 2°/30 m build). Short-wavelength micro-tortuosity (pitch 0.6–3 m) is superimposed on the wall; being stiff, the collar cannot conform to it — it rides the crests and bridges the troughs (Gaynor, SPE/IADC-67818). Measured depth is a tally along that smooth collar, so the wall tortuosity is shortcut out of the recorded length, and the MWD sensor housed in the collar reads only the smooth, bridged tangent. All radial dimensions (hole, collar, 22 mm clearance, the 2°/30 m deflection, wall roughness) are mutually to scale; the along-hole axis is compressed. The cross-section inset is fully to scale. The bridged-out excess wall length is of the order of 1.4–3 %.](figures/fig_bridging_md_shortcut.png){width=95%}

Write the shortest admissible wavelength as $\lambda^{*}$. Its scale is set by the greater
of a clearance limit and a bending-stiffness limit,
$$\lambda^{*} \sim \max\!\left(2\pi\sqrt{c/\kappa},\; 2\pi\sqrt{EI/F}\right),$$
with $c$ the collar-to-wall clearance, $\kappa$ curvature, $EI$ bending stiffness and $F$
axial load. These are order-of-magnitude estimates — good to a factor of two, not more —
and $\lambda^{*}$ *scales with hole size*: of the order of a single stand (30 m) in a
slim hole, and larger — of the order of 100 m — in a large surface hole. The important
point is not the exact number; it is that a floor *exists*, that it is physical, and that
it is not zero.

## 4. Consequence one — position uncertainty

Now feed this into the survey-interval error term. The XCL term (Codling, SPE/IADC-187249)
converts the survey spacing into an uncertainty contribution. A natural worry — and one we
held — is that as the survey interval shrinks toward zero, such a term charges without
bound, implying you could survey your way to fake precision.

The standard already prevents this. The XCL formulation floors the randomised
misalignment contribution at a minimum course length of 10 m, and states the reason
explicitly: the randomisation is *"limited by the ending stiffness of the drill collar
or casing that the survey is run inside"* (ISCWSA MWD error-model Rev5 technical
supplement; the terms were introduced by Codling, SPE/IADC-187249). In other words: the
model does not let you claim uncertainty structure from wiggle the collar cannot cut. This
is exactly $\lambda^{*}$, expressed in the error model. Surveying below the floor does not
sharpen the covariance in any physical way — it models measurement noise and reconstruction
artefacts, not geometry. The adopted model is behaving correctly; the value of stating it
plainly is that it tells you *why*, and where the floor comes from.

## 5. Consequence two — torque-and-drag fidelity

The same floor governs mechanical modelling, through a different route. In the classical
soft-string model (Johancsik et al., SPE-11380) the side (normal) force combines a
gravitational part $w\sin I$ and a tension-times-curvature (capstan) part $T\,\kappa$; the
paper's general 3D form combines these as a vector magnitude, which reduces in the planar
case to
$$N = w\sin I + T\,\kappa .$$
Either way, the curvature $\kappa$ is precisely what the survey band-limits. So:

- **Densifying the survey below $\lambda^{*}$ injects curvature that is not there.** The
  extra $\kappa$ is reconstruction ripple, not drilled geometry; it produces phantom
  normal force and *over-predicts* drag. More survey points make the answer look more
  detailed and less true.
- **Increasing model order recovers no drag the survey did not resolve.** On smooth,
  survey-resolution geometry a soft-string and a stiff-string model agree *on the
  drag / normal-force result* (Mitchell & Samuel, SPE-105068); the stiff model additionally
  resolves the bending stress the soft form only estimates (via a bending-stress
  magnification factor), but that is a *stress* result, not extra drag fidelity. On drag,
  the two diverge only where curvature exists to bend against — only on tortuosity the
  survey actually captured (Nobbs et al., SPE-207935-MS). Off a standard 30 m survey, a full
  3D finite-element *drag* result is not more accurate than a soft-string one: the
  mathematics can be exact while the *claim of drag accuracy* outruns what the data
  resolves. That is false precision — not an argument against stiff/FEA where bending stress
  or buckling is the actual question.

The correct posture is to match the model tier to the survey resolution *and* to the
question being asked: soft-string for aggregate, screening-level drag off a standard
survey; stiff/beam-column where resolved doglegs demand bending stress and buckling
(Dawson & Paslay, SPE-11167) that
the soft form structurally omits; and a micro-resolved model only where micro-resolved
geometry actually exists to justify it.

## 6. The correction we had to make (and why it stays in)

We began this work intending to publish a *new* survey-interval error term: a signed,
always-shallower directional bias from steering, plus an explicit high-frequency floor we
thought the XCL term was missing. Due diligence against the primary sources dismantled
both claims, and the honest record of that is more useful than a tidy result:

- The **high-frequency floor is not missing** — it is already in the standard (the Rev5
  minimum-course-length limit, Codling) and was described physically decades ago (Gaynor;
  the Woods–Lubinski collar-drift geometry Gaynor builds on). Our "stiff ruler" was correct physics, already
  published.
- The **directional bias, as a survey-only quantity, does not stand.** Its direction is
  already recoverable from the survey (the net-dogleg toolface, which the maximum-curvature
  method uses) and the high-side XCL component already carries it. A genuinely *deterministic*
  correction would need the recorded toolface and slide record — information that lives in
  the directional driller's log, not in the survey, and is generally not shared. So this is
  an open problem gated by data availability, not a term we can responsibly propose today.

Along the way we discarded two mechanisms we had initially reached for — a torque-driven
undulation (the amplitude is too small) and drill-pipe bridging (tensioned pipe sags *into*
troughs rather than riding over them) — each corrected against the physics rather than
defended. For an early-career reader the transferable lesson is this: **suspect your own
hypothesis first, read the primary source of any term you mean to improve, and let the
model talk you out of an appealing story.** That discipline is why the conclusion that
survives is trustworthy.

## 7. What this means in practice

- **There is a physical floor to useful survey frequency**, of order one stand and
  hole-size dependent. Below it, additional stations model noise and reconstruction
  artefacts, not the well. Spend the survey budget up to the floor, not past it.
- **Model order is not a free accuracy dial.** A higher-fidelity mechanical model earns its
  keep only when the survey resolves the geometry that model needs. Match the tier to the
  data and the decision.
- **The real frontier is elsewhere.** The remaining, genuine gain is a drilling-record-aware
  correction that uses the recorded toolface and slide programme — deterministic where the
  survey-only view can only be statistical. It is open, and it is blocked on data access, not
  on mathematics. That is where effort is well spent.

## 8. Open tools

The position-uncertainty tooling is in the open-source `welleng` library (Apache-2.0):
`Survey.maximum_curvature` for the geometry under-representation; `tortuosity_index` /
`modified_tortuosity_index` for the geometric T&D proxy; and the minimum-curvature
covariance with the ISCWSA error-model implementation for position uncertainty. Readers
are encouraged to reproduce the floor for their own hole sizes and BHAs — the point of an
open reference engine is that the position-side conclusion above is something you can
check, not something you have to take on faith. (The tiered T&D engine referenced in
Sections 2 and 5 lives in a separate drilling companion, not the public library; the T&D
conclusion here stands on the published literature cited.)

## References

_Citations verified against the local PDF pages (2026-08-12); the torque-and-drag section
(§5) and its references were co-reviewed and signed off by welleng-drilling._

- Codling, J. et al. *The Effect of Survey Station Interval on Wellbore Position
  Uncertainty.* SPE/IADC-187249-MS (2017). [Introduces the XCL survey-interval terms;
  Table 3 propagates XCL Inc/Azi as **Random**, weighting $\mathrm{Max}(|\Delta I|,\,T\cdot\Delta\mathrm{MD})$
  — verified from PDF p. 9.]
- ISCWSA MWD Error Model Rev5 — *XCL Terms and Low Angle Misalignments, Technical
  Supplement.* [Carries XCLA/XCLH as the adopted Rev5 terms and floors the misalignment
  randomisation at a 10 m minimum course length, *"limited by the ending stiffness of the
  drill collar or casing that the survey is run inside"* (p. 5) — verified from PDF.]
- Gaynor, T. M., Chen, D. C-K., Stuart, D., & Comeaux, B. *Tortuosity versus
  Micro-Tortuosity — Why Little Things Mean a Lot.* SPE/IADC-67818 (2001). [Collars
  "cannot possibly conform … lie on the crests"; micro-tortuosity "not measurable by
  survey data" and dominant; pitch 2–10 ft; Woods–Lubinski drift $=$ (Bit$+$Collar)/2
  (Eq. 6); excess wellbore length 1.4 % (up to 3 % at large clearance, Eq. 4) — verified
  verbatim from PDF, pp. 1, 3, 4.]
- Johancsik, C. A., Friesen, E. B., & Dawson, R. *Torque and Drag in Directional Wells —
  Prediction and Measurement.* SPE-11380-PA, JPT (1984). [Soft-string (lumped-parameter)
  side force is the vector magnitude of a gravity term $w\sin I$ and tension×angle-change
  (capstan) terms, reducing to $N=w\sin I+T\kappa$ in the planar case (Eq. 1, p. 988) —
  verified.]
- Nobbs, B., Aichinger, F., Dao, N.-H., & Studer, R. *Stiff String Casing Design:
  Tortuosity and Centralisation.* SPE-207935-MS (2021), Helmerich & Payne. [Soft-string
  $\approx$ stiff-string on smooth geometry, diverging only when tortuosity is present;
  "the stiffness, radial clearance and high frequency surveys needed to fully model local
  doglegs are rarely modeled" (Abstract) — verified from PDF.]
- Jamieson, A. *Introduction to Wellbore Positioning* (ISCWSA / UHI), §8.2.
  [Survey-interval effect "very hard to quantify" — verified.]
- Taylor, H. L. & Mason, C. M. *A Systematic Approach to Well Surveying Calculations.*
  SPE-3362 (1972). [Minimum-curvature method — verified.]
- Mitchell, R. F. & Samuel, R. *How Good Is the Torque/Drag Model?* SPE-105068-PA (2009).
  [Soft-string and stiff-string agree on the drag / normal-force result on smooth geometry —
  the torque-and-drag model-fidelity thesis; welleng-drilling-verified.]
- Dawson, R. & Paslay, P. R. *Drillpipe Buckling in Inclined Holes.* SPE-11167 (1984).
  [Sinusoidal buckling-onset criterion the soft-string form omits; welleng-drilling-verified.]
