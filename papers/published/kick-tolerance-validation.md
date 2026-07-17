---
title: "The welleng kick-tolerance engine: derivation, provenance and public validation against published methods"
author: |
  Jonathan Corcutt\
  Corcutt Beheer B.V., Wassenaar, Netherlands\
  ORCID 0009-0008-1953-7760
date: "2026"
papersize: a4
geometry: margin=2.2cm
fontsize: 10pt
header-includes: |
  \usepackage{amssymb}
  \emergencystretch=3em
---

**Preprint — Version 1.0 (2026-07-15)**

## Abstract

Kick tolerance (KT) is a mandated well-control barrier calculation that sizes the
maximum gas influx a well can take and still circulate out without fracturing the
weakest exposed formation. `welleng` implements an open-source, deterministic,
single-bubble KT engine. This paper is its **public validation record**: it derives
the model, documents its provenance (what is transcribed from which published
source), and demonstrates that the implementation **reproduces the worked examples of
the established methods** — the pressure intermediates **digit-exact** and the influx
volumes to **about one percent or better with a fully self-contained gas backend**
(injecting no properties; the small residual traced to a source table's own rounding) —
across the single-bubble closed form of Thorogood et al. (SPE-208788-PA), the
temperature treatment and deviated form of Kiani Nassab et al. (SPE-202426-PA), the
influx-volume formula of Santos et al. (SPE/IADC-140113), and the mandated NOGEPA
Industry Standard No. 50 static formula — each run with **that source's own inputs and
assumptions**. It then demonstrates the
engine end-to-end on public Equinor Volve field data. The model is deliberately
**single-bubble and conservative** (safe-side): a coherent slug imposes the greatest
shoe pressure, so the tolerable influx it returns is a lower bound on what a real
dispersed or multiphase influx would allow — consistent with the published finding
that single-bubble models are conservative relative to dynamic multiphase simulators.
Every published-method validation figure (§2–§7) is reproduced by an automated test
in the `welleng` test suite; the Volve field demonstration (§9) is reproducible via
the accompanying example script.

## 1. Purpose, scope and what the engine is *not*

**Motivation — why a curated reference is needed.** Kick tolerance is a mandated,
safety-critical barrier calculation, yet the answer can swing materially with two
things that are rarely controlled: **which model is used**, and **the precision of the
inputs and constants**. This work makes both visible. The same well can differ by
several percent between published methods; a single ppg→psi gradient constant carried
to three decimals rather than four shifts the result; and near a casing-seat limit the
tolerance is *hypersensitive* — we measured ~29 % change in kick tolerance per 0.1 ppg
of assumed maximum pore pressure when the fracture margin is thin (§8). Despite this,
the calculation is very often performed in **in-house or personal spreadsheets with
little or no version control and no automated test** to confirm the engine still works
after someone edits a cell. For a calculation this consequential, that is
surprisingly slack. What the industry needs is a **standard, openly curated model that
produces repeatable, comparable results every time** — version-controlled, with
continuous-integration tests that fail loudly if the engine changes — so that an
operator or regulator can trust the number, and two parties computing the same well
get the same answer. This paper documents and validates `welleng`'s kick-tolerance
engine as exactly such a reference: open, version-controlled, and covered by automated
tests in which **every published-method validation figure is itself a test**.

There is also an access dimension. Today a trustworthy kick-tolerance calculation is
typically locked inside a **full commercial well-engineering software suite**, whose
license cost is out of proportion to this single, mandated calculation — so a smaller
drilling contractor is pushed toward exactly the unversioned in-house spreadsheet the
industry should be moving away from. A **drilling contractor should not have to buy a
full well-engineering suite license to perform a safety-critical kick-tolerance
calculation.** An open, curated, validated implementation removes that cost barrier
while *raising* the trust bar — the calculation is free to run and its correctness is
publicly auditable, which a licensed black box is not.

The purpose of this document is therefore trust: a user, reviewer or regulator should
be able to see, reproducibly, that the engine matches the methods the industry already
accepts. The engine is a **single-bubble, quasi-static, deterministic** calculator.
One coherent (insoluble) gas influx is sized by the shoe / weak-zone pressure balance
and, in the migration mode, marched up the annulus with Boyle + real-gas *Z* +
optional temperature expansion.

It is **not** a transient two-phase flow simulator: no gas/liquid slip or holdup, no
flow-regime transitions, no gas dissolution, no transient annulus temperature (the
domain of OLGA / Drillbench-class tools). That is a deliberate choice, not a gap —
see §8: the single-bubble result is the conservative, safe-side barrier, and a full
multiphase model would only *relax* the tolerance.

## 2. The single-bubble closed form (provenance: SPE-208788-PA, Appendix A)

The core margin logic is the public single-bubble closed form derived in Appendix A
of Thorogood et al. (2022), Eqs A-1…A-9: the bottom-hole pressure at the maximum
credible pore pressure (A-1), the capacity constants *A* (A-5) and *B* (A-6), and the
tolerable influx volume (A-7 drilled / A-8 the zero-margin swab limit). `welleng`
transcribes these directly (`welleng/kick_tolerance/core.py`); the gravitational
constant *g* = 0.0521 psi/(ppg·ft) matches the value implied by the paper's own
tabulated pressures.

**Validation — SPE-208788-PA Table 1 worked example** (vertical 6-1/8" hole, methane,
geothermal T-shoe 212 °F / T-td 302 °F, Hall–Yarborough *Z*):

| quantity | Eq | welleng | paper | note |
|---|---|---|---|---|
| P_td (PP+unc) | A-1 | 6892.8 psi | 6893 | digit-exact |
| B | A-6 | 7688.4 psi | 7688 | digit-exact |
| A | A-5 | 239.4 bbl | 242 | self-contained computed-gas *A* — see §3 |
| V_gas (drilled) | A-7 | 27.58 bbl | 27.86 | 1.0 % (self-contained gas) |
| V_gas (swab, KT=0) | A-8 | 43.25 bbl | 43.79 | 1.2 % (self-contained gas) |

Table: SPE-208788-PA Table 1 worked example — welleng against the paper's printed values.

The pressure intermediates (A-1, A-6) are **digit-exact**. The ~1 % residual on the
volumes is traced entirely to gas properties, not the margin logic, and resolves two
ways, both shown transparently: **(a)** feeding the paper's **own** rounded constant
*A* = 242 through A-7 reproduces its *V_gas* to **0.03 %** — the closed form is exact;
**(b)** running welleng's fully **self-contained** clean-room Hall–Yarborough gas
(injecting nothing) gives *A* = 239.4 and the ~1 % above, because the paper's Table 1
is itself internally inconsistent in *Z* (§3). We report the self-contained number as
the honest one: it is what a user gets with no injected properties. [test:
`tests/test_spe208788_worked_example.py`, `tests/test_kick_tolerance.py`]

**The swab limit and kick intensity.** A-8 is the zero-margin *swab* limit — A-7 with
the bottom-hole pressure set to the mud hydrostatic. A swabbed influx enters at the
mud hydrostatic, **independent of pore pressure and therefore of any kick-intensity
margin added to it**; Kiani Nassab et al. (SPE-202426-PA, Eqs 8–9) state this
explicitly and warn that models carrying PP (or PP + kick intensity) into the swab
bottom-hole pressure *over-report* swab tolerance. welleng's swab case is therefore
**kick-intensity-independent by construction** (it stations even the influx gas at the
mud hydrostatic). SPE-208788's own Eq A-8 reuses the single scenario-stationed *A* of
A-5, so its printed swab figure (43.79 bbl) is computed at the drill scenario pressure;
we **reproduce that figure under the paper's own convention** (43.25 bbl, the 1.2 %
above) while the shipped model follows Kiani Nassab — the two differ by ~1.5 % only when
PP + kick intensity exceeds the mud weight, and agree in the swab model's proper
overbalanced domain. [tests: `test_swab_reproduces_spe208788_shared_A`,
`test_swab_kick_is_kick_intensity_invariant`]

## 3. Real-gas *Z*: a clean-room Hall & Yarborough (1973) backend

The influx compressibility is computed by a clean-room implementation of the Hall &
Yarborough (1973) correlation for methane (`gas_z.py`); a CoolProp real-EOS backend
(`gas_z_coolprop.py`) is provided for mixtures and CO$_2$ (CCUS). Validation: the
clean-room backend reproduces the paper's own H-Y-derived properties to **~1 %** — *Z*_td
1.1665 vs 1.1650, $\rho$_gas_s 1.719 vs 1.710 ppg — so the closed-form validation of §2 does
not lean on injected properties the engine could not itself produce. The one larger
gap is the printed **shoe** *Z* (paper 1.1230): evaluated at the shoe pressure implied
by the paper's own static gas column, Hall–Yarborough returns *Z*_s $\approx 1.134$, and the
paper's tabulated *A* back-solves consistently with that value rather than the printed
1.1230 — i.e. Table 1 is internally inconsistent in the shoe *Z*, which is the entire
source of the §2 self-contained ~1 % residual. We note this neutrally against the
primary table; it does not affect the paper's method or conclusions.
[test: `tests/test_kick_tolerance.py::test_gas_properties_computed`]

## 4. Temperature

Two tiers. The **basic** model assumes **isothermal at the bottom-hole temperature** —
the most conservative choice (the whole well at its hottest point gives the lightest,
most-expanded gas everywhere, the biggest bubble and hence the lowest kick tolerance),
and the simplest (one number, no profile; the same T-neglecting posture as the
mandated NOGEPA-50 formula). The **advanced** model takes a user-supplied **geothermal
gradient**, which is more realistic and returns a higher (less conservative) KT — the
margin-recovery tier. Resolution is deterministic: an explicit temperature profile
wins, else the geothermal gradient, else isothermal at BHT. The
influx-temperature effect on KT is validated against **Kiani Nassab et al. (SPE-202426-PA),
Fig. 12** (Case 1): `welleng` reproduces the two *static* Company-Model points
near-exactly — geothermal **15.60 bbl** (figure $\approx 16$), isothermal **13.28 bbl**
(figure $\approx 13$); the frac pressure at the shoe, 4102 psi, matches the paper's 4095. The
engine cannot and does not reproduce the paper's *dynamic* Simulator-D point ($\approx 17$
bbl), a transient multiphase annulus-temperature simulation. [test:
`tests/test_spe202426_fig12.py`]

**On the temperature sign.** Kiani Nassab's Fig. 12 and its own text state the geothermal
assumption is *less* conservative than the isothermal-at-TD assumption ("the
isothermal assumption is more conservative, giving 23 % lower KT"); `welleng`
reproduces this ordering. The mechanism is transparent in the closed form: *A* $\propto$
*T_td / T_s*, so a geothermal profile (*T_td* > *T_s*) inflates *A*, and hence KT,
above any single-temperature isothermal. We cite Kiani Nassab's figure as the primary
source for the sign.[^tempnote]

Physically, this is just Charles's law: a higher temperature makes the influx a
**bigger, lighter bubble** ($V \propto T$), so for a given bottom-hole influx it occupies
more of the annulus as it rises and reaches the fracture-limiting gas column at the
shoe with a *smaller* bottom-hole volume — hence a **lower** kick tolerance. The
hottest case (isothermal at the TD temperature) therefore gives the lowest KT (13
bbl) and the cooler geothermal shoe gives a higher KT (16 bbl); the shoe temperature
*T_s* is the dominant lever.

[^tempnote]: The summary of this figure in SPE-208788-PA states the reverse
geothermal/isothermal ordering; we follow Kiani Nassab's figure and text directly.

## 5. Deviated wells (provenance: SPE-202426-PA, Eq. 4)

For a deviated well the gas column's vertical height *H_gas* is converted to an
along-hole length *L_gas* = *H_gas* / cos(inc) "using well trajectory" (Kiani Nassab Eq. 4)
before it multiplies the per-MD annular capacity; for constant inclination this
scales *A* — and the tolerable influx — by exactly 1/cos(inc_shoe), and inc_shoe = 0
recovers the SPE-208788 vertical value. [test: `test_spe208788_worked_example.py`]

## 6. The gas-migration engine (whole-path barrier check)

Beyond the single static shoe balance, `welleng` provides a **gas-migration** mode
that holds bottom-hole pressure at the kill value and marches the single bubble up
the section-wise annulus (Boyle + *Z* + temperature), requiring the imposed pressure
to stay within the pore–fracture (PP–FP) envelope at **every exposed open-hole depth
over the whole circulation**. Both bounds are the constant-bottom-hole-pressure
well-control principle — keep the imposed pressure above the pore pressure (no further
influx) and below the formation-breakdown / fracture limit (no lost circulation) — set
out in API RP 59 (2nd ed.) §4.10 and §12.5. The engine
returns the **maximum influx that can be circulated out**, *with the depth and
mechanism that limit it* — which may be a weak formation deeper than the shoe, or the
BHA, not the shoe assumed by the static single-shoe check. Two safeguards observed
during development:

- **Hole-volume ceiling.** If the whole exposed open hole can be displaced to gas
  without breaching the fracture envelope, the shoe-fracture tolerance is unlimited
  and reporting a volume larger than the hole volume is meaningless; the engine flags
  this explicitly. At full displacement the governing barrier becomes **casing burst**
  when the bubble reaches surface (a casing-design check; full displacement is the
  normal casing-design basis) — a documented extension using the API-5CT burst
  ratings.
- **True annular capacity and safe-side density** (bore − string, casing IDs from the
  API-5CT catalogue; gas-top-lightest density bound by default).

**An exact breakpoint solver, not a march.** The imposed pressure as the bubble rises
is piecewise-monotone in the gas position, so its extremum can only fall at a discrete
set of *breakpoints* — section-capacity changes, PP/FP profile breaks, and the gas
faces themselves. `welleng` therefore evaluates the envelope only at these breakpoints
and closes the tolerable influx in closed form / by a short bracketed root, rather than
marching a fine grid; the two agree, and the breakpoint form cannot miss a narrow
interior binding (e.g. a tight BHA section) that a coarse march can step over.

**Constraint scope is explicit (tiering).** The depths at which the fracture envelope
is enforced are selectable: the default is the full sections-aware set above, while
restricting it to the shoe alone recovers the classical **single-shoe** answer (deeper
FP jumps not binding). This makes the free single-shoe result and the full multi-depth
result the *same engine* under an explicit constraint set, not two separate code paths.
[tests: `tests/test_kick_analytical.py`]

This migration mode is `welleng`'s reproducible realization of the "not
oversimplified … volume which can be circulated out" that NOGEPA-50 §3.2 itself calls
for; its static reduction reproduces the mandated formula exactly.

## 7. Validation summary

| reference | what is reproduced | agreement |
|---|---|---|
| SPE-208788-PA (Thorogood) | Table-1 closed form A-1…A-9 | intermediates digit-exact; 0.03 % on the paper's own *A*; ~1 % fully self-contained (§2/§3) |
| SPE-208788-PA H-Y | *Z* / gas density | ~1 % (shoe *Z* Table-1 inconsistency, §3) |
| SPE-208788-PA A-8 | swab limit under the paper's shared-*A* convention | 1.2 % |
| SPE-202426-PA (Kiani Nassab) Fig. 12 | static geothermal/isothermal KT | $\approx 0.5$ bbl of the figure dots |
| SPE-202426-PA Eqs 8–9 | swab P_bh = mud hydrostatic, KI-independent | exact (by construction) |
| SPE-202426-PA Eq. 4 | deviated 1/cos(inc) form | exact |
| SPE/IADC-140113 (Santos) | influx-volume $V_1$ formula | <0.5 % ($\equiv$ $V_1$; welleng adds P_atm rigour) |
| NOGEPA-50 §3.2 | mandated static H→$V_1$→$V_2$ | reproduces the formula |

Table: Validation summary — published methods and standards reproduced by welleng.

**Tests** (all in `tests/`, run in CI):

- SPE-208788 closed form + deviated form — `test_spe208788_worked_example`
- swab shared-*A* reproduction — `test_swab_reproduces_spe208788_shared_A`
- Kiani Nassab Fig. 12 — `test_spe202426_fig12`
- swab KI-independence — `test_swab_kick_is_kick_intensity_invariant`
- Santos $V_1$ — `test_spe140113_santos`
- NOGEPA formula — `test_nogepa`

Santos et al.: `welleng`'s tolerable-influx `capacity` is algebraically Santos's $V_1$
(single bubble on bottom, Boyle-expanded to the shoe annulus); the two agree to
<0.5 %, the only difference being that `welleng` carries atmospheric pressure in
Boyle's law where Santos's simplified expression omits it.

**What we had to do to match the printed values (shown transparently).** With the
conventional safety margin (choke operator error 100 psi + choke-line friction 100
psi = 200 psi) `welleng` and a clean-room re-derivation of Santos's own formula
*agree with each other* but sit ~18 % above the paper's printed Well-A figures
($V_1$ $\approx 61$ vs 50.9 bbl). The printed values are self-consistent at a **235-psi** margin
— the extra ~35 psi is the **annular-friction** component Santos's text notes. Kick
tolerance is a **quasi-static** balance, so this annular pressure loss is not
*computed* by `welleng` (it models no flow); it is supplied as a **static input**
(`P_apl`), exactly as any hydraulics-derived annular loss would be. Setting `P_apl` to
include it, both the clean-room Santos formula and `welleng` reproduce the paper's
Well-A values exactly (Hmax 388 ft, Vshoe 67.9, $V_1$ 50.9, $V_2$ 48.7 bbl). Annular
friction is **conventionally taken as zero** in KT for sound reasons: the safety
margin is a drill-pipe-side well-killing margin, and well killing is deliberately
circulated at a **reduced (kill) rate**, so the annular pressure loss is small by
design. It is also not constant — the annular ΔP acting at the shoe depends on the
flowing geometry above the migrating bubble and falls as the bubble rises — so a
single fixed value (as Santos adds) is itself an approximation, and a quasi-static
barrier check has no basis to bake one in. `welleng` therefore defaults to no annular
component (the convention). A user who *does* want to include annular friction simply
**adds it to the choke / surface-pressure term** (`P_apl`) — the same static input that
carries the applied choke pressure — and the paper's Well-A values are then reproduced
exactly. That the same ~35-psi offset falls out of an independent clean-room
re-derivation confirms it is an input convention, not an implementation error.

## 8. Conservatism and positioning

The single-bubble result is **conservative (safe-side)**: a coherent slug imposes the
greatest shoe pressure, so the tolerable influx is a **lower bound** on what a real
dispersed or multiphase influx would allow. This is the published consensus, not our
assertion — SPE-208788-PA's "Conservatism of Kick Tolerance Calculations" section
notes that "kick tolerance values obtained from the simplified models are generally
lower (i.e., more conservative) than those obtained from the sophisticated
simulations" (citing Leach & Wand 1992; Nakagawa & Lage 1994; Rommetveit 1994) and
that "the single bubble is considered conservative" (Karahasan et al. 2017); Kiani Nassab's
Fig. 12 shows the same (static 13–16 bbl vs dynamic $\approx 17$ bbl). For a safety barrier
this is the correct direction. A full transient multiphase model would only *relax*
(raise) the tolerance — a casing-design margin-recovery tool, not a safety
improvement — and its transient two-phase hydraulics belong with a hydraulics kernel,
not this barrier check.

**Sensitivity — and why a curated reference matters.** Kick tolerance is
hypersensitive to the assumed maximum pore pressure when the fracture margin is thin.
For the deviated Volve-scale example of §9, with a fracture margin of only ~170 psi,
we measured a **~29 % change in kick tolerance per 0.1 ppg** of assumed maximum pore
pressure ($\approx 5.7$ bbl per 0.1 ppg). At a fatter margin (~410 psi) the same input change
moves the answer only a few percent. This is the physical reason the §7 reconciliation
of a published worked example's printed values is so input-dependent — near a
casing-seat limit, sub-0.1 ppg differences in inputs, and even the number of decimals
carried on a gradient constant, are amplified into double-digit-percent swings in the
answer. The worst-case pore-pressure *input itself* is contested: Santos & Sonnemann
(SPE-159175-MS) contrast the "predicted pore pressure + kick intensity" and "mud
weight + kick intensity" conventions for defining it — the same convention split that
governs our swab treatment (§2) — so which number a user enters is not even uniquely
defined across the industry, compounding the sensitivity. For a mandated,
safety-critical barrier that is a strong argument for a single **curated,
version-controlled, CI-tested reference model** producing repeatable, comparable
results, rather than method- and spreadsheet-dependent numbers that cannot be audited
or reproduced.

## 9. Field demonstration (public Volve data)

The engine runs end-to-end on Equinor Volve well F-13 (real pore/fracture profiles,
definitive survey, nested casing program from the public EDM export; 13-3/8" shoe at
7737 ft TVD, TD 10148 ft). Drilling the 8-1/2" hole below the 13-3/8" shoe, the maximum gas
kick tolerance is **79.4 bbl** (isothermal at the EDM bottom-hole temperature 223 °F)
rising to **86.0 bbl** with the real Volve geothermal gradient (0.0191 °F/ft), binding
at 7904 ft TVD — ~170 ft below the 13-3/8" shoe, the fracture-weakest exposed point.
[Data: Equinor Volve, CC BY 4.0; reproduced by `examples/kick_tolerance_volve.py`.]

## 10. Reproducibility and continuous validation

The engine is open source (`welleng.kick_tolerance`). Every published-method
validation figure (§2–§7) is checked by an automated test in the `welleng` suite, run
in CI; the Volve field demonstration (§9) is reproducible via the accompanying example
script, and the input-sensitivity figures (§1, §8) are measured from that example.
Inputs for each published example are transcribed from the source PDFs and cited by
table / figure / equation.

Crucially, this makes validation a **continuous property of the maintained codebase,
not a one-time result**. Each of the reproductions above is a test that re-runs on
every change; if a future edit alters the engine's output, the corresponding
validation fails loudly and visibly, rather than the number drifting silently as it
can in an unversioned spreadsheet. A safety-critical reference is only trustworthy for
as long as it stays validated, and continuous integration is what keeps it so.

## Assumptions (explicit)

Single coherent insoluble bubble; methane (or a user-specified CoolProp mixture);
worst-credible pore pressure; fracture at the weakest *checked* exposed depth (shoe,
weak zone, or BHA in the migration mode); annular capacity = bore − string;
deterministic (no Monte-Carlo); quasi-static (no transient two-phase hydrodynamics).

\newpage

## References
- Thorogood, J., Robertson, E., Castillo, D., Sawaryn, S. (2022). *An Assessment of the
  Kick Tolerance Calculation, Its Uncertainty, and Sensitivity.* SPE Drilling &
  Completion 37(3):232–243. SPE-208788-PA. DOI: [10.2118/208788-PA](https://doi.org/10.2118/208788-PA)
- Kiani Nassab, K., Ting, S.Z., Buapha, S., MatNoh, N., Hemmati, M.N. (2022). *How to
  Improve Accuracy of a Kick Tolerance Model by Considering the Effects of Kick
  Classification, Frictional Losses, Pore Pressure Profile, and Influx Temperature.*
  SPE Drilling & Completion 37(1):15–25. SPE-202426-PA. DOI: [10.2118/202426-PA](https://doi.org/10.2118/202426-PA)
- Santos, H., Catak, E., Valluri, S. (2011). *Kick Tolerance Misconceptions and
  Consequences to Well Design.* SPE/IADC Drilling Conference and Exhibition, Amsterdam,
  1–3 March 2011. SPE/IADC-140113-MS. DOI: [10.2118/140113-MS](https://doi.org/10.2118/140113-MS)
- Santos, H., Sonnemann, P. (2012). *Transitional Kick Tolerance.* SPE Annual Technical
  Conference and Exhibition, San Antonio, Texas, 8–10 October 2012. SPE-159175-MS.
  DOI: [10.2118/159175-MS](https://doi.org/10.2118/159175-MS)
- NOGEPA (Netherlands Oil and Gas Exploration and Production Association) (2020).
  *Industry Standard No. 50 — Kick Tolerances for Well Design and Drilling Operations.*
  Version 09-12-2020.
- API (2023). *Recommended Practice for Well Control Operations* (API RP 59, 2nd ed.).
  §4.8.7.3, §4.10, §12.5.
- Hall, K.R., Yarborough, L. (1973). *A new equation of state for Z-factor
  calculations.* Oil & Gas Journal 71(25):82–92.
- Bell, I.H., Wronski, J., Quoilin, S., Lemort, V. (2014). *Pure and Pseudo-Pure Fluid
  Thermophysical Property Evaluation and the Open-Source Thermophysical Property Library
  CoolProp.* Industrial & Engineering Chemistry Research 53(6):2498–2508.
  DOI: [10.1021/ie4033999](https://doi.org/10.1021/ie4033999)
- Equinor Volve Data Village (public, CC BY 4.0) — the field demonstration data.
