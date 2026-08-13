# Catalogue data — sources (cite by number; never bundle the PDFs)

Dimensional/weight data are factual industry-standard values extracted into an
SI-ready schema. API documents are copyrighted — cited here **by number**, not
reproduced. No API PDF text is copied into this repository.

- **casing.json** — API Spec 5CT (Specification for Casing and Tubing), casing
  dimensional tables (OD, nominal weight, wall thickness, ID). ID = OD − 2·wall.
  Drift = ID − API standard drift undersize (1/8 in for OD ≤ 8⅝, 5/32 in for
  9⅝–13⅜, 3/16 in for OD ≥ 16). Geometry is grade-independent; grade sets the
  minimum yield only (see `grades`: J55/K55/L80/N80/P110 = 55/55/80/80/110 ksi).
- **tubing.json** — API Spec 5CT (Specification for Casing and Tubing), tubing
  dimensional tables (OD, nominal weight, wall, ID). Drift = ID − 3/32 in for
  OD < 3½, − 1/8 in for OD ≥ 3½. Grade → min yield as above.
- **couplings.json** — API Spec 5CT (11th ed., 2023), coupling dimensions
  (regular OD *W*, special-clearance OD *Wc*, minimum coupling length *NL*),
  cited by table number: **Table E.27** (API Round Thread Casing Coupling —
  STC uses the Short *NL* column, LTC the Long *NL*; *W* is shared),
  **Table E.28** (API Buttress Thread Casing Coupling — BTC, incl. special
  clearance), **Table E.29** (API Non-upset Tubing Coupling — NUE),
  **Table E.30** (API External Upset Tubing Coupling — EUE, incl. special
  clearance). USC-unit annex E; SI equivalents are Tables C.27–C.30. Values
  are dimensional facts only; no table text/layout reproduced.

## Grade & body-performance enrichment

- **Grade tensile properties** (`grades` block: `max_yield_psi`, `min_uts_psi`;
  `min_yield_psi` unchanged) — API Spec 5CT (11th ed., 2023) **Table E.5**
  (Tensile and Hardness Requirements).
- **Body performance** — computed by the loader from the *formulas* in
  **API TR 5C3** (7th ed., 2018), cited by equation (TR 5C3 tabulates methods,
  not per-size values), and surfaced on the resolved `TubularSpec`:
  - `plain_end_weight_ppf` = 10.69·(D−t)·t (historical API plain-end mass;
    geometry-only, grade-independent).
  - `min_wall_pct` = 87.5 (wall-tolerance factor k_wall = 0.875; TR 5C3 6.6.2.2).
  - `pipe_body_yield_klb` = f_ymn·(π/4)(D²−d²) — **TR 5C3 Eq. (11)**.
  - `internal_yield_pressure_psi` = 2·f_ymn·(k_wall·t)/D (Barlow) —
    **TR 5C3 Eq. (10)**, k_wall = 0.875, rounded to nearest 10 psi.
  - `collapse_pressure_psi` — **TR 5C3 8.4** four-regime design equations
    (yield Eq.35 / plastic Eq.37 / transition Eq.39 / elastic Eq.41), regime
    boundaries Eqs (36)/(38)/(40), empirical factors A/B/C/F/G from **Table 6**
    (plastic) + **Table 7** (transition). `None` for any grade whose factors are
    not tabulated (flagged, never guessed).
  - Validation: 9⅝″ 47 lb/ft L80 → 1086 klb / 6870 psi / 4760 psi, matching the
    published API values exactly.

## Connection performance (ConnectionSpec)

`ConnectionSpec` carries the full connection datasheet field set. For **API**
connections only the *dimensional* fields (`connection_od_in` = W,
`coupling_length_in` = NL) are filled, from 5CT Tables E.27–E.30. All
premium-performance fields (efficiencies, strengths, pressure resistances,
make-up torques, delta-turn, bending limits) are **proprietary and
user/vendor-supplied** for premium threads (VAM, Tenaris, etc.) and are left
`None` — NOT vendored, never fabricated.

Conversions applied by the loader (from each file's `_meta.to_SI`):
`in × 0.0254 = m`, `lb/ft × 1.4881639 = kg/m`, `psi × 6894.757 = Pa`.

## Premium connections (`premium_connections.json`)

A NAME + provenance **registry** of premium (proprietary metal-to-metal /
gas-tight) connection designations — VAM TOP/ACE, Hydril 563, NSCC, Fox, BDS —
observed on real completion strings. The designations were parsed from the
**Volve open dataset (Equinor)** WellCat completion models; each connection's
specification is published on its **vendor's public website**. Equinor released
the Volve dataset publicly and these are public vendor/industry identifiers
(no operator, well, or individual identifiers) — provenance queries go to
Equinor's Volve open-data release and to the respective vendor.

Consistent with the premium-thread policy above, the registry asserts **no
dimensions and no ratings**: `id_in`, `drift_in`, joint efficiency, pressure
rating and make-up torque are proprietary/vendor-supplied and left `None` —
consult the vendor datasheet, never fabricated. `vendor` is asserted only where
the brand owner is unambiguous (VAM → Vallourec, Hydril → Tenaris); a `null`
vendor (Fox, BDS) means it was not determined with confidence — not guessed.

## OSDU mapping (TubularComponent.1.0.0)

| welleng            | OSDU property                                   |
|--------------------|-------------------------------------------------|
| od_in              | MaximumOuterDiameter / TubularComponentNominalSize |
| id_in              | InnerDiameter                                   |
| drift_in           | DriftDiameter                                   |
| nominal_weight_ppf | TubularComponentNominalWeight                   |
| grade              | TubularComponentTubingGradeID                   |
| yield_psi          | TubularComponentTubingGradeStrength             |
| (type)             | TubularComponentTypeID                          |

Wall thickness is derived ((OD−ID)/2) and is **not** stored in OSDU.

These catalogues are the casing/tubing counterparts of the drilling-mechanics
drill-string catalogues (`drillpipe.json`, `tooljoints.json`, `drillcollar.json`,
`hwdp.json`) and share their `_meta` + rows format.
