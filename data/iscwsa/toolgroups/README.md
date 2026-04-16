# OWSG toolgroup workbooks (ISCWSA / OWSG public reference data)

Canonical, publicly redistributable Excel workbooks defining the OWSG
standard tool error models. These are the source data consumed by the
``welleng/errors/tools/owsg_to_json.py`` converter to produce the
ISCWSA-format JSON tool models in ``welleng/errors/iscwsa_json/``.

## Contents

- `toolgroup-owsg-a-rev-5-1-08-oct-2020-produced-23-sep-2022.xlsx` —
  **OWSG Set A, Rev 5-1** (October 2020). Covers the standard tool
  variants: MWD with BGGM/IFR1/IFR2 + AX/SAG/MS combinations, EMS,
  film tools, gyro (GYRO-MWD, GYRO-NS, GYRO-NS-CT), inclination-only,
  and BLIND. Each tool sheet has its full per-error-term table:
  weight-function name, magnitude, units, propagation mode, and
  formula strings (depth/inclination/azimuth/singularity).
- `toolgroup-owsg-b-rev-5-1-08-oct-2020-produced-23-sep-2022.xlsx` —
  **OWSG Set B, Rev 5-1**. Extended models on top of Set A
  (HRGM, IGRF, advanced EMS combinations). MWD/EMS only; no gyro.

## Source

Downloaded from the public ISCWSA (Industry Steering Committee on
Wellbore Survey Accuracy) website. The same files are referenced in
the README of [iscwsa/error-models](https://github.com/iscwsa/error-models),
the upstream JSON-schema repository:

- Set A: <https://www.iscwsa.net/files/807>
- Set B: <https://www.iscwsa.net/files/808>

The OWSG (Operators Wellbore Survey Group) is the operators-side
group that maintains the canonical *tool model data* (magnitudes,
identifiers, applicability ranges) using the ISCWSA framework.
ISCWSA defines the *math* (weight functions, propagation rules, via
SPE 67616 / 90408 / etc.); OWSG provides the standardised reference
values for specific tools. They publish jointly.

## Licence

The OWSG toolgroup workbooks are distributed by ISCWSA for industry
use without attached licence text. Treated here as public reference
data, freely redistributable as the canonical input to any compliant
implementation of the ISCWSA error model.

## Why these are tracked

The companion ``data/iscwsa/`` directory is mostly gitignored (per
the project's convention to keep large datasets out of the
repository). The OWSG toolgroup xlsx files are an exception — they
are small (~700 kB combined), public, and the conformance-harness
work depends on them being deterministically present at a relative
repo path. The selective un-ignore lives in the project root
``.gitignore``.
