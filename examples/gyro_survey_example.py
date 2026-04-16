'''
examples/gyro_survey_example.py
-------------------------------
Demonstrates welleng's new ISCWSA gyro error model support, comparing
position uncertainty for the same wellbore surveyed with three
different tool stacks:

  1. ``ISCWSA MWD Rev5.11``   -- legacy hand-coded path (YAML-driven).
  2. ``GYRO-NS``              -- North-Seeking gyrocompass, gyrocompass
                                 stop at every survey station. JSON-driven
                                 via the new formula-string interpreter.
  3. ``GYRO-NS-CT``           -- mixed gyrocompass + continuous gyro
                                 (gyrocompass at survey stations, integrated
                                 gyro rates while running between them).

The gyro models are driven by the canonical OWSG Set A Rev 5-1 (Oct
2020) tool definitions, converted to ISCWSA JSON schema format and
shipped at ``welleng/errors/iscwsa_json/owsg_a/``. The interpreter
that evaluates the per-term formula strings lives at
``welleng/errors/interpreter.py``; the dispatch logic in
``welleng/errors/tool_errors.py`` routes ``error_model='GYRO-...'`` to
the JSON path automatically.

Some terms in the new gyro tool models reference variables the
ISCWSA JSON schema has not yet formalised (cross-station
``MDPrev`` / ``AzPrev`` / ``IncPrev``, per-tool calibration constants
like ``NoiseReductionFactor``). These emit a ``RuntimeWarning`` at
``Survey`` construction and contribute zero covariance to the
specific terms that need them. Everything else evaluates cleanly.
See ``welleng/errors/conformance.py`` for the parallel-paths
agreement matrix that catalogues exactly which terms are unaffected.

author: welleng contributors
date: 2026-04-16

Requirements: pip install welleng
'''

from __future__ import annotations

import warnings

import numpy as np

import welleng as we


# ----------------------------------------------------------------------
# Build a representative wellpath
# ----------------------------------------------------------------------
# Rather than depending on any external survey file, we build one in
# code so the example runs anywhere welleng installs.

print("Constructing a build-and-tangent wellpath ...")
connector = we.connector.Connector(
    pos1=[0.0, 0.0, 0.0],
    inc1=0.0,
    azi1=0.0,
    pos2=[800.0, 200.0, 3000.0],
    inc2=60.0,
    azi2=14.0,
)
trajectory = we.survey.from_connections(connector, step=30.0)

header = we.survey.SurveyHeader(
    name="example",
    latitude=60.0,        # North-Sea-ish
    b_total=50000.0,      # nT
    dip=72.0,             # deg
    declination=-4.0,     # deg
    convergence=0.0,
    G=9.80665,
    azi_reference="grid",
)


def make_survey(error_model: str) -> we.survey.Survey:
    """Build the Survey with the selected error model."""
    return we.survey.Survey(
        md=trajectory.md,
        inc=trajectory.inc_deg,
        azi=trajectory.azi_grid_deg,
        header=header,
        error_model=error_model,
    )


# ----------------------------------------------------------------------
# Run the survey under each tool stack
# ----------------------------------------------------------------------
TOOLS = ("ISCWSA MWD Rev5.11", "GYRO-NS", "GYRO-NS-CT")

print(f"\nBuilding Survey with error_model = ...")
results: dict[str, we.survey.Survey] = {}
for tool in TOOLS:
    print(f"  {tool}")
    # Catch the gyro RuntimeWarnings for un-evaluable terms; we'll
    # report them once at the end rather than spam the loop.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        results[tool] = make_survey(tool)
        if caught:
            for w in caught:
                print(f"    ! warning: {w.message}")


# ----------------------------------------------------------------------
# Compare cov_nev at TD across the three models
# ----------------------------------------------------------------------
print("\n" + "=" * 78)
print("Position uncertainty (1-sigma, m) per axis at total depth")
print("=" * 78)
print(f"{'tool':<26s}  {'sigma_N':>9s}  {'sigma_E':>9s}  {'sigma_V':>9s}  "
      f"{'sigma_max':>10s}")
for tool, survey in results.items():
    cov = survey.cov_nev[-1]
    sN, sE, sV = np.sqrt(np.diag(cov))
    sigma_max = np.sqrt(np.linalg.eigvalsh(cov).max())
    print(f"{tool:<26s}  {sN:>9.3f}  {sE:>9.3f}  {sV:>9.3f}  {sigma_max:>10.3f}")

print("\n" + "=" * 78)
print("Notes on the comparison")
print("=" * 78)
print("""\
- MWD uses magnetic-field measurement of azimuth -- subject to magnetic
  declination uncertainty, BGGM/IFR field-model error, and crustal
  anomalies. The horizontal sigma split between N and E depends on
  the well's trending azimuth: declination errors project as cross-
  track uncertainty, so a near-northerly well sees those errors
  primarily on sigma_E (as in the numbers above for this 14 deg
  trajectory).

- GYRO-NS gyrocompasses (measures Earth's rotation) at every station,
  bypassing the magnetic environment entirely. At low latitudes the
  gyrocompass takes longer to converge; at high latitudes the
  Earth-rate signal is weaker. North-Sea conditions (lat 60) sit in
  the sweet spot.

- GYRO-NS-CT runs the gyro continuously between gyrocompass stops,
  integrating the gyro rates to interpolate azimuth between fixes.
  Accumulates gyro-drift / random-walk error during the continuous
  segments, but is much faster operationally than stopping for a full
  gyrocompass at every station.

- 'sigma_max' is the worst-case 1-sigma direction (the largest semi-axis
  of the position-uncertainty ellipsoid). For anti-collision purposes
  this is the direction you actually care about.
""")
