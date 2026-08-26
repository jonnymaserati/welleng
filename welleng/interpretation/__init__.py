"""Directional-survey interpretation: raw sensors -> survey, QC and multi-station
analysis (MSA).

This is welleng's survey-*interpretation* layer, upstream of the forward error
model (which maps inc/azi -> ISCWSA covariance). It is the open, scalar reference
implementation:

- :func:`~welleng.interpretation.forward.sensor_to_survey` -- accelerometer +
  magnetometer -> inclination, azimuth, toolface (public-domain MWD navigation
  equations; Williamson, SPE-67616).
- :func:`~welleng.interpretation.forward.gyro_to_survey` -- accelerometer +
  rate-gyro -> inclination, azimuth (true north, no declination), toolface
  (gyrocompassing). The gyro reference is Earth's rotation rate -- a constant,
  with local components a closed-form function of latitude
  (:func:`~welleng.interpretation.forward.earth_rate_components`).
- :func:`~welleng.interpretation.qc.georef_checks` -- georeference QC tests
  (total-gravity, total-field, dip) with pass/fail flags (Ekseth et al.,
  SPE-133417).
- :func:`~welleng.interpretation.qc.dual_depth_difference` -- dual-depth-
  difference depth QC test (pipe tally vs wireline), SPE-133417.
- :func:`~welleng.interpretation.msa.estimate_sensor_errors` -- multi-station
  analysis: closed-form linear least-squares estimate of the tool's actual
  sensor biases/scale-factors, with an *analytical* estimability (correlation)
  matrix that gates unreliable (poorly-observed) components (SPE-133417,
  multistation test).

References
----------
Williamson, H.S. (2000) "Accuracy Prediction for Directional MWD", SPE-67616.
Ekseth, R. et al. (2010) "High-Integrity Wellbore Surveying", SPE-133417.
Grindrod, S.J. et al. (2016) OWSG survey-tool error models, IADC/SPE-178843.
"""
from .forward import (
    sensor_to_survey,
    gyro_to_survey,
    earth_rate_components,
    EARTH_RATE,
)
from .qc import (
    GeomagReference,
    QCResult,
    georef_checks,
    DualDepthResult,
    dual_depth_difference,
)
from .msa import MSAResult, estimate_sensor_errors, apply_sensor_errors
from .correction_uncertainty import (
    CorrectionUncertainty,
    correction_covariance_mc,
)

__all__ = [
    "CorrectionUncertainty",
    "correction_covariance_mc",
    "sensor_to_survey",
    "gyro_to_survey",
    "earth_rate_components",
    "EARTH_RATE",
    "GeomagReference",
    "QCResult",
    "georef_checks",
    "DualDepthResult",
    "dual_depth_difference",
    "MSAResult",
    "estimate_sensor_errors",
    "apply_sensor_errors",
]
