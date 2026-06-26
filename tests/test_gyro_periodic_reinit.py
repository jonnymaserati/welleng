"""SPE 90408 App. C box-11 / min_D periodic re-initialisation (opt-in switch).

Box 11 of the gyro implementation flow chart re-initialises a continuous
survey every ``min_D`` along-hole, splitting an uninterrupted continuous run
into consecutive ~min_D sections each surveyed independently, to bound the
drift / random-walk growth (author's recommendation, Ekseth).

This feature is **opt-in, default OFF, and UNVALIDATED**: no SPE 90408
Appendix E test well exercises a genuine box-11 within-run re-init bounded by
the recommended min_D (the engine's validated path is unchanged), so there is
no published reference to check the ON path against. These tests therefore pin
the **mechanism and the safety rails**, not absolute covariance values:

  1. Default OFF is silent and byte-identical (no min_D in the tool params).
  2. OFF but a continuous section actually runs longer than min_D -> a one-shot
     advisory points at the switch; the covariance is UNCHANGED (warn-only).
  3. ON -> the experimental/unverifiable warning fires AND the continuous
     recurrence accumulation resets at ~min_D (a fresh ~min_D segment starts).
  4. ON but no section reaches min_D -> byte-identical to OFF (nothing to do).
  5. ON without a usable min_D -> a no-op warning; default output returned.

All warnings are warn-and-continue (RuntimeWarning); none is fatal. The
stationary-init carry is deliberately NOT touched by box 11 (its weights --
``Tan(Inc)``, ``1/Cos(Inc)`` -- diverge at high inclination, which is why the
validated carry freezes them at the gate; see
``ToolError._carry_per_section``). The finite box-11 effect lives in the
continuous recurrence reset (``ToolError._call_interpreter_recurrence``).
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

import welleng as we
from welleng.survey import Survey, SurveyHeader
from welleng.errors import tool_errors as te_mod
from welleng.errors.tool_errors import _json_to_em_adapter

GYRO_JSON = (Path(te_mod.__file__).parent
             / "iscwsa_json" / "owsg_a" / "GYRO-NS-CT.json")
GATE_RAD = 0.29670597283903605   # GYRO-NS-CT static-gyro end inc (= 17 deg)
MIN_D = 2500.0
STEP = 50.0


def _well(td_m: float) -> Survey:
    """Synthetic build-and-hold well: build to 30 deg by ~600 m (crossing the
    17 deg gate ~340 m), then hold 30 deg to ``td_m``. A single continuous
    section above the gate of ~(td_m - 340) m."""
    md = np.arange(0.0, td_m + 1.0, STEP)
    inc = np.clip(md / 20.0, 0.0, 30.0)
    azi = np.full_like(md, 45.0)
    sh = SurveyHeader(latitude=60.0, azi_reference="true")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return Survey(md=md, inc=inc, azi=azi, header=sh)


def _wire(survey: Survey, params_extra: dict):
    """Borrow a wired GYRO-NS-CT ToolError and swap in the tool model with
    ``params_extra`` merged into its ``parameters`` block (mirrors the
    Appendix-E test helper). ``_min_D`` is left for the caller to resolve so
    the resolve-time warning can be observed."""
    src = json.loads(GYRO_JSON.read_text())
    src.setdefault("parameters", {}).update(params_extra)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        em = we.error.ErrorModel(survey, error_model="GYRO-NS-CT")
    te = em.errors
    te.em = _json_to_em_adapter(src)
    return te


def _run(te):
    """Resolve the periodic-reinit config (may warn) then run every term;
    return (summed cov_NEV, GXY-GD per-station azimuth weight e_DIA[:, 2])."""
    te._min_D = te._resolve_periodic_reinit()
    n = len(te.e.survey.md)
    cov = np.zeros((n, 3, 3))
    e_gd = None
    for code, entry in te.em["codes"].items():
        err = te._call_interpreter(
            code, entry["_iscwsa_term"], entry["magnitude"], entry["propagation"])
        cov += np.asarray(err.cov_NEV)
        if code == "GXY-GD":
            e_gd = np.asarray(err.e_DIA)[:, 2]
    return cov, e_gd


def test_default_off_is_silent():
    """No min_D in the tool params -> the feature is inert and silent."""
    survey = _well(4000.0)
    te = _wire(survey, {})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _run(te)
    assert te._min_D is None
    msgs = [str(w.message) for w in rec]
    assert not any("min_D" in m or "re-init" in m for m in msgs)


def test_off_rail_warns_and_is_byte_identical():
    """OFF + a section longer than min_D -> advisory pointing at the switch,
    but the covariance is byte-identical to the no-min_D default."""
    survey = _well(4000.0)
    cov_plain, _ = _run(_wire(survey, {}))

    te = _wire(survey, {"min_dist_between_initialisations": MIN_D})
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cov_rail, _ = _run(te)
    assert te._min_D is None    # OFF: feature NOT applied
    assert any("exceeds min_D" in str(w.message) for w in rec)
    np.testing.assert_array_equal(cov_plain, cov_rail)


def test_enabled_emits_experimental_warning():
    survey = _well(4000.0)
    te = _wire(survey, {"min_dist_between_initialisations": MIN_D,
                        "periodic_reinit": True})
    with pytest.warns(RuntimeWarning, match="results are unverifiable"):
        te._min_D = te._resolve_periodic_reinit()
    assert te._min_D == MIN_D


def test_enabled_without_min_D_is_noop():
    survey = _well(4000.0)
    te = _wire(survey, {"periodic_reinit": True})
    with pytest.warns(RuntimeWarning, match="no-op"):
        te._min_D = te._resolve_periodic_reinit()
    assert te._min_D is None


def test_recurrence_resets_at_min_D():
    """The continuous recurrence (GXY-GD drift) accumulates monotonically with
    the flag OFF, and RE-INITS (drops to zero, then re-accumulates) at ~min_D
    with the flag ON -- the box-11 mechanism."""
    survey = _well(4000.0)
    md = np.asarray(survey.md)
    inc = np.asarray(survey.inc_rad)
    above = np.where(inc > GATE_RAD)[0]
    assert above.size > 0 and md[above[-1]] - md[above[0] - 1] > MIN_D

    # OFF: monotone non-decreasing across the whole continuous section.
    te_off = _wire(survey, {})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, e_off = _run(te_off)
    assert te_off._min_D is None
    assert np.all(np.diff(e_off[above]) >= -1e-12)
    assert e_off[above[-1]] > 0.0

    # ON: a single re-init resets the accumulation to zero ~min_D in.
    te_on = _wire(survey, {"min_dist_between_initialisations": MIN_D,
                           "periodic_reinit": True})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, e_on = _run(te_on)
    assert te_on._min_D == MIN_D
    assert np.all(np.isfinite(e_on))

    drops = [k for k in above[1:] if e_on[k] == 0.0 and e_on[k - 1] > 0.0]
    assert drops, "box-11 re-init did not reset the recurrence accumulation"
    reinit_k = drops[0]
    # the reset lands within one survey step of min_D past the section start
    seg_start = md[above[0] - 1]
    assert MIN_D <= md[reinit_k] - seg_start < MIN_D + STEP
    # a fresh segment then accumulates again
    nxt = reinit_k + 1
    assert nxt in above and e_on[nxt] > 0.0
    # before the reset the ON and OFF curves agree (same accumulation)
    assert np.allclose(e_on[above[0]:reinit_k], e_off[above[0]:reinit_k])


def test_no_long_section_unchanged_when_enabled():
    """Enabling the feature on a well whose continuous section is SHORTER than
    min_D changes nothing (box 11 has no boundary to fire at)."""
    survey = _well(1500.0)        # section above the gate ~1160 m < min_D
    cov_off, _ = _run(_wire(survey, {}))
    te_on = _wire(survey, {"min_dist_between_initialisations": MIN_D,
                           "periodic_reinit": True})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cov_on, _ = _run(te_on)
    assert te_on._min_D == MIN_D       # feature enabled ...
    np.testing.assert_array_equal(cov_off, cov_on)   # ... but byte-identical
