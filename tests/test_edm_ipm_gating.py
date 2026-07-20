"""COMPASS IPM inclination-window gating (min_range/max_range).

COMPASS DP_TOOL_TERM rows may carry an inclination window (deg) outside
which the term is inactive -- e.g. mutually-exclusive continuous-gyro mode
terms. The converter previously emitted these as a dead ``inc_range`` key
nothing consumed, silently disabling the gating; it now emits the
``inc_min_deg``/``inc_max_deg`` keys the evaluation engine's per-term
gating reads (window semantics: zero contribution outside the window).
"""
import warnings

import numpy as np

import welleng as we
from welleng.errors.edm_ipm import IPMTerm, IPMTool, ipm_to_error_model


def _tool(min_inc, max_inc):
    return IPMTool(tool_id="T1", name="test", terms=[
        IPMTerm(name="gated", sequence_no=1, vector_type="i", tie_type="s",
                value=0.5, units="d", formula="1",
                min_inc=min_inc, max_inc=max_inc),
    ])


def _sigma(model):
    md = np.arange(0, 3001, 30.0)
    inc = np.clip((md - 300) / 30.0, 0, 60)   # crosses 15 deg at 750 m
    azi = np.full_like(md, 45.0)
    sh = we.survey.SurveyHeader(
        name="t", latitude=53.0, longitude=4.0, b_total=49500.0, dip=67.0,
        declination=1.5, azi_reference="true",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = we.survey.Survey(
            md=md, inc=inc, azi=azi, header=sh, error_model=model
        )
    cov = np.asarray(s.cov_nev)
    return md, inc, np.sqrt(cov[:, 0, 0] + cov[:, 1, 1] + cov[:, 2, 2])


def test_converter_emits_engine_gating_keys():
    model = ipm_to_error_model(_tool(0.0, 15.0))
    term = model["terms"][0]
    assert term["inc_min_deg"] == 0.0
    assert term["inc_max_deg"] == 15.0


def test_no_window_emits_no_gating_keys():
    model = ipm_to_error_model(_tool(0.0, 0.0))
    term = model["terms"][0]
    assert "inc_min_deg" not in term and "inc_max_deg" not in term


def test_windowed_term_stops_accumulating_above_gate():
    md, inc, sig = _sigma(ipm_to_error_model(_tool(0.0, 15.0)))
    i_gate = int(np.argmin(np.abs(inc - 15.0)))
    # active inside the window
    assert sig[i_gate] > 0.1
    # no new contribution above it: sigma stays flat to TD
    assert np.isclose(sig[-1], sig[i_gate + 2], rtol=1e-2)
    # sanity: ungated same term keeps growing to TD
    _, _, sig_open = _sigma(ipm_to_error_model(_tool(0.0, 0.0)))
    assert sig_open[-1] > 2.0 * sig[-1]
