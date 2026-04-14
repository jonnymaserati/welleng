import json
import unittest

import numpy as np
import pandas as pd

from welleng.survey import Survey, SurveyHeader
from welleng.error import ErrorModel
from welleng.utils import get_sigmas

"""
Test that the ISCWSA MWD error models reproduce the example workbook values.

Each model carries its own ``(rtol, atol)`` pair because the reference JSON
files have different precision:

- Rev 5.11 is regenerated from the Rev 5.11 workbook's full-precision
  Calculated block, so welleng matches it to ~1e-5 relative / 1e-6 absolute.
- Rev 4's JSON is extracted from the workbook's Diagnostic block which
  rounds to 4 decimal places. Cells whose reference value is itself
  ~1e-4 produce artificially large *relative* errors, so an absolute
  tolerance of 5e-5 covers the rounding floor while the relative tolerance
  catches real drift on bigger values.
"""

# Pass condition per cell: ``abs_err <= atol  OR  rel_err <= rtol``.
MODEL_TOLERANCES = {
    "ISCWSA MWD Rev4":    {"rtol": 1e-3, "atol": 5e-5},  # 0.1% / rounding-floor
    "ISCWSA MWD Rev5.11": {"rtol": 5e-5, "atol": 1e-6},  # atol tightened 50x
}

input_files = {
    "ISCWSA MWD Rev4": "tests/test_data/error_mwdrev4_1_iscwsa_data.json",
    "ISCWSA MWD Rev5.11": "tests/test_data/error_mwdrev5_1_iscwsa_data.json"
}


def initiate(error_model, filename):
    # Read validation data from file:
    wd = json.load(open(filename))
    err = get_err(error_model, wd)
    df = pd.DataFrame(wd['vd'])

    return (df, err)


def get_md_index(error_data, md):
    i = np.where(error_data.survey.md == md)[0][0]
    return i


def get_err(error_model, wd):
    sh = SurveyHeader()

    for k, v in wd['header'].items():
        setattr(sh, k, v)

    survey = Survey(
        md=wd['survey']['md'],
        inc=wd['survey']['inc'],
        azi=wd['survey']['azi'],
        header=sh,
        error_model=error_model
    )

    err = survey.err

    return err


def test_iscwsa_error_models(input_files=input_files):
    for error_model, filename in input_files.items():
        df, err = initiate(error_model, filename)

        data = [[], [], [], [], [], []]
        for _, row in df.iterrows():
            i = get_md_index(err, row['md'])
            s = row['source']
            if s in ["Totals", "TOTAL"]:
                source_cov = err.errors.cov_NEVs[i]
            else:
                source_cov = err.errors.errors[s].cov_NEV[i]
            v = get_sigmas(source_cov, long=True)
            for j, d in enumerate(v):
                data[j].append(d[0])

        computed = np.array(data).T
        reference = np.array(df.iloc[:, 2:], dtype=float)

        tol = MODEL_TOLERANCES[error_model]
        abs_err = np.abs(computed - reference)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.nan_to_num(abs_err / np.abs(reference))
        passing = (abs_err <= tol['atol']) | (rel_err <= tol['rtol'])

        if not np.all(passing):
            # Surface the worst few offending cells for debugging.
            bad = np.argwhere(~passing)
            labels = ('NN', 'EE', 'VV', 'NE', 'NV', 'EV')
            msgs = [
                f"    md={df.md.iloc[r]} src={df.source.iloc[r]} "
                f"{labels[c]}: ref={reference[r, c]:.6g} "
                f"got={computed[r, c]:.6g} abs={abs_err[r, c]:.3e} "
                f"rel={rel_err[r, c]:.3e}"
                for r, c in bad[:8]
            ]
            raise AssertionError(
                f"{error_model}: {len(bad)} cells exceed tolerance "
                f"(rtol={tol['rtol']:g}, atol={tol['atol']:g}):\n"
                + "\n".join(msgs)
            )


def test_drdp_single_pass_matches_column_methods():
    """
    _drdp single-pass implementation must produce identical output to
    assembling the result from the 6 individual drk_d* column methods.
    """
    s = Survey(
        md=[0, 500, 1000, 2000, 3000],
        inc=[0, 15, 45, 60, 80],
        azi=[0, 30, 90, 150, 270],
        unit='meters',
    )
    for model in ('ISCWSA MWD Rev4', 'ISCWSA MWD Rev5.11'):
        em = ErrorModel(s, error_model=model)
        expected = np.hstack((
            em.drk_dDepth(em.survey_drdp),
            em.drk_dInc(em.survey_drdp),
            em.drk_dAz(em.survey_drdp),
            em.drkplus1_dDepth(em.survey_drdp),
            em.drkplus1_dInc(em.survey_drdp),
            em.drkplus1_dAz(em.survey_drdp),
        ))
        assert em.drdp.shape == (len(s.md), 18), f"{model}: unexpected shape"
        assert np.allclose(em.drdp, expected, atol=1e-14), (
            f"{model}: max diff {np.max(np.abs(em.drdp - expected))}"
        )
