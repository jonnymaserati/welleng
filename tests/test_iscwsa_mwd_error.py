import json
import unittest

import numpy as np
import pandas as pd

from welleng.survey import Survey, SurveyHeader
from welleng.utils import get_sigmas

"""
Test that the ISCWSA MWD Rev5 error model is working within a defined
tolerance (the default has been set to 0.001%), testing against the
MWD Rev 5 error model example provided by ISCWSA.
"""

# Set test tolerance as percentage
TOLERANCE = 0.001
# input_files = {
#     "iscwsa_mwd_rev4": "tests/test_data/error_mwdrev4_1_iscwsa_data.json",
#     "iscwsa_mwd_rev5": "tests/test_data/error_mwdrev5_1_iscwsa_data.json"
# }
input_files = {
    "ISCWSA MWD Rev4": "tests/test_data/error_mwdrev4_1_iscwsa_data.json",
    "ISCWSA MWD Rev5": "tests/test_data/error_mwdrev5_1_iscwsa_data.json"
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

        nn_c, ee_c, vv_c, ne_c, nv_c, ev_c = [], [], [], [], [], []
        data = [
            nn_c, ee_c, vv_c, ne_c, nv_c, ev_c
        ]

        # generate error data
        for index, row in df.iterrows():
            i = get_md_index(err, row['md'])
            s = row['source']
            if s in ["Totals", "TOTAL"]:
                source_cov = err.errors.cov_NEVs.T[i]
            else:
                source_cov = err.errors.errors[s].cov_NEV.T[i]
            v = get_sigmas(source_cov, long=True)
            for j, d in enumerate(v):
                data[j].append(d[0])

        # convert to dictionary
        ed = {}
        headers = [
            'nn_c', 'ee_c', 'vv_c', 'ne_c', 'nv_c', 'ev_c'
        ]
        for i, h in enumerate(headers):
            ed[h] = data[i]

        df_c = pd.DataFrame(ed)

        df_r = df.join(df_c)

        headers = [
            'nn_d', 'ee_d', 'vv_d', 'ne_d', 'nv_d', 'ev_d'
        ]
        df_d = pd.DataFrame(
            np.around(
                np.array(df_c) - np.array(df.iloc[:, 2:]),
                decimals=4
            ),
            columns=headers
        )

        df_r = df_r.join(df_d)

        with np.errstate(divide='ignore', invalid='ignore'):
            error = np.nan_to_num(np.absolute(
                np.array(df_d) / np.array(df.iloc[:, 2:])
                ) * 100
            )

        assert np.all(error < TOLERANCE), (
            f"failing error {d}"
        )

        # if you wanted to view the results, this would save then to an Excel
        # file.
        # df_r.to_excel(
        #     "tests/test_data/error_mwdrev5_iscwsa_validation_results.xlsx"
        # )
