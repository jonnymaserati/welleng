import json
import numpy as np
import pandas as pd
from welleng.utils import get_sigmas
from welleng.survey import Survey, SurveyHeader

"""
Test that the ISCWSA MWD Rev4 error model is working within a defined
tolerance (the default has been set to 0.001%), testing against the
MWD Rev 4 error model example provided by ISCWSA.
"""

# Set test tolerance as percentage
TOLERANCE = 0.001

# Read validation data from file:
wd = json.load(open("test/test_data/error_mwdrev4_iscwsa_well_data.json"))
vd = json.load(open("test/test_data/error_mwdrev4_iscwsa_validation.json"))

df = pd.DataFrame(vd)


def get_md_index(error_data, md):
    i = np.where(error_data.survey.md == md)[0][0]
    return i


sh = SurveyHeader()

for k, v in wd['header'].items():
    setattr(sh, k, v)


survey = Survey(
    md=wd['survey']['md'],
    inc=wd['survey']['inc'],
    azi=wd['survey']['azi'],
    header=sh,
    error_model="iscwsa_mwd_rev4"
)

err = survey.err


# initiate lists
def test_error_model_mwdrev4_iscwsa(df=df, err=err):
    nn_c, ee_c, vv_c, ne_c, nv_c, ev_c = [], [], [], [], [], []
    data = [
        nn_c, ee_c, vv_c, ne_c, nv_c, ev_c
    ]

    # generate error data
    for index, row in df.iterrows():
        i = get_md_index(err, row['md'])
        s = row['source']
        if s == "Totals":
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

    assert np.all(error < TOLERANCE)

    # if you wanted to view the results, this would save then to an Excel
    # file.
    # df_r.to_excel(
    #     "test/test_data/error_mwdrev4_iscwsa_validation_results.xlsx"
    # )


# make above test runnanble separately
if __name__ == '__main__':
    test_error_model_mwdrev4_iscwsa(df=df, err=err)
