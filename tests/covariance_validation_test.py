import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd

from welleng.error_formula_extractor.enums import Propagation, VectorType
from welleng.error_formula_extractor.formula_utils import function_builder
from welleng.error_formula_extractor.models import ErrorTerm, SurveyToolErrorModel
from welleng.survey import Survey, SurveyHeader
from welleng.units import TORTUOSITY_RAD_PER_M
from welleng.utils import errors_from_cov

warnings.filterwarnings("ignore")

var_map = {
    "Inc-IncPrev": "din",
    "MD-MDPrev": "dmd",
    "AzT-AzPrev": "dazt",
    "MD": "tmd",
    "TVD": "tvd",
    "Inc": "inc",
    "Gfield": "gtot",
    "Dip": "dip",
    "BField": "mtot",
    "AzT": "azt",
    "AzM": "azm",
}

survey_unit = {
    'error-model-example-mwdrev5-1-iscwsa-1.xlsx': 'meter',
    'error-model-example-mwdrev5-1-iscwsa-2.xlsx': 'ft',
    'error-model-example-mwdrev5-1-iscwsa-3.xlsx': 'meter',
}

# the following list of folder are needed to keep the generated files
# separate which is later is used to compare the generated files with the expected files.
# check inside the directory to see if there are following list of folders
folder_list = ['cov_per_error_test_1', 'cov_per_error_test_2', 'cov_per_error_test_3',
               'error_absdif_df_plots_test_1', 'error_absdif_df_plots_test_2', 'error_absdif_df_plots_test_3',
               'error_absdif_df_test_1', 'error_absdif_df_test_2', 'error_absdif_df_test_3',
               'error_figures_test_1', 'error_figures_test_2', 'error_figures_test_3']
# if folder is not exist then create it
for folder in folder_list:
    if not os.path.exists(folder):
        os.makedirs(folder)


def run():
    # Note that the ISCWSA tests 2 and 3 in this directory were edited for XCLL, XYM2 and XYM4E.
    # for more details please review the Excel file provided in the directory.
    filename = 'error-model-example-mwdrev5-1-iscwsa-3.xlsx'

    ISCWSA_cases = {
        'error-model-example-mwdrev5-1-iscwsa-1.xlsx': {
            'filename_to_save': "ISCWSA_code_ex1.csv",
            'path_to_save_cov_csv_files': 'cov_per_error_test_1',
        },
        'error-model-example-mwdrev5-1-iscwsa-2.xlsx': {
            'filename_to_save': "ISCWSA_code_ex2.csv",
            'path_to_save_cov_csv_files': 'cov_per_error_test_2'},

        'error-model-example-mwdrev5-1-iscwsa-3.xlsx': {
            'filename_to_save': "ISCWSA_code_ex3.csv",
            'path_to_save_cov_csv_files': 'cov_per_error_test_3'},
    }

    # Load the error model data from the ISCWSA test file
    dfs = pd.read_excel(
        filename,
        sheet_name="Model",
        usecols="D:W",
        header=2
    )

    # map the column names from the df to the column names needed for the code
    desired_cols = {
        "No": "sequence_no",
        'Code': "term_name",
        'Units': "c_units",
        'Prop.': "tie_type",
        'Convert Magnitudes Degrees to Radians': "c_value",
        'Depth Formula': 'Depth Formula',
        'Inclination Formula': 'Inclination Formula',
        'Azimuth Formula': 'Azimuth Formula',
        'Singularity North Formula': 'Singularity North Formula',
        'Singularity East Formula': 'Singularity East Formula',
        'Singularity Vert. Formula': 'Singularity Vert. Formula'
    }

    # filter the df to just keep the columns specified in the dict above and change the column names
    df = dfs[desired_cols.keys()]
    df.columns = desired_cols.values()

    formula_cols = ['Depth Formula', 'Inclination Formula', 'Azimuth Formula']

    # for the formula columns, replace the variable names using the replace str function
    for col in formula_cols:
        df[col] = df[col].apply(replace_str)

    # for each error term in the df, create the list of ErrorTerm class
    error_terms = {}
    new_sequence_no = 0
    for index, row in df.iterrows():
        error_terms, new_sequence_no = convert_error_terms(row, error_terms, new_sequence_no)

    # creat the SurveyToolErrorModel class object using the error terms
    survey_tool = SurveyToolErrorModel(
        survey_tool_id="none",
        survey_tool_name="A001Mc",
        sequence_no=0,
        error_terms=list(error_terms.values()),
        start_depth=0,
        end_depth=8000
    )

    # load the survey from the excel sheet and store it in the Survey object
    df_survey = pd.read_excel(
        filename,
        sheet_name="Wellpath",
        usecols="E:I",
        header=1
    )
    df_survey.drop(["Toolface (°)", "TVD "], axis=1, inplace=True)
    df_survey.columns = ["measured_depth", "inclination", "azimuth"]

    df_survey_header = pd.read_excel(
        filename,
        sheet_name="Wellpath",
        usecols="A:C",
        header=0
    )
    df_survey_header = df_survey_header[0:10]
    df_survey_header.columns = ["parameter", "value", "value in radians"]
    df_sh = df_survey_header.T
    df_sh.columns = list(df_sh.loc["parameter"])
    df_sh.drop(["parameter"], axis=0, inplace=True)

    val_dict = {}
    for col in df_sh.columns:
        if df_sh[col]["value in radians"] is np.nan:
            val = df_sh[col]["value"]
        else:
            val = df_sh[col]["value in radians"]

        val_dict.update({col: val})

    sh = SurveyHeader(
        latitude=val_dict['Latitude (deg)'],
        G=val_dict['G (m/s2)'],
        b_total=val_dict['Btotal (nT)'],
        earth_rate=0.26251614,
        dip=val_dict['Dip (deg)'],
        declination=val_dict['Declination (deg)'],
        convergence=val_dict['Convergence (deg)'],
        azi_reference="true",
        vertical_inc_limit=val_dict['Vertical Inc Limit (deg)'],
        deg=False,
        depth_unit='meters',
        surface_unit='meters'
    )

    if survey_unit[filename] == 'ft':
        survey_unit_conversion_ft_meter = 0.3048
    else:
        survey_unit_conversion_ft_meter = 1

    iscwsa_survey = Survey(
        md=df_survey["measured_depth"].values * survey_unit_conversion_ft_meter,
        inc=df_survey["inclination"].values,
        azi=df_survey["azimuth"].values,
        header=sh,
    )

    # calculate covariance for the Survey object
    iscwsa_survey.get_error(edm_error_model=survey_tool)

    # get the covariances for each individual error term and save them in a csv file
    for key, value in iscwsa_survey.err.errors.errors.items():
        # vstack the covariances
        reshaped_cov_nev = np.vstack(value.cov_NEV).T
        columns = ['nn', 'ne', 'nv', 'en', 'ee', 'ev', 'vn', 've', 'vv']
        df = pd.DataFrame(reshaped_cov_nev, columns=columns)
        keep_cols = ['nn', 'ee', 'vv', 'ne', 'nv', 'ev']
        df = df[keep_cols]
        df.to_csv(f"{ISCWSA_cases[filename]['path_to_save_cov_csv_files']}/{key}.csv")

    cov_nevs = []
    cov_nevs.append(iscwsa_survey.cov_nev)

    azimuth_reference = iscwsa_survey.azi_true_deg
    result_with_covariance = construct_results_dict(iscwsa_survey, azimuth_reference, cov_nevs)

    df_output = pd.DataFrame(result_with_covariance)
    cov_code = pd.DataFrame.from_records(df_output["covariance"])
    data_code_combined = pd.concat([df_output, cov_code], axis=1)
    data_code_combined.drop(["covariance"], axis=1, inplace=True)
    columns = ['measured_depth', 'inclination', 'azimuth', 'northing', 'easting',
               'tvd', 'nn', 'ee', 'vv', 'ne', 'nv', 'ev']
    data_code_combined = data_code_combined[columns]

    # data_code_combined.to_csv(filename_to_save[filename])
    data_code_combined.to_csv(ISCWSA_cases[filename]['filename_to_save'])


def replace_str(formula_str: str) -> str:
    formula_str = str(formula_str)
    for search, replace in var_map.items():
        formula_str = formula_str.replace(search, replace)

    if formula_str == "0":
        return None
    return formula_str.lower()


def extract_tie_type(code: dict) -> str:
    if code["tie_type"].lower() == 'r':
        return Propagation.RANDOM
    elif code["tie_type"].lower() == "s":
        return Propagation.SYSTEMATIC
    elif code["tie_type"].lower() == "g":
        return Propagation.GLOBAL
    elif code["tie_type"].lower() == "n":
        return Propagation.NA

    return Propagation.SYSTEMATIC


def create_error_class(row, formula: str, vector: VectorType) -> ErrorTerm:
    tie_type = extract_tie_type(row)

    return ErrorTerm(
        sequence_no=row["sequence_no"],
        term_name=row["term_name"],
        formula=[formula],
        error_function=[(result := function_builder(formula, row["term_name"].lower().replace("-", "_")))[0]],
        magnitude=[row["c_value"]],
        units=[row["c_units"]],
        tie_type=[tie_type],
        vector_type=[vector],
        arguments=[result[1]],
        func_string=[result[2]]
    )


def convert_error_terms(row: pd.Series, error_terms: dict, sequence_no: int) -> Tuple[dict, int]:
    term_name = row["term_name"]
    if row["Depth Formula"]:
        formula = row["Depth Formula"]
        if "0.0328084" in formula:
            formula = formula.replace("0.0328084", str(TORTUOSITY_RAD_PER_M))
        vector = VectorType.DEPTH_TERMS
        error_terms[term_name] = create_error_class(row, formula, vector)

    if row["Inclination Formula"]:
        formula = row["Inclination Formula"]
        if "0.0328084" in formula:
            formula = formula.replace("0.0328084", str(TORTUOSITY_RAD_PER_M))
        vector = VectorType.INCLINATION_TERMS
        error_terms[term_name] = create_error_class(row, formula, vector)

    if row["Azimuth Formula"]:
        formula = row["Azimuth Formula"]
        if "0.0328084" in formula:
            formula = formula.replace("0.0328084", str(TORTUOSITY_RAD_PER_M))
        if type(row['Singularity North Formula']) == str or type(row['Singularity East Formula']) == str:
            vector = VectorType.LATERAL
        else:
            vector = VectorType.AZIMUTH_TERMS

        if term_name in error_terms.keys():
            tie_type = extract_tie_type(row)
            formula = formula.replace("sqr(", "sqrt(")

            error_terms[term_name].formula.append(formula)
            error_terms[term_name].error_function.append(
                (result := function_builder(formula, row["term_name"].lower().replace("-", "_")))[0]
            )
            error_terms[term_name].magnitude.append(row["c_value"])
            error_terms[term_name].units.append(row["c_units"])
            error_terms[term_name].vector_type.append(vector)
            error_terms[term_name].tie_type.append(tie_type)
            error_terms[term_name].arguments.append(result[1])
            error_terms[term_name].func_string.append(result[2])
        else:
            error_terms[term_name] = create_error_class(row, formula, vector)

    error_terms[term_name].sequence_no = sequence_no + 1
    return error_terms, error_terms[term_name].sequence_no


def construct_results_dict(
        survey: Survey,
        azimuth: np.ndarray,
        cov_nev: np.ndarray = None
) -> list:
    """
    This function constructs the output results dictionary.
    :param survey:
    :param azimuth:
    :param cov_nev:
    :return:
    """
    if cov_nev:
        stacked_cov_nev = np.round(errors_from_cov(
            np.vstack(cov_nev)
        ), 5)
    else:
        stacked_cov_nev = np.zeros((len(survey.md), 6))

    return [
        {
            'measured_depth': md,
            'inclination': inc,
            'azimuth': azi,
            'northing': n,
            'easting': e,
            'tvd': tvd,
            "covariance": {
                "nn": nn,
                'ne': ne,
                'nv': nv,
                'ee': ee,
                'ev': ev,
                'vv': vv
            }
        }
        for md, inc, azi, n, e, tvd, (nn, ne, nv, ee, ev, vv) in zip(
            np.round(survey.md, 3),
            np.round(survey.inc_deg, 3),
            np.round(azimuth, 3),
            np.round(survey.n, 3),
            np.round(survey.e, 3),
            np.round(survey.tvd, 3),
            stacked_cov_nev
        )
    ]


# make above test run separately
if __name__ == '__main__':
    run()
