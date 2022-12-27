import os
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")
# get the Path of current file
Path = os.path.dirname(os.path.abspath(__file__))
# print(Path)


# RUN for following
# read excel file
ISCWSA_file_name = 'error-model-example-mwdrev5-1-iscwsa-3.xlsx'

ISCWSA_case = {
    'error-model-example-mwdrev5-1-iscwsa-1.xlsx': {
        'folder_csv': 'cov_per_error_test_1',
        'folder_to_save_fig': 'error_figures_test_1',
        'sum_of_error_file_name_csv': 'sum_of_error_test_1.csv',
        'error_absdif_df_folder': 'error_absdif_df_test_1',
        'error_absdif_df_folder_for_plots': 'error_absdif_df_plots_test_1',
    },
    'error-model-example-mwdrev5-1-iscwsa-2.xlsx': {
        'folder_csv': 'cov_per_error_test_2',
        'folder_to_save_fig': 'error_figures_test_2',
        'sum_of_error_file_name_csv': 'sum_of_error_test_2.csv',
        'error_absdif_df_folder': 'error_absdif_df_test_2',
        'error_absdif_df_folder_for_plots': 'error_absdif_df_plots_test_2',
    },
    'error-model-example-mwdrev5-1-iscwsa-3.xlsx': {
        'folder_csv': 'cov_per_error_test_3',
        'folder_to_save_fig': 'error_figures_test_3',
        'sum_of_error_file_name_csv': 'sum_of_error_test_3.csv',
        'error_absdif_df_folder': 'error_absdif_df_test_3',
        'error_absdif_df_folder_for_plots': 'error_absdif_df_plots_test_3',
    },
}

list_of_errors_csv = os.listdir(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_csv'])
print('list_of_errors_csv ', len(list_of_errors_csv))

# get the error csv file and sum them up and save as a new csv file
# read the csv file
columns = ['nn', 'ee', 'vv', 'ne', 'nv', 'ev']
# make a empty pandas dataframe with columns
sum_of_errors = pd.DataFrame(columns=columns)
for file in list_of_errors_csv:
    # print file name
    # print(file)
    # read the csv file
    df = pd.read_csv(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_csv'] + '/' + file)
    # sum up row by row to get the total error just for columns nn, ee, vv, ne, nv, ev
    sum_of_errors = sum_of_errors.add(df[columns], fill_value=0)

# print(sum_of_errors)
# save the sum of errors to csv file
sum_of_errors.to_csv(Path + '/' + ISCWSA_case[ISCWSA_file_name]['sum_of_error_file_name_csv'], index=False)

# read excel file
# ISCWSA_file_name = 'error-model-example-mwdrev5-1-iscwsa-1.xlsx'
df = pd.read_excel(ISCWSA_file_name)

jump = False
for tab in list_of_errors_csv:
    if jump:
        continue
    tab = tab.replace('.csv', '')

    # there is no XCLL tab in for test 2 and 3
    # replace it with XCLA
    if tab == 'XCLL':
        tab = 'XCLA'

    ISCWSA_cov_nev = pd.read_excel(
        ISCWSA_file_name,
        sheet_name=tab,
        usecols="Q:V",
        header=1
    )

    # drop the last row of ISCWSA_cov_nev
    ISCWSA_cov_nev.drop(ISCWSA_cov_nev.tail(1).index, inplace=True)
    # check if in the column name there is string 1 if so remove it
    ISCWSA_cov_nev.columns = [col.replace('.1', '') for col in ISCWSA_cov_nev.columns]

    # and return the tab name to XCLL
    # in the cov_per_error_test_2 there is XCLL csv file
    if tab == 'XCLA':
        tab = 'XCLL'

    # read the data generated by the corva-welleng for each error
    corva_welleng_cov_nev = pd.read_csv(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_csv'] + '/' + tab + '.csv')
    # drop the first column of corva_welleng_cov_nev
    corva_welleng_cov_nev.drop(corva_welleng_cov_nev.columns[0], axis=1, inplace=True)
    # rename the columns of corva_welleng_cov_nev
    corva_welleng_cov_nev.columns = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV']

    # get the top N rows of ISCWSA_cov_nev. N is equal to the size of corva_welleng_cov_nev
    # in the ISCWSA_cov_nev file there are some rows that should be deleted from the bottom manually
    # check the iscwsa 2 and 3 and go to error tabs
    ISCWSA_cov_nev = ISCWSA_cov_nev.head(corva_welleng_cov_nev.shape[0])

    # compute the error between ISCWSA_cov_nev and corva_welleng_cov_nev for each column
    diff = corva_welleng_cov_nev - ISCWSA_cov_nev
    # calculate abs diff
    abs_diff = abs(diff)
    error = diff / ISCWSA_cov_nev
    # calculate error percentage
    error_percentage = error * 100

    # print the shape of error_percentage for each tab
    print(tab, error_percentage.shape)
    # if tab == "XCLA":
    #     print(tab)
    #     print("")

    # a depth column for plotting the error
    # between 0 and 8000 increment by 30
    # depth = np.arange(0, 8030, 30)

    # plot scatter the error percentage for each column in one plot and use index as x axis
    fig, ax = plt.subplots()
    ax.scatter(error_percentage.index, error_percentage['NN'], label='NN')
    ax.scatter(error_percentage.index, error_percentage['EE'], label='EE')
    ax.scatter(error_percentage.index, error_percentage['VV'], label='VV')
    ax.scatter(error_percentage.index, error_percentage['NE'], label='NE')
    ax.scatter(error_percentage.index, error_percentage['NV'], label='NV')
    ax.scatter(error_percentage.index, error_percentage['EV'], label='EV')
    ax.set_xlabel('index')
    ax.set_ylabel('Error (%)')
    ax.set_title(tab)
    ax.legend()
    # save the plot in error figures folder
    plt.savefig(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_to_save_fig'] + '/' + tab + '.png')
    # plt.show()

    # concatenate the ISCWSA_cov_nev, abs diff and error percentage to one dataframe
    error_absdif_df = pd.concat([ISCWSA_cov_nev, abs_diff, error_percentage], axis=1)

    new_columns_name = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV',
                        'NN_abs_diff', 'EE_abs_diff', 'VV_abs_diff', 'NE_abs_diff', 'NV_abs_diff', 'EV_abs_diff',
                        'NN_error_%', 'EE_error_%', 'VV_error_%', 'NE_error_%', 'NV_error_%', 'EV_error_%']
    error_absdif_df.columns = new_columns_name

    error_absdif_df['NN_abs_diff_2'] = error_absdif_df['NN_abs_diff']
    # if NN > 200, then replace NN_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NN'] > 200, 'NN_abs_diff_2'] = np.nan
    error_absdif_df['NN_error_%_2'] = error_absdif_df['NN_error_%']
    # if NN < 200, then replace NN_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NN'] < 200, 'NN_error_%_2'] = np.nan

    error_absdif_df['EE_abs_diff_2'] = error_absdif_df['EE_abs_diff']
    # if EE > 200, then replace EE_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['EE'] > 200, 'EE_abs_diff_2'] = np.nan
    error_absdif_df['EE_error_%_2'] = error_absdif_df['EE_error_%']
    # if EE < 200, then replace EE_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['EE'] < 200, 'EE_error_%_2'] = np.nan

    error_absdif_df['VV_abs_diff_2'] = error_absdif_df['VV_abs_diff']
    # if VV > 200, then replace VV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['VV'] > 200, 'VV_abs_diff_2'] = np.nan
    error_absdif_df['VV_error_%_2'] = error_absdif_df['VV_error_%']
    # if VV < 200, then replace VV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['VV'] < 200, 'VV_error_%_2'] = np.nan

    error_absdif_df['NE_abs_diff_2'] = error_absdif_df['NE_abs_diff']
    # if NE > 200, then replace NE_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NE'] > 200, 'NE_abs_diff_2'] = np.nan
    error_absdif_df['NE_error_%_2'] = error_absdif_df['NE_error_%']
    # if NE < 200, then replace NE_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NE'] < 200, 'NE_error_%_2'] = np.nan

    error_absdif_df['NV_abs_diff_2'] = error_absdif_df['NV_abs_diff']
    # if NV > 200, then replace NV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NV'] > 200, 'NV_abs_diff_2'] = np.nan
    error_absdif_df['NV_error_%_2'] = error_absdif_df['NV_error_%']
    # if NV < 200, then replace NV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NV'] < 200, 'NV_error_%_2'] = np.nan

    error_absdif_df['EV_abs_diff_2'] = error_absdif_df['EV_abs_diff']
    # if EV > 200, then replace EV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['EV'] > 200, 'EV_abs_diff_2'] = np.nan
    error_absdif_df['EV_error_%_2'] = error_absdif_df['EV_error_%']
    # if EV < 200, then replace EV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['EV'] < 200, 'EV_error_%_2'] = np.nan

    # keep following columns
    cols = ['NN_abs_diff_2', 'EE_abs_diff_2', 'VV_abs_diff_2', 'NE_abs_diff_2', 'NV_abs_diff_2', 'EV_abs_diff_2',
            'NN_error_%_2', 'EE_error_%_2', 'VV_error_%_2', 'NE_error_%_2', 'NV_error_%_2', 'EV_error_%_2']
    error_absdif_df = error_absdif_df[cols]
    # save the error_absdif_df to csv file
    error_absdif_df.to_csv(ISCWSA_case[ISCWSA_file_name]['error_absdif_df_folder'] + '/' + tab + '.csv', index=False)

    # plot scatter the error percentage for each column in one plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # add fig title
    fig.suptitle('Abs Diff and Error percentage for ' + tab + '\n' + ISCWSA_file_name, fontsize=20)
    # first plot an scatter plot for absolute difference consider the length of the data
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NN_abs_diff_2'], label='NN_abs_diff_2')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['EE_abs_diff_2'], label='EE_abs_diff_2')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['VV_abs_diff_2'], label='VV_abs_diff_2')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NE_abs_diff_2'], label='NE_abs_diff_2')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NV_abs_diff_2'], label='NV_abs_diff_2')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['EV_abs_diff_2'], label='EV_abs_diff_2')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Absolute Difference')
    ax[0].legend()

    # first plot an scatter plot for absolute difference consider the length of the data
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NN_error_%_2'], label='NN_error_%_2')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['EE_error_%_2'], label='EE_error_%_2')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['VV_error_%_2'], label='VV_error_%_2')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NE_error_%_2'], label='NE_error_%_2')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NV_error_%_2'], label='NV_error_%_2')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['EV_error_%_2'], label='EV_error_%_2')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Error Percentage')
    ax[1].legend()

    # save the plot
    plt.savefig(ISCWSA_case[ISCWSA_file_name]['error_absdif_df_folder_for_plots'] + '/' + tab + '.png')

# calculate sum of errors percentage and plot
Calculate_sum_err = True
if Calculate_sum_err:
    ISCWSA_TOTAL_cov_nev = pd.read_excel(
        ISCWSA_file_name,
        sheet_name='TOTALS',
        usecols="B:G",
        header=1
    )

    # get the top N rows of ISCWSA_TOTAL_cov_nev. N is equal to the size of sum_of_errors
    # in the ISCWSA_cov_nev file there are some rows that should be deleted from the bottom manually
    # check the iscwsa 2 and 3 and go to error tabs
    ISCWSA_TOTAL_cov_nev = ISCWSA_TOTAL_cov_nev.head(sum_of_errors.shape[0])

    # compute the error between ISCWSA_cov_nev and corva_welleng_cov_nev for each column
    sum_of_errors.columns = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV']
    diff = sum_of_errors - ISCWSA_TOTAL_cov_nev
    # change the type of diff to float
    diff = diff.astype(float)
    # calculate abs diff
    abs_diff = abs(diff)
    error = diff / ISCWSA_TOTAL_cov_nev
    # depth = np.arange(0, 8030, 30)
    # calculate error percentage
    error_percentage = error * 100
    error_percentage.columns = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV']
    # plot scatter the error percentage for each column in one plot
    fig, ax = plt.subplots()
    ax.scatter(error_percentage.index, error_percentage['NN'], label='NN')
    ax.scatter(error_percentage.index, error_percentage['EE'], label='EE')
    ax.scatter(error_percentage.index, error_percentage['VV'], label='VV')
    ax.scatter(error_percentage.index, error_percentage['NE'], label='NE')
    ax.scatter(error_percentage.index, error_percentage['NV'], label='NV')
    ax.scatter(error_percentage.index, error_percentage['EV'], label='EV')
    ax.set_xlabel('index')
    ax.set_ylabel('Error (%)')
    ax.set_title("TOTALS")
    ax.legend()

    # save the plot in error figures folder
    # plt.savefig(Path + '/error_figures/' + "TOTALS" + '.png')
    plt.savefig(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_to_save_fig'] + '/' + "TOTALS" + '.png')
    # plt.show()

    # concatenate the ISCWSA_cov_nev, abs diff and error percentage to one dataframe
    error_absdif_df = pd.concat([ISCWSA_TOTAL_cov_nev, abs_diff, error_percentage], axis=1)

    new_columns_name = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV',
                        'NN_abs_diff', 'EE_abs_diff', 'VV_abs_diff', 'NE_abs_diff', 'NV_abs_diff', 'EV_abs_diff',
                        'NN_error_%', 'EE_error_%', 'VV_error_%', 'NE_error_%', 'NV_error_%', 'EV_error_%']
    error_absdif_df.columns = new_columns_name

    error_absdif_df['NN_abs_diff_2'] = error_absdif_df['NN_abs_diff']
    # if NN > 200, then replace NN_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NN'] > 200, 'NN_abs_diff_2'] = np.nan
    error_absdif_df['NN_error_%_2'] = error_absdif_df['NN_error_%']
    # if NN < 200, then replace NN_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NN'] < 200, 'NN_error_%_2'] = np.nan

    error_absdif_df['EE_abs_diff_2'] = error_absdif_df['EE_abs_diff']
    # if EE > 200, then replace EE_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['EE'] > 200, 'EE_abs_diff_2'] = np.nan
    error_absdif_df['EE_error_%_2'] = error_absdif_df['EE_error_%']
    # if EE < 200, then replace EE_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['EE'] < 200, 'EE_error_%_2'] = np.nan

    error_absdif_df['VV_abs_diff_2'] = error_absdif_df['VV_abs_diff']
    # if VV > 200, then replace VV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['VV'] > 200, 'VV_abs_diff_2'] = np.nan
    error_absdif_df['VV_error_%_2'] = error_absdif_df['VV_error_%']
    # if VV < 200, then replace VV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['VV'] < 200, 'VV_error_%_2'] = np.nan

    error_absdif_df['NE_abs_diff_2'] = error_absdif_df['NE_abs_diff']
    # if NE > 200, then replace NE_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NE'] > 200, 'NE_abs_diff_2'] = np.nan
    error_absdif_df['NE_error_%_2'] = error_absdif_df['NE_error_%']
    # if NE < 200, then replace NE_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NE'] < 200, 'NE_error_%_2'] = np.nan

    error_absdif_df['NV_abs_diff_2'] = error_absdif_df['NV_abs_diff']
    # if NV > 200, then replace NV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['NV'] > 200, 'NV_abs_diff_2'] = np.nan
    error_absdif_df['NV_error_%_2'] = error_absdif_df['NV_error_%']
    # if NV < 200, then replace NV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['NV'] < 200, 'NV_error_%_2'] = np.nan

    error_absdif_df['EV_abs_diff_2'] = error_absdif_df['EV_abs_diff']
    # if EV > 200, then replace EV_abs_diff_2 with NaN
    error_absdif_df.loc[error_absdif_df['EV'] > 200, 'EV_abs_diff_2'] = np.nan
    error_absdif_df['EV_error_%_2'] = error_absdif_df['EV_error_%']
    # if EV < 200, then replace EV_error_%_2 with NaN
    error_absdif_df.loc[error_absdif_df['EV'] < 200, 'EV_error_%_2'] = np.nan

    # plot scatter the error percentage for each column in one plot
    fig, ax = plt.subplots(1, 2, figsize=(20, 10))
    # add fig title
    fig.suptitle('Abs Diff and Error percentage TOTAL' + '\n' + ISCWSA_file_name, fontsize=20)
    # consider index as x-axis
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NN_abs_diff_2'], label='NN')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['EE_abs_diff_2'], label='EE')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['VV_abs_diff_2'], label='VV')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NE_abs_diff_2'], label='NE')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['NV_abs_diff_2'], label='NV')
    ax[0].scatter(error_absdif_df.index, error_absdif_df['EV_abs_diff_2'], label='EV')
    ax[0].set_xlabel('Index')
    ax[0].set_ylabel('Absolute difference')
    ax[0].set_title('Absolute difference')
    ax[0].legend()

    # consider index as x-axis
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NN_error_%_2'], label='NN')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['EE_error_%_2'], label='EE')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['VV_error_%_2'], label='VV')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NE_error_%_2'], label='NE')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['NV_error_%_2'], label='NV')
    ax[1].scatter(error_absdif_df.index, error_absdif_df['EV_error_%_2'], label='EV')
    ax[1].set_xlabel('Index')
    ax[1].set_ylabel('Error percentage')
    ax[1].set_title('Error percentage')
    ax[1].legend()

    # save the plot
    plt.savefig(ISCWSA_case[ISCWSA_file_name]['error_absdif_df_folder_for_plots'] + '/' + 'TOTAL' + '.png')

# import r2_score from sklearn.metrics
# from sklearn.metrics import r2_score

# plot the ISCWSA TOTALS cov NEV versus the sum of errors cov NEV
# on a 3x2 subplot
fig, axs = plt.subplots(3, 2)
fig.suptitle('Cov NEV')

columns = ['NN', 'EE', 'VV', 'NE', 'NV', 'EV']

sum_of_errors = sum_of_errors.astype(float)
ISCWSA_TOTAL_cov_nev = ISCWSA_TOTAL_cov_nev.astype(float)

# plot using loop for each column
for i in range(3):
    for j in range(2):
        axs[i, j].scatter(ISCWSA_TOTAL_cov_nev[columns[i * 2 + j]], sum_of_errors[columns[i * 2 + j]])
        axs[i, j].set_xlabel('ISCWSA TOTALS')
        axs[i, j].set_ylabel('Corva-welleng')
        axs[i, j].set_title(columns[i * 2 + j])

        # fit a line to the data
        z = np.polyfit(ISCWSA_TOTAL_cov_nev[columns[i * 2 + j]],
                       sum_of_errors[columns[i * 2 + j]], 1)
        p = np.poly1d(z)
        axs[i, j].plot(ISCWSA_TOTAL_cov_nev[columns[i * 2 + j]],
                       p(ISCWSA_TOTAL_cov_nev[columns[i * 2 + j]]), "r--")

        # show the equation of the line with 2 decimal places
        axs[i, j].text(0.05, 0.9, 'y = ' + str(z[0]) + 'x + ' + str(z[1]),
                       transform=axs[i, j].transAxes,
                       fontsize=8,
                       verticalalignment='top')

        # calculate the r2 score
        r2 = r2_score(ISCWSA_TOTAL_cov_nev[columns[i * 2 + j]],
                      sum_of_errors[columns[i * 2 + j]])
        axs[i, j].text(0.05, 0.7, 'r2 = ' + str(r2),
                       transform=axs[i, j].transAxes,
                       fontsize=8,
                       verticalalignment='top')

# make plot tight layout
plt.tight_layout()

# save the plot in error figures folder
plt.savefig(Path + '/' + ISCWSA_case[ISCWSA_file_name]['folder_to_save_fig'] + '/' + "ISCWSAvsCORVA" + '_cov_nev.png')
# plt.show()
