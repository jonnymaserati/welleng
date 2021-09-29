'''
An example of how to extract torque and drag data for a specific wellbore
in an EDM datafile. The data has been sources from Equinor's Volve data
village.

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''
from welleng.exchange.edm import EDM
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# probably not necessary... vscode runs headless on my machine for some reason
# import os
# os.environ['DISPLAY'] = ':1'


def main():
    # import EDM data
    FILENAME = 'data/Volve.xml'
    edm = EDM(FILENAME)

    # input a case_id
    case_id = '3jcyt'
    case_name = edm.get_case_name_from_id(case_id)
    wellbore_name = edm.wellbore_id_to_name['3W9ZjV2yj0']

    drag_data = edm.get_attributes(
        tags=['WP_TDA_DRAGCHART'],
        attributes={'case_id': case_id}
    )

    # the data is unsorted and all floats so sort on floated "run_depth"
    dd = sorted(
        drag_data['WP_TDA_DRAGCHART'],
        key=lambda k: float(k['run_depth'])
    )

    # the easiest way to extract and plot the data is with pandas
    df = pd.DataFrame(dd)

    x = df['run_depth']

    # first the drag traces
    traces = [
        'ob_hookload',
        'ti_hookload',
        'to_hookload',
        'rd_hookload',
        'maximum_hookload',
        'mb_hookload',
        'minmwt_buckle',
        'minmwt_plastic',
    ]
    fig = make_subplots(
        rows=1, cols=2, subplot_titles=("Drag", "Torque")
    )
    for t in traces:
        fig.add_trace(
            go.Scatter(
                x=df[t],
                y=x,
                name=t
            ),
            row=1,
            col=1,
        )

    # then the torque traces
    traces = [
        'ob_torque',
        'ti_torque',
        'to_torque',
        'rd_torque',
        'maximum_torque',
    ]
    for t in traces:
        fig.add_trace(
            go.Scatter(
                x=df[t],
                y=x,
                name=t
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(
        title='md (ft)',
        autorange='reversed',
        tickformat=':.0f'
    )
    fig.update_xaxes(
        title_text='hookload (kips)',
        row=1, col=1
    )
    fig.update_xaxes(
        title_text='torque (klb-ft)',
        row=1, col=2
    )
    fig.update_layout(
        title_text=f"Torque and Drag: {wellbore_name} - {case_name}"
    )
    fig.show()

    print("Done")


if __name__ == '__main__':
    main()
