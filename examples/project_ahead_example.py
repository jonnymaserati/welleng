'''
examples/project_ahead_example.py
---------------------------------
An example of how to:
  - project ahead from a position and vector for a given delta_md, dls and
  toolface
  - create a survey listing and error model
  - use a convenience method for creating a figure of the survey
  - determine the node properties at the bit relative to the last survey
  station
  - determine the trajectory required to get from the last survey station in
  a survey, to a target node (position and vector)
  - visualize the combined surveys

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''

import welleng as we

# generate a node describing the position of a point relative to a survey
# station, projected using the delta_md, dls and toolface
node0 = we.survey.project_ahead(
    pos=[0, 0, 0],
    vec=[0, 0, 1],
    delta_md=30,
    dls=3,
    toolface=0
)

# create a survey, interpolate every 30m and then generate the error model
survey = we.survey.Survey(
    md=[0, 100, 500, 3000],
    inc=[0, 0, 30, 30],
    azi=[0, 0, 90, 90],
    deg=True,
).interpolate_survey(step=30).get_error('ISCWSA MWD Rev5')

# make a plotly figure of the survey
fig = survey.figure()

# get the `Node` for the bit relative to the last station in the survey,
# where the bit is 9m along hole from the survey tool
node1 = survey.project_to_bit(9)

# create a `Node` describing a target position and vector
node_target = we.node.Node(
    pos=[2000, 2000, 3500],
    vec=[0, 1, 0]
)

# generate a wellpath from the last station in the survey to the target node
survey_to_target = survey.project_to_target(
    node_target
)

# combine the surveys in a figure to view them
for t in survey_to_target.figure().data:
    fig.add_trace(t)

# render the figure to the default web browser
fig.show()

print("Done")
