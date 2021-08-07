"""
examples/interpolate_sparse_survey.py
-------------------------------------

Sometimes a survey will only have a few points and when it's plotted it
looks weird and angular because these points are connected together with
straight lines.

Here's an example of how to use the `interpolate_survey` function in the
`connector` module to calculate extra points along the minimum curvature
path between the survey stations.

This is also an alternative quick method for creating a well path.
"""

import welleng as we

# create a sparse survey with only 4 stations
s_ref = we.survey.Survey(
    md=[0., 1000., 2000., 5000.],
    inc=[0., 0., 30., 90.],
    azi=[0., 10., 20., 30.],
    error_model='iscwsa_mwd_rev4'
)

# interpolate points between survey stations with delta md of 30
s_ref_interp = we.survey.interpolate_survey(s_ref, step=30)

# generate meshes for visualizing the well paths
m_ref = we.mesh.WellMesh(s_ref)
m_ref_interp = we.mesh.WellMesh(s_ref_interp)

# plot the results
we.visual.plot(
    [m_ref.mesh, m_ref_interp.mesh],
    colors=['red', 'blue'],
    names=['sparse survey', 'interpolated survey']
)
