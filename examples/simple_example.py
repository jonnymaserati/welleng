'''
examples/simple_example.py
--------------------------
A simple example of how to generate a pair of well trajectoris, generate
their error ellipses and the associated mesh and then determine the clearances
and Separation Factors between the meshes, along with the closest point on the
offset well for each point on the reference well.

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''
import welleng as we
from tabulate import tabulate
# import os

# os.environ['DISPLAY'] = ':1'

# construct simple well paths
print("Constructing wells...")
connector_reference = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0.,
        azi1=0.,
        pos2=[-100., 0., 2000.],
        inc2=90,
        azi2=60,
    ),
    step=50
)

connector_offset = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0.,
        azi1=225.,
        pos2=[-280., -600., 2000.],
        inc2=90.,
        azi2=270.,
    ),
    step=50
)

# make survey objects and calculate the uncertainty covariances
print("Making surveys...")
sh_reference = we.survey.SurveyHeader(
    name="reference",
    azi_reference="grid"
)
survey_reference = we.survey.Survey(
    md=connector_reference.md,
    inc=connector_reference.inc_deg,
    azi=connector_reference.azi_grid_deg,
    header=sh_reference,
    error_model='ISCWSA MWD Rev4'
)
sh_offset = we.survey.SurveyHeader(
    name="offset",
    azi_reference="grid"
)
survey_offset = we.survey.Survey(
    md=connector_offset.md,
    inc=connector_offset.inc_deg,
    azi=connector_offset.azi_grid_deg,
    start_nev=[100., 200., 0.],
    header=sh_offset,
    error_model='ISCWSA MWD Rev4'
)

# generate mesh objects of the well paths
print("Generating well meshes...")
mesh_reference = we.mesh.WellMesh(
    survey_reference
)
mesh_offset = we.mesh.WellMesh(
    survey_offset
)

# determine clearances
print("Setting up clearance models...")
c = we.clearance.Clearance(
    survey_reference,
    survey_offset
)

print("Calculating ISCWSA clearance...")
clearance_ISCWSA = we.clearance.ISCWSA(c)

print("Calculating mesh clearance...")
clearance_mesh = we.clearance.MeshClearance(c, sigma=2.445)

# tabulate the Separation Factor results and print them
results = [
    [md, sf0, sf1]
    for md, sf0, sf1
    in zip(c.reference.md, clearance_ISCWSA.SF, clearance_mesh.SF)
]

print("RESULTS\n-------")
print(tabulate(results, headers=['md', 'SF_ISCWSA', 'SF_MESH']))

# get closest lines between wells
lines = we.visual.get_lines(clearance_mesh)

# plot the result
we.visual.plot(
    [mesh_reference.mesh, mesh_offset.mesh],  # list of meshes
    names=['reference', 'offset'],  # list of names
    colors=['red', 'blue'],  # list of colors
    lines=lines
)

print("Done!")
