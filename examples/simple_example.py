import welleng as we
import numpy as np
from tabulate import tabulate

# construct simple well paths
print("Constructing wells...")
md = np.linspace(0,3000,100) # 30 meter intervals to 3000 mTD
inc = np.concatenate((
    np.zeros(30), # vertical section
    np.linspace(0,90,60), # build section to 60 degrees
    np.full(10,90) # hold section at 60 degrees
))
azi1 = np.full(100,60) # constant azimuth at 60 degrees
azi2 = np.full(100,225) # constant azimuth at 225 degrees

# make a survey object and calculate the uncertainty covariances
print("Making surveys...")
survey_reference = we.survey.Survey(
    md,
    inc,
    azi1,
    error_model='ISCWSA_MWD'
)

# make another survey with offset surface location and along another azimuth
survey_offset = we.survey.Survey(
    md,
    inc,
    azi2,
    start_nev=[100,200,0],
    error_model='ISCWSA_MWD'
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
    [mesh_reference.mesh, mesh_offset.mesh], # list of meshes
    names=['reference', 'offset'], # list of names
    colors=['red', 'blue'], # list of colors
    lines=lines
)

print("Done!")