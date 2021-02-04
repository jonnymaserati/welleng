import welleng as we
import os

os.environ['DISPLAY'] = ':1'

# Make up a list of Cartesian points
carts = [
    [0., 0., 0.],
    [0., 0., 300.],
    [0., 1000., 1000.],
    [-1000, -1000, 2000]
]

# Push the points to the connect_points function to generate a survey
survey = we.connector.connect_points(
    carts,
    vec_start=[0, 0, 1],
    dls_design=3.0,
    nev=True,
    step=30,
    md_start=0.
)

# Generate a mesh for plotting
mesh = we.mesh.WellMesh(
    survey,
    method='circle'
).mesh

# Plot the results
we.visual.plot(mesh, points=carts)
