import welleng as we
import numpy as np
from vedo import Arrows, Lines
import random
import os

# os.environ['DISPLAY'] = ':1'

# Some code for testing the connector module.

# Generate some random pairs of points
pos1 = [0., 0., 0.]
md1 = 0

pos2 = np.random.random(3) * 500

vec1 = np.random.random(3)
vec1 /= np.linalg.norm(vec1)
inc1, azi1 = np.degrees(we.utils.get_angles(vec1, nev=True)).reshape(2).T

vec2 = np.random.random(3)
vec2 /= np.linalg.norm(vec2)

inc2, azi2 = np.degrees(we.utils.get_angles(vec2, nev=True)).reshape(2).T

md2 = 100 + random.random() * 1000

# Define some random input permutations

number = 7
rand = random.random()
rand = 7

# 1: test only md2 (hold)
if rand < 1 * 1 / number:
    option = 1
    expected_method = 'hold'
    vec2, pos2, inc2, azi2 = None, None, None, None

# 2: test md2 and an inc2
elif rand < 1 * 2 / number:
    option = 2
    expected_method = 'min_curve'
    pos2, vec2, azi2 = None, None, None

# 3: test md2 and azi2
elif rand < 1 * 3 / number:
    option = 3
    expected_method = 'min_curve'
    pos2, vec2, inc2 = None, None, None

# 4: test md2, inc2 and azi2
elif rand < 1 * 4 / number:
    option = 4
    expected_method = 'min_curve'
    pos2, vec2 = None, None

# 5 test pos2
elif rand < 1 * 5 / number:
    option = 5
    expected_method = 'min_dist_to_target'
    vec2, inc2, azi2, md2 = None, None, None, None

# 6 test pos2 vec2
elif rand < 1 * 6 / number:
    option = 6
    expected_method = 'curve_hold_curve'
    md2, inc2, azi2, = None, None, None

# 7 test pos2, inc2 and azi2
else:
    option = 7
    expected_method = 'curve_hold_curve'
    md2, vec2 = None, None

# Print the input parameters

print(
    f"Option: {option}\tExpected Method: {expected_method}\n"
    f"md1: {md1}\tpos1: {pos1}\tvec1: {vec1}\tinc1: {inc1}\tazi1: {azi1}\n"
    f"md2: {md2}\tpos2: {pos2}\tvec2: {vec2}\tinc2: {inc2}\tazi2: {azi2}\n"
)

# Initialize a connector object and connect the inputs
section = we.connector.Connector(
    pos1=[0., 0., 0],
    vec1=vec1,
    md2=md2,
    pos2=pos2,
    vec2=vec2,
    inc2=inc2,
    azi2=azi2,
    degrees=True,
    min_tangent=0.,
    dls_design=1.0,
    delta_dls=0.1,
    max_iterations=1_000
)

# Print some pertinent calculation data

print(
    f"Method: {section.method}\n",
    f"radius_design: {section.radius_design}\t",
    f"radius_critical: {section.radius_critical}\n"
    f"radius_design2: {section.radius_design2}\t",
    f"radius_critical2: {section.radius_critical2}\n"
    f"iterations: {section.iterations}"
)

# Create a survey object of the section with interpolated points and coords
survey = section.survey(radius=5, step=30)

# test interpolate_md function
node = survey.interpolate_md(123).properties()

# As a QAQC step, check that the wellpath hits the defined turn points
start_points = np.array([section.pos1])
end_points = np.array([section.pos_target])
if section.pos2 is not None:
    start_points = np.concatenate((start_points, [section.pos2]))
    end_points = np.concatenate(([section.pos2], [section.pos_target]))
if section.pos3 is not None:
    start_points = np.concatenate((start_points, [section.pos3]))
    end_points = np.concatenate(
        ([section.pos2], [section.pos3], [section.pos_target])
    )
lines = Lines(
    startPoints=start_points,
    endPoints=end_points,
    c='green',
    lw=5
)

# Add some arrows to represent the vectors at the start and end positions
scalar = 150
arrows = Arrows(
    startPoints=np.array([
        section.pos1,
        section.pos_target
    ]),
    endPoints=np.array([
        section.pos1 + scalar * section.vec1,
        section.pos_target + scalar * section.vec_target
    ]),
    s=0.5,
    res=24
)

# generate a mesh of the generated section from the survey
# use the 'circle' method to construct a cylinder with constant radius
mesh = we.mesh.WellMesh(
    survey=survey,
    method='circle',
    n_verts=24,
)

# plot the results
we.visual.plot(
    [mesh.mesh],
    lines=lines,
    arrows=arrows,
)

print("Done")
