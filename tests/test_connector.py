from welleng.connector import Connector
import numpy as np

# test hold with only md provided
c1 = Connector(
    vec1=[0, 0, 1],
    md2=500,
)
assert (
    c1.inc_target == c1.inc1
    and c1.azi_target == c1.azi1
    and c1.pos_target[2] == c1.md_target
), "Failed c1"
assert c1.method == 'hold'

c1_survey = c1.survey()

# # test with md2 and vec2 provided (minimum curvature)
c2 = Connector(
    vec1=[0, 0, 1],
    md2=1000,
    vec2=[0, 1, 0]
)

# c2_survey = c2.survey()

# test with pos2 provided (minimum distance)
c3 = Connector(
    vec1=[0, 0, 1],
    pos2=[100, 100, 1000],
)
assert c3.md_target > c3.pos1[2], "Failed c3"

c3_survey = c3.survey()

# test with pos2 needing more aggressive dls (minimum curvature)
c4 = Connector(
    vec1=[0, 0, 1],
    pos2=[200, 400, 200]
)
assert c4.method == 'min_curve_to_target'

# test with pos2 and vec2 provided ()
vec1 = [-1, -1, 1]
vec2 = [1, -1, 0]
c5 = Connector(
    pos1=[0., 0., 0],
    vec1=vec1 / np.linalg.norm(vec1),
    pos2=[0., 1000., 500.],
    vec2=vec2 / np.linalg.norm(vec2),
)
assert c5.method == 'curve_hold_curve'

# test if interpolator and survey functions are working
c5_survey = c5.survey()

# test with pos2, inc1 and azi1 provided ()
c6 = Connector(
    pos1=[0., 0., 0],
    inc1=0.,
    azi1=90,
    pos2=[1000., 1000., 1000.],
    vec2=[0., 0., 1.],
)

# test with different dls for second curve section
c7 = Connector(
    pos1=[0., 0., 0],
    vec1=[0., 0., 1.],
    pos2=[0., 100., 1000.],
    vec2=[0., 0., 1.],
    dls_design2=5
)

# test with dls_critical requirement (actual dls < dls_design)
c8 = Connector(
    pos1=[0., 0., 0],
    vec1=[0., 0., 1.],
    pos2=[0., 100., 100.],
    vec2=[0., 0., 1.],
)
assert c8.radius_critical < c8.radius_design

# test min_curve (inc2 provided)
c9 = Connector(
    pos1=[0., 0., 0],
    vec1=[0., 0., 1.],
    inc2=30,
)
assert c9.method == 'min_curve'

# test min_curve with md less than required radius
c10 = Connector(
    pos1=[0., 0., 0],
    inc1=0,
    azi1=0,
    md2=500,
    inc2=90,
    azi2=0,
)
assert c10.radius_critical < c10.radius_design
