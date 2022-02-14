'''
Create some sections between pairs of points with varying parameters along
a wellpath and then connect these sections together to create a design survey
listing.

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''

from welleng.connector import Connector
from welleng.survey import from_connections
from welleng.mesh import WellMesh
from welleng.visual import plot
# import os

# os.environ['DISPLAY'] = ':1'

# define start position and vector
pos1 = [0., 0., 0.]
vec1 = [0., 0., 1.]  # this is the same as inc=0, azi=0

# for the first section, we want to hold vertical for the first 500m
md2 = 500
vec2 = [0., 0., 1.]

# let's connect those points
s1 = Connector(
    pos1=pos1,
    vec1=vec1,
    md2=md2,
    vec2=vec2
)

# next we want to build to 30 degrees towards east
inc3 = 30
azi3 = 90

# let's connect this to the end of the previous section
# use the `_target` suffix to get the last position of the previous section
s2 = Connector(
    pos1=s1.pos_target,
    vec1=s1.vec_target,
    md1=s1.md_target,
    inc2=inc3,
    azi2=azi3
)

# the subsurface target has coordinates [-800, 300, 1800]
# we want to be near horizontal when we hit this point so that we can
# drill the reservoir horizontal (let's say 88 deg) and the reseroir
# orientation is southeast-northwest and we have good directional control
pos4 = [-800., 300., 1800.]
inc4 = 88
azi4 = 315
dls_design4 = 4

s3 = Connector(
    pos1=s2.pos_target,
    vec1=s2.vec_target,
    md1=s2.md_target,
    pos2=pos4,
    inc2=inc4,
    azi2=azi4,
    dls_design=dls_design4
)

# finally, we want a 500m horizontal section in the reservoir
md5 = s3.md_target + 500
inc5 = 90

s4 = Connector(
    pos1=s3.pos_target,
    vec1=s3.vec_target,
    md1=s3.md_target,
    md2=md5,
    inc2=inc5
)

# make a list of the well sections
well = [s1, s2, s3, s4]

# generate the survey listing and interpolate to get desired survey spacing
survey = from_connections(well, step=30)

# make a mesh
mesh = WellMesh(survey, method='circle')
    
# finally, plot it
plot([mesh.mesh], interactive=False)

print("Done!")
