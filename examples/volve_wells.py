'''
examples/volve_wells.py
-----------------------
An example of importing an EDM dataset and extracting the well survey
(including error data) for all the wellbores in the dataset and using welleng
to generate the uncertainty ellipses to generate meshes which can be plotted
for visual comparison.

author: Jonny Corcutt
email: jonnycorcutt@gmail.com
date: 29-09-2021
'''


import numpy as np
import pandas as pd
from tqdm import tqdm
import welleng as we
import xml.etree.ElementTree as ET

# for ease I accessed the data file locally and gave it a
# shorter name. You'll need to change this to reflect the
# local location of the data file.
filename = 'data/Volve.xml'

# read the WITSML data
print("Importing the data...")
try:
    tree = ET.parse(filename)
    root = tree.getroot()
except:# noqa E722
    print("Please download the volve data and point filename to its location")

# extract the survey data and create a dataframe
print("Extracting survey data...")
survey_data = [
    c.attrib for c in root
    if c.tag == "CD_DEFINITIVE_SURVEY_STATION"
]
df = pd.DataFrame(survey_data)
df['md'] = df['md'].astype(float)

wells = df['def_survey_header_id'].unique()

data = {}

print("Processing wells...")
# this is a bit slow... multithread this if you want to do it faster
for i, well in enumerate(tqdm(wells)):
    sh = we.survey.SurveyHeader(
        name=well,
        azi_reference="grid"
    )
    w = df.loc[
        df['def_survey_header_id'] == well
    ].sort_values(by=['md'])
    cov_nev = we.survey.make_long_cov(np.array([
        w['covariance_yy'],
        w['covariance_xy'],
        w['covariance_yz'],
        w['covariance_xx'],
        w['covariance_xz'],
        w['covariance_zz']
    ]).T).astype(float)

    # radius data is sometimes missing or zero and looks to be in inches
    # default these to 15" radius and convert to meters
    radius = np.array(w['casing_radius']).astype(float)
    radius = np.where((np.isnan(radius) | (radius == 0)), 15, radius)
    radius *= 0.0254

    s = we.survey.Survey(
        md=np.array(w['md']).astype(float) / 3.281,
        inc=np.array(w['inclination']).astype(float),
        azi=np.array(w['azimuth']).astype(float),
        n=np.array(w['offset_north']).astype(float),
        e=np.array(w['offset_east']).astype(float),
        tvd=np.array(w['tvd']).astype(float) / 3.281,  # appears that TVD data is in feet?
        header=sh,
        cov_nev=cov_nev,
        radius=radius
    )

    # some wells are missing covariance data: skip those for now
    try:
        m = we.mesh.WellMesh(s)
        data[well] = m
    except:# noqa E722
        print(f"{well} is missing data")

# create a trimesh scene and plot with welleng plotter
print("Making a scene and plotting...")
scene = we.mesh.make_trimesh_scene(data)
we.visual.plot(scene)

##########################################################################
# if you wanted to export a transformed scene so that you can, for example
# import it into blender, this is how you can do it (note that blender)
# is quite restrictive in how it expects the scene to be set up, so this
# transform function modifies the scene so that it will be visible in
# blender):
##########################################################################

# scene_transformed = we.mesh.transform_trimesh_scene(
#     scene, origin=None, scale=100, redux=1
# )
# scene_transformed.export('blender/volve.glb')

print("Done")
