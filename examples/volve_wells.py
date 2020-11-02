import welleng as we
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET

# for ease I accessed the data file locally and gave it a
# shorter name. You'll need to change this to reflect the
# local location of the data file.
filename = f'data/Volve.xml'

# read the WITSML data
tree = ET.parse(filename)
root = tree.getroot()

# extract the survey data and create a dataframe
survey_data = [
    c.attrib for c in root
    if c.tag == "CD_DEFINITIVE_SURVEY_STATION"
]
df = pd.DataFrame(survey_data)
df['md'] = df['md'].astype(float)

wells = df['def_survey_header_id'].unique()

data = {}

for well in wells:
    w = df.loc[
        df['def_survey_header_id'] == well
    ].sort_values(by=['md'])
    cov_nev = we.survey.make_long_cov(
        w['covariance_yy'],
        w['covariance_xy'],
        w['covariance_yz'],
        w['covariance_xx'],
        w['covariance_xz'],
        w['covariance_zz']
    ).astype(float)

    # radius data is sometimes missing or zero and looks to be in inches
    # default these to 15" radius and convert to meters
    radius = np.array(w['casing_radius']).astype(float)
    radius = np.where(((radius == np.nan) | (radius == 0)), 15, radius)
    radius *= 0.0254

    s = we.survey.Survey(
        md=np.array(w['md']).astype(float),
        inc=np.array(w['inclination']).astype(float),
        azi=np.array(w['azimuth']).astype(float),
        n=np.array(w['offset_north']).astype(float),
        e=np.array(w['offset_east']).astype(float),
        tvd=np.array(w['tvd']).astype(float),
        cov_nev=cov_nev,
        radius=radius
    )

    # some wells are missing covariance data: skip those for now
    try:
        m = we.mesh.WellMesh(s)
        data[well] = m
    except:
        print(f"{well} is missing data")

# create a trimesh scene, transform the scene to one that can be
# rendered in blender and export to file
scene = we.mesh.make_trimesh_scene(data)
scene_transformed = we.mesh.transform_trimesh_scene(
    scene, origin=None, scale=100, redux=1
)
scene_transformed.export('blender/volve.glb')

print("Done")