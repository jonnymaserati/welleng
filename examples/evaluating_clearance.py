import pandas as pd
import trimesh

import welleng as we
from welleng.survey import Survey, SurveyHeader
import welleng.clearance
from welleng.mesh import WellMesh
import os

# os.environ['DISPLAY'] = ':1'

try:
    import ray
    multiprocessing = True
except ModuleNotFoundError:
    multiprocessing = False

filename = (
    "reference/standard-set-of-wellpaths"
    "-for-evaluating-clearance-scenarios-r4-17-may-2017.xlsx"
)

# Import some well trajectory data. Here we'll use the ISCWSA trajectories,
# extracting the data from the Excel file downloaded from their website.
# filename = (
#     "../reference/standard-set-of-wellpaths"
#     "-for-evaluating-clearance-scenarios-r4-17-may-2017.xlsx"
# )

print("Loading data...")
try:
    data = we.io.import_iscwsa_collision_data(filename)
except:
    print(
        "Make sure you've updated filename to your local copy of ISCWSA's"
        " clearance scenarios"
    )

# well_ref_params = dict(
#     Latitude=60.000000,
#     BTotal=50000.00,
#     Dip=70.00,
#     Declination=0.00,
#     Convergence=0.0,
#     G=9.80665
# )

# Make a dictionary of surveys
print("Making surveys...")
surveys = {}
for well in data["wells"]:
    sh = SurveyHeader(
        name=well,
        latitude=60.,
        b_total=50000.,
        dip=70.,
        declination=.0,
        convergence=0.,
        G=9.80665,
        azi_reference='grid'
    )
    if well == "Reference well":
        radius = 0.4572
    else:
        radius = 0.3048

    s = Survey(
        md=data["wells"][well]["MD"],
        inc=data["wells"][well]["IncDeg"],
        azi=data["wells"][well]["AziDeg"],
        n=data["wells"][well]["N"],
        e=data["wells"][well]["E"],
        tvd=data["wells"][well]["TVD"],
        radius=radius,
        header=sh,
        error_model="iscwsa_mwd_rev5",
        start_xyz=[
            data["wells"][well]["E"][0],
            data["wells"][well]["N"][0],
            data["wells"][well]["TVD"][0]
            ],
        start_nev=[
            data["wells"][well]["N"][0],
            data["wells"][well]["E"][0],
            data["wells"][well]["TVD"][0]
            ],
        deg=True,
        unit="meters"
    )
    s_inter = we.connector.interpolate_survey(s, step=10.)
    # surveys[well] = s
    surveys[well] = s_inter

# Add clearance data to dictionary
results = {}
reference = surveys["Reference well"]
scene = trimesh.scene.scene.Scene()
names = []
colors = []

if multiprocessing:
    @ray.remote
    def worker(well, reference, surveys):
        if well == "Reference well":
            color = 'red'
            result_iscwsa, R = None, None
        else:
            offset = surveys[well]
            if well == "10 - well":
                c = we.clearance.Clearance(reference, offset, kop_depth=900)
            else:
                c = we.clearance.Clearance(reference, offset)

            print(f"Calculating ISCWSA clearance for {well}...")
            result_iscwsa = we.clearance.ISCWSA(c)

            print(f"Calculating mesh clearance for {well}...")
            rm = we.clearance.MeshClearance(c, sigma=2.445)

            color = 'blue'

            class R:
                pass

            R = R()
            R.ref_md = rm.ref_md
            R.nev = rm.nev
            R.off_md = rm.off_md
            R.hoz_bearing_deg = rm.hoz_bearing_deg
            R.distance_CC = rm.distance_CC
            R.ref_PCR = rm.ref_PCR
            R.off_PCR = rm.off_PCR
            R.calc_hole = rm.calc_hole
            R.SF = rm.SF

        m = WellMesh(
            survey=surveys[well],
            n_verts=12,
            sigma=2.445,
        )

        return (result_iscwsa, R, m, well, color)

    ray.init()
    s = ray.put(surveys)

    data = ray.get([
        worker.remote(well, reference, s)
        for well in surveys
    ])

    for i, well in enumerate(surveys):
        results[well] = {
                "iscwsa": data[i][0],
                "mesh": data[i][1]
            }
        scene.add_geometry(
            data[i][2].mesh, node_name=well, geom_name=well, parent_node_name=None
        )
        colors.append(data[i][4])
        names.append(data[i][3])


else:
    for well in surveys:
        names.append(well)
        if well == "Reference well":
            colors.append('red')
            pass
        else:
            offset = surveys[well]
            if well == "10 - well":
                c = we.clearance.Clearance(reference, offset, kop_depth=900)
            else:
                c = we.clearance.Clearance(reference, offset)

            print(f"Calculating ISCWSA clearance for {well}...")
            result_iscwsa = we.clearance.ISCWSA(c)

            print(f"Calculating mesh clearance for {well}...")
            result_mesh = we.clearance.MeshClearance(c, sigma=2.445)

            colors.append('blue')

            results[well] = {
                "iscwsa": result_iscwsa,
                "mesh": result_mesh
            }

        # make a well mesh and add it to the scene for visualizing the wells
        m = WellMesh(
            survey=surveys[well],
            n_verts=12,
            sigma=2.445,
        )

we.visual.plot(scene, names=names, colors=colors)

# if you want to export the scene, say to blender, then do something like this
# save the scene (make sure the blender directory exists else change the
# save location)
# scene.export("blender/scene.glb")

# transform the scene so that it imports nicely into Blender
# transform_trimesh_scene(
#   scene, origin=([0,0,0]), scale=100, redux=1
# ).export(
#       "blender/scene_transform.glb"
# )

# output error data
# well = "08 - well"
# errors = [
#   e for e in results[well]["iscwsa"].c.offset.err.errors.errors.keys()
# ]
# error_data = []
# for i, md in enumerate(results[well]["iscwsa"].c.offset.md):
#     error_data.append(
#         [
#             md,
#             [
#                 {f'{e}': (
#                     results[well]["iscwsa"].c.offset.err.errors.errors[e].cov_NEV.T[i]}
#                 )
#                 for e in errors
#             ]
#         ]
#     )

# export the data to Excel
save_as = f'data/output/output.xlsx'
print(f"Exporting data to {save_as}...")
with pd.ExcelWriter(f'data/output/output.xlsx') as writer:
    for well in results.keys():
        if well == "Reference well": continue
        r = results[well]['iscwsa']
        data = {
            "REF_MD (m)": r.c.ref.md,
            "REF_TVD (m)": r.c.ref.tvd,
            "REF_N (m)": r.c.ref.n,
            "REF_E (m)": r.c.ref.e,
            "Offset_MD (m)": r.off.md,
            "Offset_TVD (m)": r.off.tvd,
            "Offset_N (m)": r.off.n,
            "Offset_E (m)": r.off.e,
            "Hoz_Bearing (deg)": r.hoz_bearing_deg,
            "C-C Clr Dist (m)": r.dist_CC_Clr,
            "Ref_PCR (m 1sigma)": r.ref_PCR,
            "Offset_PCR (m 1 sigma)": r.off_PCR,
            "Calc hole": r.calc_hole,
            "ISCWSA ACR": r.SF
        }
        df = pd.DataFrame(data=data)
        df.to_excel(writer, sheet_name=f'{well} - iscwsa')

        r = results[well]['mesh']
        data = {
            "REF_MD (m)": r.ref_md,
            "REF_TVD (m)": [z[0][0][2] for z in r.nev],
            "REF_N (m)": [n[0][0][0] for n in r.nev],
            "REF_E (m)": [e[0][0][1] for e in r.nev],
            "Offset_MD (m)": r.off_md,
            "Offset_TVD (m)": [z[1][0][2] for z in r.nev],
            "Offset_N (m)": [n[1][0][0] for n in r.nev],
            "Offset_E (m)": [e[1][0][1] for e in r.nev],
            "Hoz_Bearing (deg)": r.hoz_bearing_deg,
            "C-C Clr Dist (m)": r.distance_CC,
            "Ref_PCR (m 1sigma)": r.ref_PCR,
            "Offset_PCR (m 1 sigma)": r.off_PCR,
            "Calc hole": r.calc_hole,
            "SF": r.SF
        }
        df = pd.DataFrame(data=data)
        df.to_excel(writer, sheet_name=f'{well} - mesh')

input("Done! Press ENTER to continue...")
