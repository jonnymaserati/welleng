import json, pandas as pd, numpy as np

from welleng.survey import Survey

filename = f'data/surveys/nodes_rrt_donor_25-11-G-38 AY3_target_PR67_IncAzi_HighRes.json'

with open(filename) as f:
    data = json.load(f)

with pd.ExcelWriter(f'data/surveys/output.xlsx') as writer:
    for i, w in enumerate(data):
        md = w["md"]
        inc = w["inclination"]
        azi = w["azimuth"]
        dls = w["dog_leg"]
        e, n, tvd = np.array(w["coords"]).T
        start_nev = np.array([n, e, tvd]).T.reshape(-1,3)[0]
        start_xyz = np.array(w["coords"])[0]
        x, y, z = (np.array(w["coords"]) - start_xyz).T

        s = Survey(
            md=md,
            inc=inc,
            azi=azi,
            start_nev=start_nev,
            deg=True,
        )

        data = {
            "MD": md,
            "INC": inc,
            "AZI": azi,
            "DLS": dls,
            "E": e,
            "N": n,
            "TVD": tvd,
            "X": x,
            "Y": y,
            "DLS_MC": s.dls,
            "E_MC": s.e,
            "N_MC": s.n,
            "TVD_MC": s.tvd,
            "X_MC": s.x,
            "Y_MC": s.y
        }
        df = pd.DataFrame(data=data)
        df.to_excel(writer, sheet_name=f'well_{i}')

print("Finished")