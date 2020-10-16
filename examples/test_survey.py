import pandas as pd

from welleng.survey import Survey

def regenerate_survey(filename):
    s = pd.read_csv(filename)

    survey = Survey(
        md=s["MD"].to_list(),
        inc=s["INC"].to_list(),
        azi=s["AZI"].to_list(),
        start_nev=[s["UTM_Y"][0], s["UTM_X"][0], s["TVDSS"][0]]
    )

    df = pd.DataFrame(
        dict(
            UTM_X = survey.e,
            UTM_Y = survey.n,
            TVDSS = survey.tvd,
            MD = survey.md,
            INC = survey.inc_deg,
            AZI = survey.azi_deg,
            DLS = survey.dls,
            X_OFFSET = survey.x,
            Y_OFFSET = survey.y
        )
    )

    return df

if __name__ == "__main__":

    filename = f'data/surveys/donor_25-11-G-38 AY3_target_PR67_mlt0.csv'
    df = regenerate_survey(filename)
    df.to_csv(f"{filename.split('.')[0]}_jonny.{filename.split('.')[1]}")

    filename = f'data/surveys/donor_25-11-G-3 AY1T2_target_PR59_0.csv'
    df.to_csv(f"{filename.split('.')[0]}_jonny.{filename.split('.')[1]}")
    df.to_csv(f"{filename.split('.')[0]}_jonny.{filename.split('.')[1]}")

    filename = f'data/surveys/donor_25-11-G-3 AY1T2_target_PR59_1.csv'
    df.to_csv(f"{filename.split('.')[0]}_jonny.{filename.split('.')[1]}")
    df.to_csv(f"{filename.split('.')[0]}_jonny.{filename.split('.')[1]}")
