import numpy as np
import pandas as pd

import welleng as we

from openpyxl import load_workbook
from dataclasses import dataclass

# import the ISCWSA standard Excel data
print("Importing data...")
try:
    workbook = load_workbook(
        filename="examples/data/error-model-example-mwdrev4-iscwsa-1.xlsx",
         data_only=True
    )
except:
    print("Make sure you have a local copy of ISCWSA's Excel file and have updated the location in the code.")
sheets = workbook.sheetnames
model = workbook['Model']
wellpath = workbook['Wellpath']


@dataclass
class WellHeader:
    Latitude: float
    G: float
    BTotal: float
    Dip: float
    Declination: float
    Convergence: float
    AzimuthRef: float
    DepthUnits: str
    VerticalIncLimit: float
    FeetToMeters: float

wh = WellHeader(
    Latitude=wellpath['B2'].value,
    G = wellpath['B3'].value,
    BTotal = wellpath['B4'].value,
    Dip = wellpath['B5'].value,
    Declination = wellpath['B6'].value,
    Convergence = wellpath['B7'].value,
    AzimuthRef = wellpath['B8'].value,
    DepthUnits = wellpath['B9'].value,
    VerticalIncLimit = wellpath['B10'].value,
    FeetToMeters = wellpath['B11'].value
)

well_ref_params = dict(
    Latitude = wh.Latitude,
    G = wh.G,
    BTotal = wh.BTotal,
    Dip = wh.Dip,
    Declination = wh.Declination,
    Convergence = wh.Convergence,   
)

MD = []
IncDeg = []
AziDeg = []
TVD = []

for row in wellpath.iter_rows(
    min_row=3,
    max_row=(),
    min_col=5,
    max_col=9
):
    MD.append(row[0].value)
    IncDeg.append(row[1].value)
    AziDeg.append(row[2].value)
    TVD.append(row[3].value)
    
survey_deg = np.stack((MD,IncDeg,AziDeg), axis=-1)

IncRad = np.radians(IncDeg)
AziRad = np.radians(AziDeg)
AziTrue = AziRad + np.full(len(AziRad), np.radians(wh.Convergence))
AziG = AziTrue - np.full(len(AziRad), np.radians(wh.Convergence))
AziMag = AziTrue - np.full(len(AziRad), np.radians(wh.Declination))

survey_master = np.stack((MD,IncRad,AziRad), axis=-1)
TVD = np.array([TVD])
AziTrue = np.array([AziTrue])
AziG = np.array([AziG])
AziMag = np.array([AziMag])

# get error for entire survey
print("Calculating ISCWSA MWD Rev4 errors...")
err0 = we.error.ErrorModel(
    survey=survey_deg,
    well_ref_params=well_ref_params,
    AzimuthRef=wh.AzimuthRef,
)

cov_nev0 = err0.errors.cov_NEVs.T

# print final covariance matrix for each tool
print("Tool errors at well TD:")
for tool, e in err0.errors.errors.items():
    print(
        f'{tool}:\n{e.cov_NEV.T[-1]}'
    )
print(
        f'TOTAL:\n{cov_nev0[-1]}'
    )

input("Press Enter to end...")