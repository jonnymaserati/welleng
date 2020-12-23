import numpy as np
from openpyxl import load_workbook
from dataclasses import dataclass
import json

# Short script to extract and serialize the well data from the
# ISCWSA error model spreadsheet. The intent is to save on test time versus
# having to extract the data from Excel workbook each time.
# This code will remain in the repo to demonstrate where the data is
# derived and how it's been extracted.

# import the ISCWSA standard Excel data
workbook = load_workbook(
    filename="examples/data/error-model-example-mwdrev4-iscwsa-1.xlsx",
    data_only=True
)
# model = workbook['Model']
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
    G=wellpath['B3'].value,
    BTotal=wellpath['B4'].value,
    Dip=wellpath['B5'].value,
    Declination=wellpath['B6'].value,
    Convergence=wellpath['B7'].value,
    AzimuthRef=wellpath['B8'].value,
    DepthUnits=wellpath['B9'].value,
    VerticalIncLimit=wellpath['B10'].value,
    FeetToMeters=wellpath['B11'].value
)

well_ref_params = dict(
    Latitude=wh.Latitude,
    G=wh.G,
    BTotal=wh.BTotal,
    Dip=wh.Dip,
    Declination=wh.Declination,
    Convergence=wh.Convergence,
    AzimuthRef=wh.AzimuthRef
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

survey_deg = np.stack((MD, IncDeg, AziDeg), axis=-1)

IncRad = np.radians(IncDeg)
AziRad = np.radians(AziDeg)
AziTrue = AziRad + np.full(len(AziRad), np.radians(wh.Convergence))
AziG = AziTrue - np.full(len(AziRad), np.radians(wh.Convergence))
AziMag = AziTrue - np.full(len(AziRad), np.radians(wh.Declination))

survey_master = np.stack((MD, IncRad, AziRad), axis=-1)
TVD = np.array([TVD])
AziTrue = np.array([AziTrue])
AziG = np.array([AziG])
AziMag = np.array([AziMag])

# convert required data to dictionary
well_data = {}
well_data['well_ref_params'] = well_ref_params
well_data['survey'] = survey_deg.tolist()

# serialize data into file:
json.dump(
    well_data,
    open("test/test_data/error_mwdrev4_iscwsa_well_data.json", 'w')
)
