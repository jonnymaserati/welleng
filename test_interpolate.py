import numpy as np

from utils.well_utils import *

s = Survey(md=[100,150], inc=[5,10], azi=[0,315])

x = 10

s_interp_new = interpolate(s,0,x)

s_interp_old = interpolate_survey(s,0,x)

print(s_interp_new)