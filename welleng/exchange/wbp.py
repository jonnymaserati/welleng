import os, yaml
import numpy as np
from ..survey import get_sections

class Target:
    def __init__(
        self,
        name,
        location=None,
        geometry={
            'type': None,
            'locked': None,
            'offset': None,
            'orientation': None,
            'radius_1': None,
            'radius_2': None,
            'dip': None,
            'azimuth': None,
            'vertices': [],
            'thickness_up': None,
            'thickness_down': None,
            'color': {
                'color': None,
                'interpreter': None,
                'application': None,
                'feature': None
            },
            'category': None,
        },
    ):
        self.name = name
        self.location = location
        self.geometry = geometry

class TurnPoint:
    def __init__(
        self,
        md=None,
        inc=None,
        azi=None,
        build_rate=None,
        turn_rate=None,
        dls=None,
        toolface=None,
        method=None,
        target=None,
        tie_on=False,
        location=None
    ):
        self.md = md
        self.inc = inc
        self.azi = azi
        self.build_rate = build_rate
        self.turn_rate = turn_rate
        self.dls = dls
        self.toolface = toolface
        self.method = method
        self.target = target
        self.tie_on = tie_on
        self.location = location

class SurveyPoint:
    def __init__(
        self,
        md=None,
        inc=None,
        azi=None,
        cov_xx=None,
        cov_xy=None,
        cov_xz=None,
        cov_yy=None,
        cov_yz=None,
        cov_zz=None,
        x_bias=None,
        y_bias=None,
        z_bias=None,
        tool=None,
        location=None,
    ):
        self.md = md
        self.inc = inc
        self.azi = azi
        self.cov_xx = cov_xx
        self.cov_xy = cov_xy
        self.cov_xz = cov_xz
        self.cov_yy = cov_yy
        self.cov_yz = cov_yz
        self.cov_zz = cov_zz
        self.x_bias = x_bias
        self.y_bias = y_bias
        self.z_bias = z_bias
        self.tool = tool
        self.location = location

class WellPlan:
    def __init__(
        self,
        data,
        depth_unit=None,
        surface_unit=None,
        targets=[],
        line=None,
        # survey,
        # location_type,
        # plan_method,
        # plan_name,
        # parent_name,
        # depth_unit="meters",
        # surface_unit="meters",
        # dirty_flag=3,
        # sidetrack_id=0,
        # extension="0.00000",
        # filename=None,
        # **targets
    ):
        # Import the wbp.yaml file as a dictionary
        wbp_dict_file = os.path.join(
            os.path.dirname(__file__),
            'wbp.yaml')

        with open(wbp_dict_file) as f:
            self.wbp_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.action = {
            'depth': self._get_units,
        }

        self.depth_unit = depth_unit
        self.surface_unit = surface_unit
        self.targets = targets
        self.steps = []
        self.tie_on_point_flag = False
        if line is None:
            self.lines = 0
        else:
            self.lines = line

        self.wbp_data = data

        # if filename is not None:
            # self.load(filename)
        self._process_wbp_data()



        # LOCATIONS = [
        #     "unknown",
        #     "surface",
        #     "ow_sidetrack",
        #     "wp_sidetrack",
        #     "lookahead",
        #     "ow_well",
        #     "complex_extension",
        #     "site",
        #     "site_based_plan",
        #     "compass_well"
        # ]

        # METHODS = [
        #     "curve_only",
        #     "curve_hold",
        #     "opt_align"
        # ]

        # UNITS = [
        #     "feet",
        #     "meters"
        # ]

        # assert location_type in LOCATIONS, "location_type not in LOCATIONS"
        # assert plan_method in METHODS, "plan_method not in METHODS"
        # assert len(plan_name) < 21, "plan_name too long (max length 20)"
        # assert len(parent_name) < 21, "parent_name too long (max length 20)"
        # assert surface_unit in UNITS, "surface_units must be feet or meters"
        # assert depth_unit in UNITS, "depth_units must be feet or meters"

        # self.survey = survey
        # self.location = str(LOCATIONS.index(location_type))
        # self.surface_unit = surface_unit
        # self.depth_unit = depth_unit
        # self.method = str(METHODS.index(plan_method))
        # self.flag = str(dirty_flag)
        # self.st_id = str(sidetrack_id).rjust(8)
        # self.plan_name = str(plan_name).ljust(21)
        # self.parent_name = str(parent_name).ljust(20)
        # self.dls = f"{str(np.around(self.survey.dls[1]))[:5]}".rjust(5)
        # self.kickoff_dls = f"{survey.dls[1]:.2f}".rjust(7)
        # self.extension = str(extension).rjust(8)
        # self.targets = targets

        # self._get_unit()

        # self.doc = []

        # self._make_header()
        # # self._make_tie_on(target=0)
        # self._make_wellplan()

    def _process_wbp_data(self):
        # TODO: finish coding the rest of the target inputs
        self.flag = None
        for i, line in enumerate(self.wbp_data):
            if i < self.lines:
                continue
            # first split the line and look for section headers
            l = line.split()
            m = line.split(':')
            if l[0] == '!':
                pass
            elif l[0] == "DEPTH":
                self._get_units(l[1])
                self.flag = None
            elif l[0] == "TARGETS:":
                self.flag = 'targets'
            elif l[0] == "WELLPLANS:":
                # pass
                self.flag = 'wellplans'
                self.tie_on_point_flag = True
            elif self.flag == 'targets':
                if m[0] == "T":
                    self._initiate_target(m[1])
                elif m[0] == "L":
                    self._add_target_location(m[1])
                elif m[0] == "C":
                    self._add_target_color(line)
                elif m[0] == "G":
                    self._add_target_geometry(line)
                # need to add the rest of the target inputs
            else: # self.flag == 'wellplans':
                if m[0] == "W":
                    if self.flag == 'done':
                        break    
                    # if self.flag == 'wellplans':
                    #     self.flag = 'done'
                    #     break
                    else:
                        self.flag = 'done'
                        self.tie_on_point_flag = True
                        self._add_wellplan_header(line)
                elif m[0] == "P":
                    self._add_turn_point(line)
                elif m[0] == "L":
                    self._add_location_data(m[1])
                    self.tie_on_point_flag = False
                elif m[0] == "X":
                    self._add_extended_survey_point(l[1:])
                elif m[0] == "S":
                    self._add_survey_data(l[1:])
                else:
                    pass
            # else:
            #     pass
            self.lines += 1

    def _initiate_target(self, name):
        self.targets.append(Target(name))

    def _add_target_location(self, data):
        x, y, z = data.split()
        self.targets[-1].location = [float(x), float(y), float(z)]

    def _add_target_geometry(self, data):
        tg = self.targets[-1].geometry
        tg.type = self.wbp_dict['TARGETS']['type'][data[2]]
        tg.locked = data[3]
        tg.offset = [float(data[4:13], float(data[13:23]))]
        
    def _add_target_color(self, data):
        tgc = self.targets[-1].geometry['color']
        if len(data) > 5:
            tgc['color'] = int(data[2:5])
            tgc['interpreter'] = data[5:10]
            tgc['application'] = data[11:21]
            tgc['feature'] = data[22:64]
        else:
            tgc['color'] = int(data[2:])

    def _add_wellplan_header(self, data):
        self.location_type = self.wbp_dict['LOCATION']['type'][data[2]]
        self.plan_method = self.wbp_dict['LOCATION']['plan_method'][data[4]]
        if data[5] in self.wbp_dict['LOCATION']['dirty_flag']:
            self.dirty_flag = self.wbp_dict['LOCATION']['dirty_flag'][data[5]]
        else:
            self.dirty_flag = None
        self.sidetrack_id = string_strip(data[14:22])
        self.plan_name = string_strip(data[23:84])
        self.parent_name = string_strip(data[84:144])    
        self.dls = string_strip(data[144:151], is_float=True)
        self.extension = string_strip(data[151:160], is_float=True)
        self.dls_kickoff = string_strip(data[160:171], is_float=True)

    def _add_turn_point(self, line):
        to = TurnPoint()
        data = line.split(':')[1].split()
        to.md = float(data[0])
        to.azi = float(data[1])
        to.inc = float(data[2])
        to.build_rate = float(data[3])
        to.turn_rate = float(data[4])
        to.dls = float(data[5])
        to.toolface = float(data[6])
        if self.tie_on_point_flag:
            to.tie_on = True
        else:
            to.method = string_strip(data[7])
            try:
                to.target = string_strip(data[8])
            except:
                to.target = None
        self.steps.append(to)

    def _add_extended_survey_point(self, data):
        xsp = SurveyPoint()
        xsp.cov_xx = float(data[0])
        xsp.cov_xy = float(data[1])
        xsp.cov_xz = float(data[2])
        xsp.cov_yy = float(data[3])
        xsp.cov_yz = float(data[4])
        xsp.cov_zz = float(data[5])
        xsp.x_bias = float(data[6])
        xsp.y_bias = float(data[7])
        xsp.z_bias = float(data[8])
        xsp.tool = " ".join(data[9:])
        self.steps.append(xsp)

    def _add_survey_data(self, data):
        s = self.steps[-1]
        s.md, s.azi, s.inc = [float(x) for x in data]

    def _add_location_data(self, data):
        x, y, z = data.split()
        self.steps[-1].location = [float(x), float(y), float(z)]

    def _get_units(self, key):
        self.depth_unit = self.wbp_dict['DEPTH'][key]['depth']
        self.surface_unit = self.wbp_dict['DEPTH'][key]['surface']
    
    def _get_unit(self):
        if self.depth_unit == "meters":
            if self.surface_unit == "meters":
                self.unit = 2
            else:
                self.unit = 3
        elif self.depth_unit == "feet":
            if self.surface_unit == "feet":
                self.unit = 1
            else:
                self.unit = 4

    def _add_plan(
        self,
        section
    ):
        self.doc.append(
            f"P:"
            f"{str(section.md)[:8].rjust(8)}"
            f" {str(section.azi)[:7].rjust(7)}"
            f" {str(section.inc)[:7].rjust(7)}"
            f" {str(section.build_rate)[:7].rjust(7)}"
            f" {str(section.turn_rate)[:7].rjust(7)}"
            f" {str(np.around(section.dls, decimals=1))[:7].rjust(7)}"
            f" {str(np.degrees(section.toolface))[:7].rjust(7)}"
            f" {str(section.method).rjust(4)}"
            f" {str(section.target).rjust(10)}"
        )
        self.doc.append(
            f"L:"
            f"{str(section.x)[:13].rjust(13)}"
            f"{str(section.y)[:13].rjust(13)}"
            f"{str(section.z)[:11].rjust(11)}"
        )


    def _make_wellplan(
        self,
    ):
        sections = get_sections(self.survey)

        for s in sections:
            self._add_plan(s)
    
    def _make_header(self):
        self.doc.append(
            f"DEPTH {self.unit}"
        )

        if self.targets:
            self.doc.append(
                "TARGETS:"
            )
            # TODO: add function to retrieve and list target data and implement
            # a Target class.

        self.doc.append(
            "WELLPLANS:",
        )
        self.doc.append(
            (
                f"W:{self.location}{self.unit}{self.method}{self.flag}   0   0"
                f"{self.st_id} {self.plan_name} {self.parent_name} {self.dls}"
                f"{self.extension}{self.kickoff_dls}"
            )
        )

    # I think this is redundent as it's the same line as the first section
    def _make_tie_on(self, target):
        if self.location == "unknown":
            self.doc.append(
                f"P:0.000000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000    0"
            )
        else:
            self.doc.append((
                f"P:"
                f"{str(self.survey.md[0])[:8].rjust(8)}"
                f" {str(self.survey.azi_deg[0])[:7].rjust(7)}"
                f" {str(self.survey.inc_deg[0])[:7].rjust(7)}"
                f" {str(0.00000).rjust(7)}" # build rate
                f" {str(0.00000).rjust(7)}" # turn rate
                f" {str(0.00000).rjust(7)}" # DLS
                f" {str(np.degrees(self.survey.toolface[0]))[:7].rjust(7)}"
                f" {self.method.rjust(4)}"
                f" {str(target).rjust(10)}"
            ))

            self.doc.append(
                f"L:"
                f"{str(self.survey.x)[:13].rjust(13)}"
                f"{str(self.survey.y)[:13].rjust(13)}"
                f"{str(self.survey.z)[:11].rjust(11)}"
            )

def string_strip(string, is_float=False):
    s = string.strip()
    if len(s) > 0:
        if is_float:
            return float(s)
        else:
            return s
    else:
        return None

def load(filename):
    assert filename[-4:] == '.wbp', 'Wrong format'
    with open(filename) as f:
        wbp_data = [line.rstrip() for line in f]

    total_lines = len(wbp_data)
    line = 0

    depth_unit = None
    surface_unit = None
    targets = []

    well_plans = []

    while True:
        well_plans.append(
            WellPlan(
                data=wbp_data,
                depth_unit=depth_unit,
                surface_unit=surface_unit,
                targets=[],
                line=line
            )
        )
        w = well_plans[-1]
        line = w.lines
        if line == total_lines:
            return well_plans
        else:
            depth_unit = w.depth_unit
            surface_unit = w.surface_unit
            targets = w.targets
