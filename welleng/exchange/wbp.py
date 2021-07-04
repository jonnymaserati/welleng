import os
import yaml
import welleng
import utm
import numpy as np
from datetime import datetime
from welleng.version import __version__ as VERSION

# TODO: need to relocate the class Target to target.py


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
        depth_unit='meters',
        surface_unit='meters',
        survey=None,
        plan_name=None,
        parent_name=None,
        location_type=None,
        plan_method='curve_only',
        dirty_flag=None,
        sidetrack_id=None,
        dls=3.0,
        extension=0,
        wbp_data=None,
        targets=[],
        line=None,
        parent_wbp_file=None,
    ):
        """
        An object for storing data extracted from or for writing to a .wbp
        file. As such, the following parameters are driven by those
        required by Landmark's .wbp format.

        Parameters
        ----------
            depth_unit: string (default: 'meters')
                The units used for expressing depth (z axis or tvd) in either
                'meters' or 'feet'.
            surface_unit: string (default: 'meters')
                The units used for expressing lateral distances (x, y, N, E)
                in either 'meters' or 'feet'.
            survey: welleng.survey.Survey object (default: None)
            plan_name: string (default: None)
                The name of the well bore plan.
            parent_name: string (default: None)
                The name of the parent well bore plan (in the event that
                the planned well is a sidetrack or lateral).
            location_type: string (default: None)
                Best to review the wbp.yaml file for options.
            plan_method: string (default: 'curve_only')
                The method used for joining the plan points in the .wbp file.
                Options can be reviewed in the wbp.yaml file but won't
                currently effect how the code runs, so just leave default.
            dirt_flag: string (default: None)
                Again, review the wbp.yaml file for options, but this is
                not currently used in this code.
            sidetrack_id: string (default: None)
                Leave default, not used.
            dls: float (default: 0)
                Suggests that this sets the design dls for planning, but
                doesn't appear to matter so leave as default.
            extension: float (default: 0)
                Not really sure what this does.
            wbp_data: list of strings (default: None)
                A list of strings with each string representing a line from
                of text loaded from a .wbp file. Used for importing .wbp
                data.
            targets: list of welleng.exchange.wbp.Target objects (default: [])
                A list of target objects, but more of a future function.
            line: int (default: None)
                Used for processing .wbp files that contain multiple well
                bores.

        Returns
        -------
            A welleng.exchange.wbp.WellPlan object representing a well bore.
        """
        # Import the wbp.yaml file as a dictionary
        wbp_dict_file = os.path.join(
            os.path.dirname(__file__),
            'wbp.yaml')

        with open(wbp_dict_file) as f:
            self.wbp_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.parent_data = get_parent_survey(parent_wbp_file)

        self.action = {
            'depth': self._get_units,
        }

        self.depth_unit = depth_unit
        self.surface_unit = surface_unit
        self.targets = targets
        if line is None:
            self.lines = 0
        else:
            self.lines = line

        self.wbp_data = wbp_data
        self.survey = survey
        if self.wbp_data is not None:
            assert self.survey is None, "Either wbp_data or survey"
            self.steps = []
            self.tie_on_point_flag = False
            self._process_wbp_data()
        else:
            assert isinstance(
                survey, welleng.survey.Survey
            ), "Not a welleng Survey"
            assert self.wbp_data is None, "Either wbp_data or survey"
            assert plan_name is not None, "Must provide plan_name"
            assert location_type is not None, "Must provide a location type"

            self.plan_name = plan_name
            self.parent_name = parent_name
            self.location_type = str(location_type)
            self.plan_method = plan_method
            self.dirty_flag = str(dirty_flag)
            self.sidetrack_id = "" if sidetrack_id is None else sidetrack_id
            self.dls = dls
            self.extension = extension

            self.steps = welleng.survey.get_sections(survey)

            self._get_surface_location()
            self._get_local_coordinates()

    def _get_local_coordinates(self):
        sd = np.zeros(3)
        sd[:2] = self.surface_datum[:2]
        self.env_local = np.vstack([
            s.location - sd for s in self.steps
        ])

    def _get_surface_location(self):
        if self.parent_data is not None:
            for line in self.parent_data:
                if "L:" in line:
                    code, e, n, v = line.split()
                    self.surface_datum = np.array([e, n, v]).astype(np.float)
                    break
        else:
            self.surface_datum = np.array([
                self.survey.n,
                self.survey.e,
                self.survey.tvd
            ]).T[0]

    def _process_wbp_data(self):
        """
        Steps through imported .wbp data and interprets line by line to
        populate a WellPlan object.
        """
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
            else:
                if m[0] == "W":
                    if self.flag == 'done':
                        break
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


def string_strip(string, is_float=False):
    s = string.strip()
    if len(s) > 0:
        if is_float:
            return float(s)
        else:
            return s
    else:
        return None


def get_parent_survey(filename):
    if filename is None:
        return None

    assert filename[-4:] == '.wbp', 'Wrong format'
    with open(filename) as f:
        wbp_data = [line.rstrip() for line in f]

    flag = True
    data = []
    for line in wbp_data:
        if "WELLPLANS" in line:
            flag = False
            continue
        if flag:
            continue
        data.append(line)

    return data


# TODO: update so that filename can also be data
def load(filename):
    """
    Loads data line by line from a .wbp file, initiates a WellPlan object
    and populates it with data.

    Parameters
    ----------
        filename: string
            The location and filename of the .wbp file to load.

    Returns
    -------
        A welleng.exchange.wbp.WellPlan object
    """
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
                wbp_data=wbp_data,
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
            # keep passing these on to each well sequentially
            depth_unit = w.depth_unit
            surface_unit = w.surface_unit
            targets = w.targets


def add_targets(doc, targets):
    for t in targets:
        doc.append(f"T:{t.name}")
        doc = add_location(doc, t.location)
        doc.append((
            f"C:"
            f"{str(t.geometry['color']['color']).rjust(3)}"
        ))
    return doc


def add_location(doc, location):
    x, y, z = location
    doc.append((
            f"L:"
            f"{x:>13.2f}"
            f"{y:>13.2f}"
            f"{z:>11.2f}"
        ))
    return doc


def add_header(doc, data):
    d = data.wbp_dict['LOCATION']
    ignored_integer_1 = "   0"
    ignored_integer_2 = " 123"
    plan_name = "" if data.plan_name is None else data.plan_name
    parent_name = "" if data.parent_name is None else data.parent_name
    doc.append((
        f"W:"
        f"{get_key(d['type'], data.location_type)}"
        f"{get_unit_key(data)}"
        f"{get_key(d['plan_method'], data.plan_method)}"
        f"{get_key(d['dirty_flag'], data.dirty_flag)}"
        f"{ignored_integer_1}"
        f"{ignored_integer_2}"
        f" {data.sidetrack_id:>7}"
        f" {plan_name:<60}"
        f" {parent_name:<60}"
        f" {data.dls:>6.2f}"
        f" {data.extension:>7.5f}"
    ))
    # the documentation says there should be a kickoff dls at the end,
    # but I've never seen it and what does the other dls refer to?
    return doc


def add_step(doc, step):
    if isinstance(step, welleng.exchange.wbp.TurnPoint):
        doc = add_turn_point(doc, step)
    else:
        doc = add_survey_point(doc, step)
    return doc


def add_turn_point(doc, step):
    if step.tie_on:
        method = ""
    else:
        method = "0" if step.method is None else step.method
    target = "" if step.target is None else step.target
    # if the toolface is > -99 then need to chop off a decimal
    # this might also be the case with other variables... add similar logic
    # if that needs changing
    toolface = step.toolface
    toolface = f'{step.toolface:>7.2f}' if toolface < -99 else f'{toolface:>7.3f}'
    doc.append((
        f"P:"
        f" {step.md:>7.2f}"
        f" {step.azi:>7.3f}"
        f" {step.inc:>7.3f}"
        f" {step.build_rate:>7.3f}"
        f" {step.turn_rate:>7.3f}"
        f" {step.dls:>7.3f}"
        f" {toolface}"
        f" {method:>4}"
        f" {target:>10}"
    ))
    doc = add_location(doc, step.location)
    return doc


def add_survey_point(doc, step):
    doc.append((
        f"X:"
        f" {step.cov_xx:>8.1f}"
        f" {step.cov_xy:>8.1f}"
        f" {step.cov_xz:>8.1f}"
        f" {step.cov_yy:>8.1f}"
        f" {step.cov_yz:>8.1f}"
        f" {step.cov_zz:>8.1f}"
        f" {step.x_bias:>7.2f}"
        f" {step.y_bias:>7.2f}"
        f" {step.z_bias:>7.2f}"
        f" {step.tool:<38}"
    ))
    doc.append((
        f"S:"
        f" {step.md:>7.2f}"
        f" {step.azi:>7.3f}"
        f" {step.inc:>7.3f}"
    ))
    doc = add_location(doc, step.location)
    return doc


def get_unit_key(data):
    key = [
        key for key in data.wbp_dict['DEPTH'] if (
            data.wbp_dict['DEPTH'][key]['depth'] == data.depth_unit
            and data.wbp_dict['DEPTH'][key]['surface'] == data.surface_unit
        )
    ]
    return key[0]


def get_key(d, value):
    key = [
        key for key in d if (
            d[key] == value
        )
    ]
    return key[0]


def export(data, filename=None, comments=None):
    """
    Export a WellPlan object to .wbp format.

    Parameters
    ----------
        data: welleng.exchange.wbp.WellPlan object or a list of objects
        filename: string (default: None)
            The filename to save the .wbp file to. If None then the
            output is returned as data.
        comments: list of strings (default: None)
            A list of comments to be printed in the header of the .wbp
            file.

    Returns
    -------
        doc: list of strings

    """
    doc = []
    if not isinstance(data, list):
        data = [data]
    for i, w in enumerate(data):
        assert isinstance(
            w, welleng.exchange.wbp.WellPlan
        ), "Not a WellPlan object"
        if i == 0:
            doc.append(f"DEPTH {get_unit_key(w)}")
            doc = add_comments(doc, comments)
            doc.append("TARGETS:")
            doc = add_targets(doc, w.targets)
            doc.append("WELLPLANS:")
            if w.parent_data is not None:
                for line in w.parent_data:
                    doc.append(line)
        doc = add_header(doc, w)
        for s in w.steps:
            doc = add_step(doc, s)

    if filename is None:
        return doc
    else:
        save_to_file(doc, filename)


def add_comments(doc, comments):
    doc.append(f"! {datetime.now():%Y-%m-%d %H:%M:%S%z}")
    if comments is None:
        doc.append(f"! welleng v{VERSION}")
        doc.append("! Written by Jonny Corcutt")
    else:
        for c in comments:
            doc.append(f"! {c}")
    return doc


def save_to_file(doc, filename):
    with open(f"{filename}", 'w') as f:
        f.writelines(f'{l}\n' for l in doc)


def wbp_to_survey(
    data, step=None, radius=10, azi_reference='true', convergence=0.0,
    utm_zone=31, utm_north=True
):
    """
    Converts a WellPlan object created from a .wbp file into a Survey object.

    Parameters
    ----------
        data: wellend.exchange.wbp.WellPlan object
        step: float
            The desired step interval used to create the Survey object.
            e.g. step=30 would create a survey station every 30 meters.
        radius: float (default: 10)
            The radius of the well bore generated in the survey. The
            default is used assuming that the well will be rendered with
            welleng.visual.plot.

    Returns
    -------
        survey: welleng.survey.Survey object
    """
    connections = []
    for i, s in enumerate(data.steps):
        if i == 0:
            if isinstance(s, welleng.exchange.wbp.TurnPoint):
                dls = [s.dls]
            else:
                dls = [0.]
            e, n, v = s.location
            start_nev = [n, e, v * -1]
            continue
        e, n, v = s.location
        p = np.array([n, e, v * -1])  # - np.array(start_nev)
        if i == 1:
            md = [s.md]
            inc = [s.inc]
            azi = [s.azi]
            pos = [p]
            plan = [isinstance(s, welleng.exchange.wbp.TurnPoint)]
            continue

        # need to set dls_design relatively small to trigger adaption to
        # the dls used in the imported design, or to set it to the actual
        # dls used in the design.
        # TODO: update the connector code so that None can be passed for
        # dls_design, which will set np.inf for radius_design and force
        # the dls to be set by radius_critical.
        if isinstance(s, welleng.exchange.wbp.TurnPoint):
            dls_design = s.dls if s.dls > 0 else 1e-5
        else:
            # dls_design = data.dls if data.dls > 0 else None
            dls_design = 1e-5

        c = welleng.connector.Connector(
            pos1=pos[-1],
            md1=md[-1],
            inc1=inc[-1],
            azi1=azi[-1],
            md2=s.md,
            inc2=s.inc,
            azi2=s.azi,
            dls_design=dls_design,
        )
        if isinstance(s, welleng.exchange.wbp.TurnPoint):
            dls.append(s.dls)
        else:
            dls.append(0)
        connections.append(c)
        plan.append(isinstance(s, welleng.exchange.wbp.TurnPoint))
        pos.append(c.pos_target)
        inc.append(np.degrees(c.inc_target))
        azi.append(np.degrees(c.azi_target))
        md.append(c.md_target)

    start_nev = np.array(pos[0])

    survey_data = welleng.connector.interpolate_well(
        connections,
        step=step
    )

    lat, lon = utm.to_latlon(
        start_nev[1],
        start_nev[0],
        utm_zone,
        northern=utm_north,
    )

    sh = welleng.survey.SurveyHeader(
        latitude=lat,
        longitude=lon,
        altitude=start_nev[-1] * -1,
        azi_reference=azi_reference,
        convergence=convergence,
    )

    survey = welleng.connector.get_survey(
        survey_data,
        survey_header=sh,
        start_nev=start_nev,
        radius=radius,
        deg=False
    )

    # Because of the way the imported file is processed, there will likely
    # be duplicate survey stations in the survey. This function strips out
    # these duplicates and rebuilds the survey.
    survey = strip_duplicates(survey)

    return survey


def strip_duplicates(survey):
    """
    Function to strip out identical successive survey stations from a Survey
    object.

    Parameters
    ----------
        survey: welleng.survey.Survey object

    Returns
    -------
        survey_stripped: welleng.survey.Survey object
            A survey object with repeating survey stations removed.
    """
    temp = []
    for i, s in enumerate(zip(
        survey.md, survey.inc_rad, survey.azi_grid_rad, survey.radius
    )):
        if i == 0:
            temp.append(s)
            continue
        # if s == temp[-1]:
        if s[1] == temp[-1][1]:
            continue
        else:
            temp.append(s)

    sh = survey.header
    sh.azi_reference = 'grid'

    md, inc, azi, radius = np.array(temp).reshape(-1, 4).T

    survey_stripped = welleng.survey.Survey(
        md=md,
        inc=inc,
        azi=azi,
        deg=False,
        start_nev=survey.start_nev,
        radius=radius,
        header=sh,
    )

    return survey_stripped
