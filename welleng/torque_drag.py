from copy import deepcopy
import numpy as np

try:
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    PLOTLY = True
except ImportError:
    PLOTLY = False

from .survey import interpolate_md, Survey
from .units import ureg


class TorqueDrag:
    def __init__(
        self, survey, wellbore, string, fluid_density, name=None,
        wob=None, tob=None, overpull=None,
    ):
        """
        A class for calculating wellbore torque and drag, based on the
        "Torque and Drag in Directional Wells--Prediction and Measurement
        (SPE 11380-PA) by C.A. Johancsik et al.

        Parameters
        ----------
        survey: welleng.survey.Survey instance
            The well trajectory of the scenario being modelled.
        wellbore: welleng.architecture.WellBore instance
            The well bore architecture of the scenario being modelled.
        string: welleng.architecture.BHA or welleng.architecture.CasingString
        instance
            The string being run inside the well bore for the scenario being
            modelled.
        fluid_density: float
            The density (in SG) of the fluid in the well bore.
        name: str
            The name of the scenario being modeled.
        wob: float
            The compressive force (weight on bit) applied at the bottom of the
            string in N.
        tob: float
            The torque (torque on bit) applied at the bottom of the string in
            N.m.
        overpull: float
            The tension applied at the bottom of the string in N.
        """
        assert wellbore.complete, "Wellbore not complete"
        assert string.complete, "String not complete"

        self.survey_original = survey
        self.wellbore = wellbore
        self.string = string
        self.fluid_density = fluid_density
        self.name = name
        self.add_survey_points_from_strings()
        self.index = np.where(self.survey.md == self.string.bottom)[0][0] + 1

        self.get_buoyancy_factors()
        self.get_inc_average()
        self.get_inc_delta()
        self.get_azi_delta()
        self.get_weight_buoyed_and_radius()

        self.torque, self.tension = {}, {}
        self.get_coeff_friction_sliding()
        self.get_forces_and_torsion()
        if any((wob, tob, overpull)):
            self.get_forces_and_torsion(wob=wob, tob=tob, overpull=overpull)

    def add_survey_points_from_strings(self):
        """
        Check that there's survey stations for the top and bottoms of
        the string sections to ensure that the torque and drag is
        calculated for these key locations.
        """
        q = []

        for k, v in self.wellbore.sections.items():
            if v['bottom'] in self.survey_original.md:
                continue
            else:
                q.append(interpolate_md(
                    self.survey_original, v['bottom']
                ))

        for k, v in self.string.sections.items():
            if v['bottom'] in self.survey_original.md:
                continue
            else:
                q.append(interpolate_md(
                    self.survey_original, v['bottom']
                ))

        md, inc, azi = (
            self.survey_original.md.tolist(),
            self.survey_original.inc_rad.tolist(),
            self.survey_original.azi_grid_rad.tolist()
        )

        md.extend([station.md[-1] for station in q])
        inc.extend([station.inc_rad[-1] for station in q])
        azi.extend([station.azi_grid_rad[-1] for station in q])

        md, inc, azi = zip(*sorted(zip(md, inc, azi)))

        sh = self.survey_original.header
        sh.azi_reference = 'grid'

        self.survey = Survey(
            md=md, inc=inc, azi=azi, header=sh, deg=False
        )

    def get_buoyancy_factors(self):
        """
        Determine the buoyancy factor for each string section and add it
        to the string sections dict.
        """
        for k, v in self.string.sections.items():
            v['buoyancy_factor'] = buoyancy_factor(
                self.fluid_density, v['density']
            )

    def get_inc_average(self):
        self.inc_average = np.zeros_like(self.survey.inc_rad)
        self.inc_average[1:] = np.average(
            (self.survey.inc_rad[1:], self.survey.inc_rad[:-1]),
            axis=0
        )

    def get_inc_delta(self):
        self.inc_delta = np.zeros_like(self.survey.inc_rad)
        self.inc_delta[1:] = self.survey.inc_rad[1:] - self.survey.inc_rad[:-1]

    def get_azi_delta(self):
        self.azi_delta = np.zeros_like(self.survey.azi_grid_rad)
        self.azi_delta[1:] = (
            self.survey.azi_grid_rad[1:] - self.survey.azi_grid_rad[:-1]
        )

    def get_characteristic_od(self, section):
        if bool(self.string.sections[section].get('tooljoint_od')):
            return self.string.sections[section]['tooljoint_od']
        else:
            return self.string.sections[section]['od']

    def get_weight_buoyed_and_radius(self):
        section = 0
        weights = [0]
        diameter = [self.get_characteristic_od(section)]
        for md, delta_md in zip(self.survey.md[1:], self.survey.delta_md[1:]):
            if md > self.string.bottom:
                break
            while 1:
                if (
                    self.string.sections[section]['top']
                    < md <= self.string.sections[section]['bottom']
                ):
                    weights.append(
                        self.string.sections[section]['unit_weight'] * delta_md
                        * self.string.sections[section]['buoyancy_factor']
                    )
                    diameter.append(
                        self.get_characteristic_od(section)
                    )
                    break
                else:
                    section += 1
        self.weight_buoyed = np.array(weights)
        self.radius = np.array(diameter) / 2

    def get_coeff_friction_sliding(self):
        section = 0
        friction = [self.wellbore.sections[section]['coeff_friction_sliding']]
        for md, delta_md in zip(self.survey.md[1:], self.survey.delta_md[1:]):
            if md > self.wellbore.bottom:
                break
            while 1:
                if (
                    self.wellbore.sections[section]['top']
                    < md <= self.wellbore.sections[section]['bottom']
                ):
                    friction.append(
                        self.wellbore.sections[section]['coeff_friction_sliding']  # noqa: E501
                    )
                    break
                else:
                    section += 1
        self.coeff_friction_sliding = np.array(friction)

    def get_forces_and_torsion(self, wob=False, tob=False, overpull=False):
        if any((wob, tob)):
            assert tob, "Can't have WOB without TOB"
            assert wob, "Can't have TOB wihtouh WOB"
            ft = [np.array([0.0, -wob, -wob])]
            tn = [tob]
        else:
            ft = [np.zeros(3)]
            tn = [0]
        if overpull:
            ft[0][0] = overpull
        fn = []
        for row in zip(
            self.inc_average[:self.index][::-1],
            self.inc_delta[:self.index][::-1],
            self.azi_delta[:self.index][::-1],
            self.weight_buoyed[::-1],
            self.coeff_friction_sliding[:self.index][::-1],
            self.radius
        ):
            (
                inc_average, inc_delta, azi_delta, weight_buoyed,
                coeff_friction_sliding, radius
            ) = row

            fn.append(force_normal(
                ft[-1], inc_average, inc_delta, azi_delta,
                weight_buoyed
            ))
            ft.append(ft[-1] + np.array(force_tension_delta(
                weight_buoyed, inc_average, coeff_friction_sliding, fn[-1]
            )))
            tn.append(tn[-1] + np.array(torsion_delta(
                coeff_friction_sliding, fn[-1][2], radius
            )))

        fn = np.array(fn)[::-1]
        ft = np.array(ft)[::-1][1:]
        tn = np.array(tn)[::-1][1:]
        if wob:
            self.tension["drilling"] = ft[:, 2]
            self.tension["sliding"] = ft[:, 1]
        else:
            self.tension['slackoff'] = ft[:, 1]
            self.tension['rotating'] = ft[:, 2]

        if tob:
            self.torque['drilling'] = tn
        else:
            self.torque['rotating'] = tn

        if overpull:
            self.tension['overpull'] = ft[:, 0]
        else:
            self.tension['pickup'] = ft[:, 0]

        self.wob, self.tob, self.overpull = wob, tob, overpull

    def figure(self):
        return figure_string_tension_and_torque(self)


def force_normal(
    force_tension,
    inc_average,
    inc_delta,
    azi_delta,
    weight_buoyed,
):
    result = np.sqrt(
        (force_tension * azi_delta * np.sin(inc_average)) ** 2
        + (
            force_tension * inc_delta
            + weight_buoyed * np.sin(inc_average)
        ) ** 2
    )

    return result


def force_tension_delta(
    weight_buoyed,
    inc_average,
    coeff_friction_sliding,
    force_normal
):
    A = weight_buoyed * np.cos(inc_average)
    B = coeff_friction_sliding * force_normal

    pickup, slackoff, rotating = A + B * np.array([1, -1, 0])

    return (pickup, slackoff, rotating)


def torsion_delta(
    coeff_friction_sliding,
    force_normal,
    radius
):
    result = coeff_friction_sliding * force_normal * radius

    return result


def buoyancy_factor(fluid_density, string_density=7.85):
    """
    Parameters
    ----------
    fluid_density: float
        The density of the fluid in SG.
    string_density: float
        The density of the string, typically made from steel.

    Returns
    -------
    result: float
        The buoyancy factor when when multiplied against the string weight
        yields the bouyed string weight.
    """
    result = (string_density - fluid_density) / string_density

    return result


class HookLoad:
    def __init__(
        self, survey, wellbore, string, fluid_density, step=30, name=None,
        ff_range=(0.1, 0.4, 0.1)
    ):
        """
        A class for calculating the hookload or broomstick plot data for
        running or pulling a string in a wellbore.

        Parameters
        ----------
        survey: welleng.survey.Survey instance
            The well trajectory of the scenario being modelled.
        wellbore: welleng.architecture.WellBore instance
            The well bore architecture of the scenario being modelled.
        string: welleng.architecture.BHA or welleng.architecture.CasingString
        instance
            The string being run inside the well bore for the scenario being
            modelled.
        fluid_density: float
            The density (in SG) of the fluid in the well bore.
        step: float
            The measured depth step distance in meters to move the string.
        name: str
            The name of the scenario being modeled.
        ff_range: (3) tuple
            The start, stop and step for the range of friction factors to be
            used in the hookload calculations.
        """
        self.survey = survey
        self.wellbore = wellbore
        self.string = string
        self.fluid_density = fluid_density
        self.name = name
        self.step = step
        self.get_ff_range(ff_range)
        self.get_data()

    def get_ff_range(self, ff_range):
        self.ff_range = np.arange(*ff_range).tolist()
        self.ff_range.append(ff_range[1])

    def get_data(self):
        self.data = {}
        self.md_range = np.arange(
            self.string.top + self.step, self.string.bottom, self.step
        ).tolist()
        self.md_range.append(self.string.bottom)

        for ff in self.ff_range:
            wellbore_temp = deepcopy(self.wellbore)
            for k in wellbore_temp.sections.keys():
                wellbore_temp.sections[k]['coeff_friction_sliding'] = ff

            self.data[ff] = {}
            data_temp = []

            for md in self.md_range:
                bha_temp = self.string.depth(md)
                data_temp.append(
                    TorqueDrag(
                        self.survey, wellbore_temp, bha_temp,
                        fluid_density=self.fluid_density,
                        name=self.name
                    )
                )

            for t in data_temp[0].tension.keys():
                self.data[ff][t] = [
                    d.tension[t][0] for d in data_temp
                ]

    def figure(self):
        return figure_hookload(self)


def figure_string_tension_and_torque(
    td,
    units=dict(
        depth='ft',
        tension='lbf',
        torque='ft_lbf'
    )
):
    assert PLOTLY, "Please install plotly"

    fig = make_subplots(rows=1, cols=2)

    for k, v in td.tension.items():
        fig.add_trace(
            go.Scatter(
                x=((v * ureg('N')).to(units['tension'])).m,
                y=((td.survey.md * ureg.meters).to(units['depth'])).m,
                name=f"tension: {k}",
            ),
            row=1, col=1
        )

    for k, v, in td.torque.items():
        fig.add_trace(
            go.Scatter(
                x=((v * ureg('Nm')).to(units['torque'])).m,
                y=((td.survey.md * ureg.meters).to(units['depth'])).m,
                name=f"torque: {k}",
            ),
            row=1, col=2
        )

    fig.update_layout(
        title_text=(
            f"<b>wellbore:</b> {td.wellbore.name}<br>"
            + f"<b>string:</b> {td.string.name}"
        ),
        xaxis=dict(
            title=f"Tension ({units['tension']})"
        ),
        yaxis=dict(
            autorange='reversed',
            title=f"MD ({units['depth']})",
            exponentformat='none',
        ),
        xaxis2=dict(
            title=f"Torque ({units['torque']})"
        ),
        yaxis2=dict(
            autorange='reversed',
            title=f"MD ({units['depth']})",
            exponentformat='none',
        )
    )

    return fig


def figure_hookload(
    hl,
    units=dict(
        depth='ft',
        tension='lbf',
        torque='ft_lbf'
    )
):
    assert PLOTLY, "Please install plotly"

    fig = go.Figure()

    lines = [None, 'dashdot', 'dash', 'dot']

    md = ((hl.md_range * ureg.meters).to(units['depth'])).m
    annotations = []

    for i, (k, v) in enumerate(hl.data.items()):
        x = ((v['slackoff'] * ureg('N')).to(units['tension'])).m
        fig.add_trace(
            go.Scatter(
                x=x,
                y=md,
                name=f"SOFF: {k:.2f}",
                line=dict(color='blue', dash=lines[i])
            ),
        )
        annotations.append(
            dict(
                x=x[-1], y=md[-1],
                xanchor='center', yanchor='top',
                text=f'{k:.2f}',
                showarrow=False,
                font=dict(
                    color='blue'
                )
            )
        )

    for i, (k, v) in enumerate(hl.data.items()):
        fig.add_trace(
            go.Scatter(
                x=((v['rotating'] * ureg('N')).to(units['tension'])).m,
                y=md,
                name=f"RoffBFF: {k:.2f}",
                line=dict(color='green')
            )
        )
        break

    for i, (k, v) in enumerate(hl.data.items()):
        x = ((v['pickup'] * ureg('N')).to(units['tension'])).m
        fig.add_trace(
            go.Scatter(
                x=x,
                y=md,
                name=f"PUFF: {k:.2f}",
                line=dict(color='red', dash=lines[i])
            ),
        )
        annotations.append(
            dict(
                x=x[-1], y=md[-1],
                xanchor='center', yanchor='top',
                text=f'{k:.2f}',
                showarrow=False,
                font=dict(
                    color='red'
                )
            )
        )

    title_text = ("<b>wellbore:</b>")
    for i, (k, v) in enumerate(hl.wellbore.sections.items()):
        coupler = "" if i == 0 else "and "
        title_text += (
            f" {coupler}{v['name']}"
            + f" to {((v['bottom'] * ureg.meters).to(units['depth'])).m:.0f}"
            + f"{units['depth']}"
        )
    title_text += "<br><b>string:</b>"

    last = len(hl.string.sections.keys()) - 1
    for i, (k, v) in enumerate(list(hl.string.sections.items())[::-1]):
        coupler = "" if i == 0 else "and "
        if i == last:
            title_text += (
                f" {coupler}{v['name']}"
                + " to surface"
            )
        else:
            title_text += (
                f" {coupler}{v['name']}"
                + f" ({((v['length'] * ureg.meters).to(units['depth'])).m:.0f}"
                + f"{units['depth']})"
            )

    fig.update_layout(
        title_text=title_text,
        xaxis=dict(
            title=f"Hook-Load ({units['tension']})"
        ),
        yaxis=dict(
            autorange='reversed',
            title=f"MD ({units['depth']})",
            exponentformat='none',
        ),
        annotations=annotations
    )

    return fig
