import pint
import yaml
import os
import numpy as np

PATH = os.path.dirname(__file__)
UNIT_SYSTEMS_FILENAME = os.path.join(
    '', *[PATH, 'units', 'unit_systems.yaml']
)
UNIT_REG_FILENAME = os.path.join(
    '', *[PATH, 'units', 'unit_reg.yaml']
)

with open(UNIT_SYSTEMS_FILENAME, 'r') as f:
    unit_systems = yaml.safe_load(f)
    parameters = unit_systems.pop('parameters')

with open(UNIT_REG_FILENAME, 'r') as f:
    unit_reg = yaml.safe_load(f)

ureg = pint.UnitRegistry()

for k, v in unit_reg.items():
    ureg.define(f'{k} = {v}')


class Units:
    def __init__(self, unit_system=None, **kwargs):
        if unit_system is not None:
            assert unit_system in unit_systems.keys(), (
                "Unrecognized unit system"
            )
        else:
            unit_system = 'default'

        for k, v in unit_systems[unit_system].items():
            setattr(self, k, v)

        for k, v in kwargs.items():
            if k == 'name':
                continue
            assert k in parameters.keys(), f"Invalid group {k}"

            # check if the value is in the registry - will return a
            # pint.errors.UndefinedUnitError if not.
            ureg[v]

            setattr(self, k, v)

        for k, v in self.__dict__.items():
            assert v is not None, f"Missing unit {k}"

    def update(self, group, value, name=None):
        """
        Method for updating a unit group value that checks the validity
        of the update.
        """
        assert group in self.__dict__.keys(), f"Invalid group {group}"
        ureg[value]

        if name is None:
            if self.name in unit_systems.keys():
                self.name = name
        elif name not in unit_systems.keys():
            if name in unit_systems.keys():
                self.name = None
            else:
                self.name = name

        setattr(self, group, value)


def extract_cov_nev_data(survey):
    cov_nev = {}
    nn, ne, nv, _, ee, ev, _, _, vv = survey.cov_nev.reshape(-1, 9).T

    cov_nev = {
        'nn': nn, 'ne': ne, 'nv': nv, 'ee': ee, 'ev': ev, 'vv': vv
    }

    return cov_nev


def extract_cov_hla_data(survey):
    cov_hla = {}
    hh, hl, ha, _, ll, la, _, _, aa = survey.cov_hla.reshape(-1, 9).T

    cov_hla = {
        'hh': hh, 'hl': hl, 'ha': ha, 'll': ll, 'la': la, 'aa': aa
    }

    return cov_hla


def survey_to_dict(survey, units=None):
    if units is None:
        units = survey.header.units
    elif not isinstance(units, Units):
        units = Units(units)

    survey_dict = survey.__dict__

    temp = {
        k: v.astype(float) for k, v in survey_dict.items()
        if isinstance(v, np.ndarray)
        and v.shape[-1] != 3
    }

    keys_to_delete = []
    keys_to_change = {}
    for k in temp.keys():
        if k.split('_')[-1] == 'rad':
            keys_to_delete.append(k)
        if k.split('_')[-1] == 'deg':
            keys_to_change[k] = ('_').join(k.split('_')[:-1])

    for k in keys_to_delete:
        del temp[k]

    for k, v in keys_to_change.items():
        temp[v] = temp.pop(k)

    temp['cov_nev'] = (
        None if survey.cov_nev is None
        else extract_cov_nev_data(survey)
    )

    temp['cov_hla'] = (
        None if survey.cov_hla is None
        else extract_cov_hla_data(survey)
    )

    temp_units = {}

    def process_units(k, v, temp_units, n=1):
        flag = False
        for a, b in parameters.items():
            if a == 'name':
                continue
            if flag:
                break
            if k in b:
                temp_units[k] = np.round(
                    (
                        v * ureg(f"{unit_systems['backend'][a]} ** {n}")
                    ).to(f'{getattr(units, a)} ** {n}').m,
                    2
                )
                flag = True
                break

        return temp_units, flag

    # assign default units to parameters in the dict
    for k, v in temp.items():
        if type(v) is dict:
            for kk, vv in v.items():
                n = 2 if 'cov' in k.split('_') else 1
                temp_units, flag = process_units(kk, vv, temp_units, n)
        else:
            temp_units, flag = process_units(k, v, temp_units)
        if not flag:
            temp_units[k] = v

    survey_dict_new = {}

    # TODO update the units and header to align with the wellpath units
    survey_dict_new['header'] = survey.header.__dict__

    survey_dict_new['wellpath'] = {
        i: dict(zip(temp_units, col))
        for i, col in enumerate(zip(*temp_units.values()))
    }

    return survey_dict_new
