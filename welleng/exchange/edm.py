import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import List, Union

import numpy as np
import requests

try:
    import networkx as nx
    NX = True
except ImportError:
    prNX = False


class EDM:
    def __init__(
            self,
            source_location: str,
            source: str = 'file'
    ):
        """
        Initiate an instance of an EDM object.

        Parameters
        ----------
            source_location: str
                The path and filename of the EDM file to be imported or the link to the XML file
            source: str ["file", "link"]
        """

        if source == "file":
            self.tree = ET.parse(source_location)
            self.root = self.tree.getroot()

        elif source == "link":
            response = requests.get(source_location)
            if response.status_code == 200:
                self.root = ET.fromstring(response.content)
            else:
                raise ValueError(f"Error loading the EDM file. Error code: {response.status_code}")
        else:
            raise AttributeError(f'Invalid source {source}. Source must be "file" or "link"')

        self.convert_attribute_lowercase()
        self._wellbore_id_to_name()
        self._wellbore_id_to_well_id()

    def get_attributes(
            self,
            tags: Union[str, List[str]] = None,
            attributes: dict = None,
            logic: str = 'AND'
    ) -> dict:
        """
        Get the attributes for the given tags in an EDM instance.

        Parameters
        ----------
            tags: str or list of str (default: None)
                The tag or list of tags you wish to return. The default will
                return all the tags that satisfy the given attributes.
            attributes: dict
                A dictionary of attribute keys and values to satisfy in the
                search of tags.
            logic: str (default: 'AND')
                Indicates whether the attributes should be all be satisfied
                ('AND') or if only one needs to be satisfied ('OR').

        Returns
        -------
            data: dict
                A dictionary of a list of dictionaries of tags and their
                attributes.
        """
        attributes, tags = self.check_data_and_initiate(attributes, tags, logic)
        data = {}
        for child in self.root:
            if child.tag in tags:
                if bool(attributes):
                    booleans = []

                    for k, v in attributes.items():
                        if not isinstance(v, list):
                            v = [v]
                        if k in child.attrib and child.attrib[k] in v:
                            booleans.append(True)
                        else:
                            booleans.append(False)
                        if bool(booleans):
                            if (
                                logic == 'AND' and all(booleans)
                                or logic == 'OR' and any(booleans)
                            ):
                                try:
                                    data[child.tag].append(child.attrib)
                                except KeyError:
                                    data[child.tag] = [child.attrib]
                else:
                    try:
                        data[child.tag].append(child.attrib)
                    except KeyError:
                        data[child.tag] = [child.attrib]
        return data

    def check_data_and_initiate(
            self,
            attributes: [dict, None],
            tags: Union[list, None],
            logic: str
    ):

        assert logic in ['AND', 'OR'], "logic must be 'AND' or 'OR'"
        if attributes is None:
            attributes = {}
        if tags is None:
            tags = self.get_tags()

        if not isinstance(tags, list):
            tags = [tags]

        return attributes, tags

    def _wellbore_id_to_name(self):
        """
        It's easier for the user to reference the name of the wellbore rather
        than the id, so let's make a couple of dictionaries to quickly
        perform lookups.
        """
        self.wellbore_id_to_name = {}
        self.wellbore_name_to_id = {}
        for _, v in self.get_wellbore_ids().items():
            self.wellbore_id_to_name[v['wellbore_id']] = v['wellbore_name']
            self.wellbore_name_to_id[v['wellbore_name']] = v['wellbore_id']

    def convert_attribute_lowercase(self):
        for tag in self.get_tags():
            for child in self.root:
                if child.tag == tag:
                    child.attrib = {k.lower(): v for k, v in child.attrib.items()}

    def get_wells(self):
        return {
            f"{child.attrib['well_common_name']}": child.attrib
            for child in self.root
            if child.tag == 'CD_WELL_ALL'
        }

    def get_sites(self):
        return {
            f"{child.attrib['site_name']}": child.attrib
            for child in self.root
            if child.tag == 'CD_SITE'
        }

    def get_wellbore_ids(self):
        return {
            f"{child.attrib['wellbore_name']}": child.attrib
            for child in self.root
            if child.tag == 'CD_WELLBORE'
        }

    def _wellbore_id_to_well_id(self):
        self.well_id_to_wellbore_id = {}
        for child in self.root:
            if child.tag == 'CD_WELLBORE':
                self.well_id_to_wellbore_id[child.attrib['wellbore_id']] = (
                    child.attrib['well_id']
                )

    def get_tags(self, sort=True) -> List[str]:
        tags = list(set([child.tag for child in self.root]))
        if sort:
            return sorted(tags)
        else:
            return tags

    def get_wellbore_graph(self):
        assert NX, "ImportError: try pip install welleng[all]"
        # get nodes
        G = nx.Graph()

        for _, v in self.get_wellbore_ids().items():
            if 'parent_wellbore_id' in v:
                G.add_edge(v['parent_wellbore_id'], v['wellbore_id'])
            else:
                G.add_node(v['wellbore_id'])

        return G

    def add_attributes(self, attributes, additional):
        if bool(additional):
            for k, v in additional.items():
                try:
                    attributes[k].extend(v)
                except KeyError:
                    attributes[k] = v
        return attributes

    def get_wellbore_data(self, wellbore_id):
        attributes = self._get_data(
            attributes={}, attr="wellbore_id", attr_id=wellbore_id
        )
        # some additional attributes are needed:
        well_id = attributes['CD_WELLBORE'][0]['well_id']
        tags = [
            'CD_WELL', 'CD_DATUM'
        ]
        additional = self.get_attributes(
            tags=tags, attributes={'well_id': well_id}
        )
        attributes = self.add_attributes(attributes, additional)

        # for the surveys, need to extract survey data from the parent wells
        predecessors = self.get_parents(wellbore_id)
        if bool(predecessors):
            tags = ['CD_SURVEY_HEADER', 'CD_SURVEY_STATION']
            additional = self.get_attributes(
                tags=tags, attributes={'wellbore_id': predecessors}
            )
        attributes = self.add_attributes(attributes, additional)

        return attributes

    def get_parents(self, wellbore_id, predecessors=None):

        if not predecessors:
            predecessors = []

        for child in self.root:
            if (
                child.tag == "CD_WELLBORE"
                and child.attrib['wellbore_id'] == wellbore_id
            ):
                if 'parent_wellbore_id' in child.attrib:
                    parent = child.attrib['parent_wellbore_id']
                    predecessors.append(parent)
                    predecessors = self.get_parents(parent, predecessors)
                else:
                    return predecessors
        return predecessors

    def _get_data(self, attributes, attr, attr_id):
        # get list of tags
        tags = self.get_tags()
        attributes = self.get_attributes(tags, attributes={attr: attr_id})

        return attributes

    def get_wellbore(self, wellbore, name=False):
        if name:
            wellbore_name = wellbore
            # look-up well_id in "CD_WELLS"
            wellbore_id = self.wellbore_name_to_id[wellbore_name]
        else:
            wellbore_id = wellbore
            wellbore_name = self.wellbore_id_to_name[wellbore_id]

        well = Well(
            wellbore_id,
            wellbore_name,
            self.get_wellbore_data(wellbore_id)
        )

        return well

    def get_case_name_from_id(self, case_id):
        for child in self.root:
            if (
                child.tag == 'CD_CASE'
                and child.attrib['case_id'] == case_id
            ):
                return child.attrib['case_name']


class Case(object):
    def __init__(self, case):
        for a, b in case.items():
            if isinstance(b, (list, tuple)):
                setattr(
                    self, a, [Case(x) if isinstance(x, dict) else x for x in b]
                )
            else:
                setattr(self, a, Case(b) if isinstance(b, dict) else b)


class Well:
    def __init__(self, wellbore_id, wellbore_name, well_data):
        self.wellbore_id = wellbore_id
        self.wellbore_name = wellbore_name
        self.well_data = well_data

    def _get_cases(self):
        return {
            c['case_id']: c['case_name']
            for c in self.well_data['CD_CASE']
        }

    def _set_attribute(
        self, case, label, tag_label=None, prefix="CD", suffix="id",
        out_label=None, **kwargs
    ):
        if tag_label is None:
            tag_label = label
        if out_label is None:
            out_label = label
        if hasattr(case, f"{label}_{suffix}"):
            setattr(
                case,
                out_label,
                self.get_data(
                    getattr(
                        case, f"{label}_{suffix}"
                    ),
                    label=label,
                    tag_label=tag_label,
                    prefix=prefix,
                    suffix=suffix
                )
            )
        return case

    def make_case(self, case_id):
        case = None
        for row in self.well_data['CD_CASE']:
            if row['case_id'] == case_id:
                case = Case(row)
        if case is not None:
            # the order of the list is important
            attributes = [
                {'label': 'wellbore'},
                {'label': 'fluid'},
                {'label': 'scenario'},
                {'label': 'well'},
                {'label': 'datum'},
                {'label': 'hole_sect_group'},
                {'label': 'assembly'},
                {'label': 'hole_sect', 'suffix': "group_id"},
                {'label': 'assembly', 'tag_label': 'assembly_comp'},
                {
                    'label': 'wellbore', 'tag_label': 'survey_header',
                    'out_label': 'survey_header'
                }
            ]
            # TODO could probably do this is a single loop?
            for k in attributes:
                case = self._set_attribute(
                    case, **k
                )
        self._get_surveys(case)
        self._get_ppfpt(case)

        return case

    def _get_ppfpt(self, case):
        key_words = ['PORE', 'FRAC', 'CD_TEMP']
        anti_key_words = ['TEMPLATE']
        for t in self.well_data:
            for kw in key_words:
                if kw in t:
                    for akw in anti_key_words:
                        if akw not in t:
                            data = self._sort_ppfpt_data(self.well_data[t], kw)
                            setattr(
                                case,
                                t.lower()[3:],
                                data
                            )

    @staticmethod
    def _sort_ppfpt_data(data, kw):
        lookup = {
            'PORE': 'pore_pressure',
            'FRAC': 'frac_gradient_pressure',
            'CD_TEMP': 'temperature'
        }
        kw = lookup[kw]
        sorted_data = {}
        sorted_sorted_data = {}
        for d in data:
            key = [k for k in d if "_group_id" in k][0]
            try:
                sorted_data[d[key]].append(d)
            except KeyError:
                sorted_data[d[key]] = [d]
        for item in sorted_data.items():
            try:
                sorted_sorted_data[item[0]] = sorted(
                    item[1], key=lambda k: float(k['tvd'])
                )
            except KeyError:
                return sorted_data
        return sorted_sorted_data

    def _get_surveys(self, case):  # noqa C901
        # get the surveys... turns out to be a bit of a PITA
        if hasattr(case, "survey_header"):
            survey_header_ids = [
                h['survey_header_id']
                for h in case.survey_header
            ]
            case.survey_station = {}
            for sh in survey_header_ids:
                case.survey_station[sh] = [
                    s for s in self.well_data['CD_SURVEY_STATION']
                    if s['survey_header_id'] == sh
                ]
            md = {
                sh['survey_header_id']: float(sh['md_max'])
                for sh in case.survey_header
            }
            survey_lists = {}
            for sh in survey_header_ids:
                survey_lists[sh] = self.get_parent_surveys(
                    sh,
                    data=OrderedDict({sh: md[sh]})
                )
            survey = {}
            for item in survey_lists.items():
                is_first = True
                for k, v in item[1].items():
                    if is_first:
                        survey[item[0]] = []
                        # current_survey_id = k
                        current_md = md[k]
                        is_first = False
                    current_survey_id = k
                    if float(v) < current_md:
                        for station in self.well_data['CD_SURVEY_STATION']:
                            if (
                                station['survey_header_id'] == current_survey_id
                                and float(station['md']) >= float(v)
                                and float(station['md']) <= current_md
                            ):
                                try:
                                    survey[item[0]].append(station)
                                except KeyError:
                                    survey[item[0]] = [station]
                        current_survey_id = k
                        current_md = float(v)
                survey[item[0]] = sorted(survey[item[0]], key=lambda k: float(k['md']))
        case.survey = survey

    def get_parent_surveys(self, survey_header_id, data=None):

        if not data:
            data = OrderedDict()

        for sh in self.well_data['CD_SURVEY_HEADER']:
            if (
                sh['survey_header_id'] == survey_header_id
                and 'tie_survey_header_id' in sh
            ):
                tsh_id = sh['tie_survey_header_id']
                data[survey_header_id] = sh['tie_on_depth']
                self.get_parent_surveys(tsh_id, data)
        return data

    def get_data(
        self, label_id, label, tag_label=None, prefix="CD", suffix="id"
    ):
        if tag_label is None:
            tag_label = label
        t = f"{prefix}_{tag_label}".upper()
        k = f"{label}_{suffix}"
        data = []
        for row in self.well_data[t]:
            if row[k] == label_id:
                data.append(row)
        if bool(data):
            return data
        else:
            return None

    def get_hole_sections(self):
        hole_sect_group_ids = []
        for k, v in self.well_data.items():
            hole_sect_group_ids.extend(
                [
                    p for row in v
                    for t, p in row.items()
                    if t == "hole_sect_group_id"
                ]
            )
        return list(set(hole_sect_group_ids))

    def _get_hole_section_data(self, group_id):
        hole_section_data = {}
        for k, v in self.well_data.items():
            for row in v:
                if (
                    "hole_sect_group_id" in row
                    and row["hole_sect_group_id"] == group_id
                ):
                    hole_section_data[k] = {}
                    for t, p in row.items():
                        hole_section_data[k][t] = p
        return hole_section_data

    def get_hole_section_data(self, hole_sections=None):
        """
        Parameters
        ----------
            hole_sections: list of str
        """
        hole_section_data = {}
        if hole_sections is None:
            hole_sections = self.get_hole_sections()
        for hs in hole_sections:
            hole_section_data[hs] = self._get_hole_section_data(hs)
        return hole_section_data

    def get_ppfpt_data(self):
        try:
            pp = self._get_ppfpt_data(
                self.well_data['CD_PORE_PRESSURE'],
                'pore_pressure_group_id',
                'pore_pressure'
            )
        except KeyError:
            pp = None
        try:
            fp = self._get_ppfpt_data(
                self.well_data['CD_FRAC_GRADIENT'],
                'frac_gradient_group_id',
                'frac_gradient_pressure'
            )
        except KeyError:
            fp = None
        try:
            tp = self._get_ppfpt_data(
                self.well_data['CD_TEMP_GRADIENT'],
                'temp_gradient_group_id',
                'temperature'
            )
        except KeyError:
            tp = None

        return {
            'pore_pressure': pp,
            'fracture_pressure': fp,
            'temperature': tp
        }

    @staticmethod
    def _get_ppfpt_data(data, *args):
        """
        Helper function for extracting tvd linked data.

        Parameters
        ----------
            data: extracted EDM arrtibute data
            *args: arg1, arg2
                arg1: string
                    key name for the data's group_id
                arg2: string
                    key_name for the required data

        Returns
        -------
            dict of (n, 2) arrays
        """
        arg1, arg2 = args
        temp = {}
        for p in data:
            if p[arg1] in temp:
                temp[p[arg1]].extend([
                    [
                        float(p['tvd']),
                        float(p[arg2])
                    ]
                ])
            else:
                temp[p[arg1]] = ([
                    [
                        float(p['tvd']),
                        float(p[arg2])
                    ]
                ])
        for group in temp:
            temp[group] = np.vstack(
                sorted(temp[group], key=lambda tup: tup[0])
            ).reshape(-1, 2)

        return temp


if __name__ == "__main__":

    # import EDM data
    FILENAME = 'data/Volve.xml'
    edm = EDM(FILENAME)

    well_ids = list(edm.get_wells())
    wb_ids = list(edm.get_wellbore_ids().keys())

    # name of the wellbore of interest
    well_name = well_ids[0]
    wellbore_name = wb_ids[0]

    # extract data for given wellbore name
    well = edm.get_wellbore(wellbore_name, name=True)

    case = well.make_case('3jcyt')

    hole_sections = well.get_hole_sections()
    hole_section_data = well.get_hole_section_data(hole_sections)

    G = edm.get_wellbore_graph()

    print("Done")
