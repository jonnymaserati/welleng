PARAMS = {
    'WellBore': ['id', 'coeff_friction_sliding']
}


class String:
    def __init__(
            self,
            name: str,
            top: float,
            bottom: float,
            *args,
            method: str = "bottom_up",
            **kwargs
    ):
        """
        A generic well bore architecture collection, e.g. a casing string
        made up a a number of different lengths of weights and grades.

        Parameters
        ----------
        name: str
            The name of the collection.
        top: float
            The shallowest measured depth at the top of the collection of
            items in meters.
        bottom: float
            The deepest measured depth at the bottom of the collection of
            items in meters.
        method: string (default: 'bottom up')
            The method in which items are added to the collection, either
            'bottom up' starting from the deepest element and adding items
            above, else 'top down' starting from the shallowest item and
            adding items below.
        """
        self.name = name
        self.top = top
        self.bottom = bottom
        self.sections = {}
        self.complete = False

        assert method in {"top_down", "bottom_up"}, "Unrecognized method"
        self.method = method

    def depth(self, md: float):
        assert self.top < md <= self.bottom, "Depth out of range"

        string_new = String(
            self.name, self.top, md, method="bottom_up"
        )

        reached_top = False
        for section in reversed(list(self.sections.keys())):
            if reached_top:
                break

            params = {
                k: v for k, v in self.sections[section].items()
                if k not in ['top', 'bottom', 'length', 'buoyancy_factor']
            }
            string_new.add_section(
                length=self.sections[section]['length'], **params
            )

            if string_new.sections[0]['top'] == self.top:
                reached_top = True

        return string_new

    def add_section(self, **kwargs):
        if type(self).__name__ == 'WellBore':
            for param in PARAMS.get('WellBore'):
                assert param in kwargs.keys(), f"Missing parameter {param}"

        elif type(self).__name__ == 'BHA':
            kwargs['density'] = kwargs.get('density', 7.85)

        if self.method == "top_down":
            self.add_section_top_down(**kwargs)
        elif self.method == "bottom_up":
            self.add_section_bottom_up(**kwargs)

    def add_section_top_down(self, **kwargs):
        """
        Sections built from the top down until the bottom of the bottom section
        is equal to the defined string bottom.
        """
        if bool(self.sections) is False:
            temp = 0
            top = self.top
        else:
            temp = len(self.sections)
            top = self.sections[temp - 1]['bottom']

        self.sections[temp] = {}
        self.sections[temp]['top'] = top

        # add the section to the sections dict
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # sort the dict on depth of tops
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(), key=lambda item: item[1]['top']
            )
        }

        # re-index the keys
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # check inputs
        for k, v in temp.items():
            if k == 0:
                assert v['top'] == self.top
            else:
                assert v['top'] == temp[k - 1]['bottom']
            assert v['bottom'] <= self.bottom

        if temp[len(temp) - 1]['bottom'] == self.bottom:
            self.complete = True

        self.sections = temp

    def add_section_bottom_up(
        self, **kwargs
    ):
        """
        Sections built from the bottom up until the top of the top
        section is equal to the defined string top.

        Default is to extend the section to the top of the String as
        defined in the String.top property (when length = top = None).

        Parameters:
        -----------

        """
        if bool(self.sections) is False:
            temp = 0
            bottom = self.bottom
        else:
            temp = len(self.sections)
            bottom = self.sections[0]['top']

        if bool(kwargs.get('length')):
            top = bottom - kwargs.get('length')
            if top < self.top:
                top = self.top
                length = bottom - top
        elif bool(kwargs.get('top')):
            length = bottom - kwargs.get('top')
        else:
            top = self.top
            length = self.sections[0]['top']

        self.sections[temp] = {}
        self.sections[temp]['top'] = top
        self.sections[temp]['bottom'] = bottom
        self.sections[temp]['length'] = length

        # add the section to the sections dict
        for k, v in kwargs.items():
            self.sections[temp][k] = v

        # sort the dict on depth of tops
        self.sections = {
            k: v
            for k, v in sorted(
                self.sections.items(), key=lambda item: item[1]['bottom']
            )
        }

        # re-index the keys
        temp = {}
        for i, (k, v) in enumerate(self.sections.items()):
            temp[i] = v

        # check inputs
        number_of_sections = len(temp)
        for k, v in temp.items():
            if k == number_of_sections - 1:
                assert v['bottom'] == self.bottom
            else:
                assert v['bottom'] == temp[k + 1]['top']
            assert v['top'] >= self.top

        if temp[0]['top'] == self.top:
            self.complete = True

        self.sections = temp


class WellBore(String):
    def __init__(self, *args, **kwargs):
        """
        Inherits from `String` class, but this makes it more intuitive.
        """
        super().__init__(*args, **kwargs)


class BHA(String):
    def __init__(self, *args, **kwargs):
        """
        Inherits from `String` class, but this makes it more intuitive.
        """
        super().__init__(*args, **kwargs)
