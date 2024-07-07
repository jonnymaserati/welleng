# welleng

[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/pro-well-plan/pwptemp/blob/master/LICENSE.md)
[![PyPI version](https://badge.fury.io/py/welleng.svg)](https://badge.fury.io/py/welleng)
[![Downloads](https://static.pepy.tech/personalized-badge/welleng?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads&kill_cache=1)](https://pepy.tech/project/welleng)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![welleng-tests Actions Status](https://github.com/jonnymaserati/welleng/workflows/welleng-tests/badge.svg)](https://github.com/jonnymaserati/welleng/actions)

[welleng] aspires to be a collection of useful tools for Wells/Drilling Engineers, kicking off with a range of well trajectory tools.

  - Generate survey listings and interpolation with minimum curvature
  - Calculate well bore uncertainty data (utilizing either the [ISCWSA] MWD Rev5 models) - the coded error models are within 0.001% accuracy of the ISCWSA test data.
  - Calculate well bore clearance and Separation Factors (SF)
    - standard [ISCWSA] method within 0.5% accuracy of the ISCWSA test data.
    - new mesh based method using the [Flexible Collision Library].

## Support welleng
welleng is fuelled by copious amounts of coffee, so if you wish to supercharge development please donate generously: 

<a href="https://www.buymeacoffee.com/jonnymaserati" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" width="217px" ></a>

## Documentation
It's still very much in beta (or maybe even alpha), but there's [documentation] available to help use the modules.

## New Features!
  - **Documentation:** started to add [documentation] to try and make the library more accessible to starters.
  - **Maximum Curvature Method:** added an alternative `Survey` method for calculating a well trajectory from survey stations that add a more realistic (in terms of tortuosity) versus the traditional *minimum curvature method*. See [this post](https://jonnymaserati.github.io/2022/06/19/modified-tortuosity-index-survey-frequency.html) for more details.
  - **Modified Tortuosity Index:** added a `Survey` method for calculating a *modified tortuosity index* as described [here](https://jonnymaserati.github.io/2022/05/26/a-modified-tortuosity-index.html).
  - **Panel Plot:** added the `type='panel'` to the `Survey.figure()` method to return plan and section plots.
  - **Torque and Drag:** added a simple `torque_drag` module and an `architecture` module for creating scenarios (well bore and simple strings) - see this [post](https://jonnymaserati.github.io/2022/05/22/an-example-of-welleng-torque-drag.html) for instructions.
  - **Vertical Section:** this should have been included a long time ago, but finally a vertical section will be calculated if the `vertical_section_azimuth` parameter is included in the `SurveyHeader` when initiating a `Survey` instance. Otherwise, to return the vertical section for a given azimuth (e.g. 45 degrees), or to set the vertical section azimuth and add the vertical section data to the `Survey` then:
    ```python
    survey.get_vertical_section(45)
    survey.set_vertical_section(45)
    ```
  - **Hello version 0.4:** major version update to reflect all of the changes happening in the back end. If you have code that's built on previous versions of [welleng] then please lock that version in your env since likely it will require modifying to run with version 0.4 and higher.
  - **Project Ahead:** you can now project a survey from the last station to the bit or project to a target to see how to get back on track:
    ```terminal
    >>> node_bit = survey.project_to_bit(delta_md=9.0)
    >>> survey_to_target = survey.project_to_target(node_target, dls_design=3.0, delta_md=9.0)
    ```
  - **Interpolate Survey on TVD Depth:** new `survey` function for interpolating fixed TVD intervals along a [welleng] `Survey` instance, e.g. to interpolate `survey` every 10mTVD and return the interpolated survey as `s_interp_tvd`:
    ```terminal
    >>> s_interp_tvd = survey.interpolate_survey_tvd(step=10)
    ```
  - **OWSG Tool Error Models:** the ISCWSA curated Rev 4 and Rev 5 tool models have been coded up and continue to honor the ISCWSA diagnostic data. The OWSG tool errors are  ***experimental*** with the following status:
    - **Working**: MWD, SRGM, _Fl, SAG, IFR1, IFR2, EMS
    - **Not Currently Working Correctly**: AX, GYRO
    
    The available error models can be listed with the following command:
    ```terminal
    >>> welleng.errors.ERROR_MODELS
    ```
  - **World Magnetic Model Calculator:** calculates magnetic field data from the [World Magnetic Model](http://www.geomag.bgs.ac.uk/research/modelling/WorldMagneticModel.html) if magnetic field strength is not provided with the survey data.
  - **Import from Landmark .wbp files:** using the `exchange.wbp` module it's now possible to import .wbp files exported from Landmark's COMPASS or DecisionSpace software.
    ```python
    import welleng as we

    wp = we.exchange.wbp.load("demo.wbp")  # import file
    survey = we.exchange.wbp.wbp_to_survey(wp, step=30)  # convert to survey
    mesh = we.mesh.WellMesh(survey, method='circle')  # convert to mesh
    we.visual.plot(mesh.mesh)  # plot the mesh
    ```
  
  - **Export to .wbp files *(experimental)*:** using the `exchange.wbp` module, it's possible to convert a planned survey file into a list of turn points that can be exported to a .wbp file.

    ```python
    import welleng as we

    wp = we.exchange.wbp.WellPlan(survey)  # convert Survey to WellPlan object
    doc = we.exchange.wbp.export(wp)  # create a .wbp document
    we.exchange.wbp.save_to_file(doc, "demo.wbp")  # save the document to file
    ```
  
  - **Well Path Creation:** the addition of the `connector` module enables drilling well paths to be created simply by providing start and end locations (with some vector data like inclination and azimuth). No need to indicate *how* to connect the points, the module will figure that out itself.
  - **Fast visualization of well trajectory meshes:** addition of the `visual` module for quick and simple viewing and QAQC of well meshes.
  - **Mesh Based Collision Detection:** the current method for determining the Separation Factor between wells is constrained by the frequency and location of survey stations or necessitates interpolation of survey stations in order to determine if Anti-Collision Rules have been violated. Meshing the well bore interpolates between survey stations and as such is a more reliable method for identifying potential well bore collisions, especially wth more sparse data sets.
  - More coming soon!

## Tech

[welleng] uses a number of open source projects to work properly:

* [trimesh] - awesome library for loading and using triangular meshes.
* [Flexible Collision Library] - for fast collision detection.
* [numpy] - the fundamental package for scientific computing with Python.
* [scipy] - a Python-based ecosystem of open-source software for mathematics, science, and engineering.
* [vedo] - a python module for scientific visualization, analysis of 3D objects and point clouds based on VTK.
* [magnetic-field-calculator] - a Python API for the British Geological Survey magnetic field calculator.

## Simple Installation

A default, minimal [welleng] installation requires [numpy] and [scipy] which is sufficient for importing or generating trajectories with error models. Other libraries are optional depending on usage - most of [welleng]'s functionality can be unlocked with the `easy` install tag, but if you wish to use mesh collision functionality, then an advanced install is required using the `all` install tag to get [python-fcl], after first installing the compiled dependencies as described below.

You'll receive some `ImportError` messages and a suggested install tag if you try to use functions for which the required dependencies are missing.

### Default install with minimal dependencies:
```
pip install welleng
```
### Easy install with most of the dependencies and no compiled dependencies:
```
pip install welleng[easy]
```
## Advanced Installation
If you want to use the mesh collision detection method, then the compiled dependencies are required prior to installing all of the [welleng] dependencies.
### Ubuntu
Here's how to get the trickier dependencies manually installed on Ubuntu (further instructions can be found [here](https://github.com/flexible-collision-library/fcl/blob/master/INSTALL)):

```terminal
sudo apt-get update
sudo apt-get install libeigen3-dev libccd-dev octomap-tools
```
On a Mac you should be able to install the above with brew and on a Windows machine you'll probably have to build these libraries following the instruction in the link, but it's not too tricky. Once the above are installed, then it should be a simple:

```terminal
pip install welleng[all]
```

For developers, the repository can be cloned and locally installed in your GitHub directory via your preferred Python env (the `dev` branch is usuall a version or two ahead of the `main`).

```terminal
git clone https://github.com/jonnymaserati/welleng.git
cd welleng
pip install -e .[all]
```

Make sure you include that `.` in the final line (it's not a typo) as this ensures that any changes to your development version are immediately implemented on save.

### Windows
Detailed instructions for installing [welleng] in a Windows OS can be found in this [post](https://jonnymaserati.github.io/2021/05/11/install-welleng-windows.html).

### Colaboratory
Perhaps the simplest way of getting up and running with [welleng] is to with a [colab notebook](https://colab.research.google.com/notebooks/intro.ipynb). The required dependencies can be installed with the following cell:

```python
!apt-get install -y xvfb x11-utils libeigen3-dev libccd-dev octomap-tools
!pip install welleng[all]
```
Unfortunately the visualization doesn't work with colab (or rather I've not been able to embed a VTK object) so some further work is needed to view the results. However, the [welleng] engine can be used to generate data in the notebook. Test it out with the following code:

```python
!pip install plotly jupyter-dash pint
!pip install -U git+https://github.com/Kitware/ipyvtk-simple.git

import welleng as we
import plotly.graph_objects as go
from jupyter_dash import JupyterDash


# create a survey
s = we.survey.Survey(
    md=[0., 500., 2000., 5000.],
    inc=[0., 0., 30., 90],
    azi=[0., 0., 30., 90.]
)

# interpolate survey - generate points every 30 meters
s_interp = s.interpolate_survey(step=30)

# plot the results
fig = go.Figure()
fig.add_trace(
    go.Scatter3d(
        x=s_interp.x,
        y=s_interp.y,
        z=s_interp.z,
        mode='lines',
        line=dict(
            color='blue'
        ),
        name='survey_interpolated'
    ),
)

fig.add_trace(
    go.Scatter3d(
        x=s.x,
        y=s.y,
        z=s.z,
        mode='markers',
        marker=dict(
            color='red'
        ),
        name='survey'
    )
)
fig.update_scenes(zaxis_autorange="reversed")
fig.show()
```

## Quick Start

Here's an example using `welleng` to construct a couple of simple well trajectories with the `connector` module, creating survey listings for the wells with well bore uncertainty data, using these surveys to create well bore meshes and finally printing the results and plotting the meshes with the closest lines and SF data.

```python
import welleng as we
from tabulate import tabulate


# construct simple well paths
print("Constructing wells...")
connector_reference = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0.,
        azi1=0.,
        pos2=[-100., 0., 2000.],
        inc2=90,
        azi2=60,
    ),
    step=50
)

connector_offset = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.],
        inc1=0.,
        azi1=225.,
        pos2=[-280., -600., 2000.],
        inc2=90.,
        azi2=270.,
    ),
    step=50
)

# make survey objects and calculate the uncertainty covariances
print("Making surveys...")
sh_reference = we.survey.SurveyHeader(
    name="reference",
    azi_reference="grid"
)
survey_reference = we.survey.Survey(
    md=connector_reference.md,
    inc=connector_reference.inc_deg,
    azi=connector_reference.azi_grid_deg,
    header=sh_reference,
    error_model='ISCWSA MWD Rev4'
)
sh_offset = we.survey.SurveyHeader(
    name="offset",
    azi_reference="grid"
)
survey_offset = we.survey.Survey(
    md=connector_offset.md,
    inc=connector_offset.inc_deg,
    azi=connector_offset.azi_grid_deg,
    start_nev=[100., 200., 0.],
    header=sh_offset,
    error_model='ISCWSA MWD Rev4'
)

# generate mesh objects of the well paths
print("Generating well meshes...")
mesh_reference = we.mesh.WellMesh(
    survey_reference
)
mesh_offset = we.mesh.WellMesh(
    survey_offset
)

# determine clearances
print("Setting up clearance models...")
c = we.clearance.Clearance(
    survey_reference,
    survey_offset
)

print("Calculating ISCWSA clearance...")
clearance_ISCWSA = we.clearance.IscwsaClearance(
    survey_reference, survey_offset
)

print("Calculating mesh clearance...")
clearance_mesh = we.clearance.MeshClearance(
    survey_reference, survey_offset, sigma=2.445
)

# tabulate the Separation Factor results and print them
results = [
    [md, sf0, sf1]
    for md, sf0, sf1
    in zip(c.reference.md, clearance_ISCWSA.sf, clearance_mesh.sf)
]

print("RESULTS\n-------")
print(tabulate(results, headers=['md', 'SF_ISCWSA', 'SF_MESH']))

# get closest lines between wells
lines = we.visual.get_lines(clearance_mesh)

# plot the result
plot = we.visual.Plotter()
plot.add(mesh_reference, c='red')
plot.add(mesh_offset, c='blue')
plot.add(lines)
plot.show()
plot.close()

print("Done!")

```

This results in a quick, interactive visualization of the well meshes that's great for QAQC. What's interesting about these results is that the ISCWSA method does not explicitly detect a collision in this scenario wheras the mesh method does.

![image](https://user-images.githubusercontent.com/41046859/102106351-c0dd1a00-3e30-11eb-82f0-a0454dfce1c6.png)

For more examples, including how to build a well trajectory by joining up a series of sections created with the `welleng.connector` module (see pic below), check out the [examples] and follow the [jonnymaserati] blog.

![image](https://user-images.githubusercontent.com/41046859/102206410-d56ef000-3ecc-11eb-9f1a-b2a6b45fe479.png)

Well trajectory generated by [build_a_well_from_sections.py]

## Todos

 - Add a `Target` class to see what you're aiming for - **in progress**
 - Documentation - **in progress**
 - Generate a scene of offset wells to enable fast screening of collision risks (e.g. hundreds of wells in seconds)
 - WebApp for those that just want answers
 - Add a `units` module to handle any units system - **in progress**

It's possible to generate data for visualizing well trajectories with [welleng], as can be seen with the rendered scenes below.
![image](https://user-images.githubusercontent.com/41046859/97724026-b78c2e00-1acc-11eb-845d-1220219843a5.png)
ISCWSA Standard Set of Well Paths

The ISCWSA standard set of well paths for evaluating clearance scenarios have been rendered in [blender] above. See the [examples] for the code used to generate a [volve] scene, extracting the data from the [volve] EDM.xml file.

## License

[Apache 2.0](LICENSE)

Please note the terms of the license. Although this software endeavors to be accurate, it should not be used as is for real wells. If you want a production version or wish to develop this software for a particular application, then please get in touch with [jonnycorcutt], but the intent of this library is to assist development.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [jonnycorcutt]: <mailto:jonnycorcutt@gmail.com>
   [welleng]: <https://github.com/jonnymaserati/welleng>
   [Flexible Collision Library]: <https://github.com/flexible-collision-library/fcl>
   [trimesh]: <https://github.com/mikedh/trimesh>
   [python-fcl]: <https://github.com/BerkeleyAutomation/python-fcl>
   [vedo]: <https://github.com/marcomusy/vedo>
   [numpy]: <https://numpy.org/>
   [scipy]: <https://www.scipy.org/>
   [examples]: <https://github.com/jonnymaserati/welleng/tree/main/examples>
   [blender]: <https://www.blender.org/>
   [volve]: <https://www.equinor.com/en/how-and-why/digitalisation-in-our-dna/volve-field-data-village-download.html>
   [ISCWSA]: <https://www.iscwsa.net/>
   [build_a_well_from_sections.py]: <https://github.com/jonnymaserati/welleng/tree/main/examples/build_a_well_from_sections.py>
   [magnetic-field-calculator]: <https://pypi.org/project/magnetic-field-calculator/>
   [jonnymaserati]: <https://jonnymaserati.github.io/>
   [documentation]: <https://jonnymaserati.github.io/welleng/>
