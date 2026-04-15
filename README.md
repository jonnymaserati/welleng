# welleng

[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/jonnymaserati/welleng/blob/main/LICENSE)
[![PyPI version](https://badge.fury.io/py/welleng.svg)](https://badge.fury.io/py/welleng)
[![Downloads](https://static.pepy.tech/personalized-badge/welleng?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads&kill_cache=1)](https://pepy.tech/project/welleng)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![welleng-tests Actions Status](https://github.com/jonnymaserati/welleng/workflows/welleng-tests/badge.svg)](https://github.com/jonnymaserati/welleng/actions)

[welleng] is a collection of tools for Wells/Drilling Engineers, with a focus on well trajectory design and analysis.

## Features

- **Survey listings** — generate and interpolate well trajectories using minimum curvature or maximum curvature methods
- **Well bore uncertainty** — ISCWSA MWD Rev 5.11 error model (validated 35/35 sources against all three ISCWSA example workbooks) and legacy Rev4 for back-compat; OWSG models also available
- **Clearance & Separation Factors** — standard ISCWSA method (within 0.5% of ISCWSA test data) and mesh-based method using the [Flexible Collision Library]
- **Well path creation** — the `connector` module builds trajectories between start/end locations automatically
- **Vertical section, TVD interpolation, project-ahead** — common survey planning tools
- **Torque and drag** — simple torque/drag model with architecture module
- **Visualization** — interactive 3D via [vedo]/VTK or browser-based via plotly (requires `easy` install)
- **Data exchange** — import/export Landmark .wbp files; read EDM datasets
- **World Magnetic Model** — auto-calculates magnetic field data when not supplied

Available error models:
```python
import welleng as we
we.error.get_error_models()
```

> **Error model update (welleng 0.10.0).** The MWD Rev 5 model has been brought
> into compliance with the ISCWSA Rev 5.11 example workbooks. The
> `"ISCWSA MWD Rev5"` string remains a selectable alias with a
> `DeprecationWarning` pointing at the new `"ISCWSA MWD Rev5.11"` name, but
> produces the corrected Rev 5.11 covariance (slightly different numerical
> output to welleng ≤ 0.9.x). `"ISCWSA MWD Rev4"` is unchanged for users who
> need to reproduce older results. See `welleng/errors/iscwsa_validate.py` for
> the validation harness used to audit against each ISCWSA example workbook.

## Support welleng
welleng is fuelled by copious amounts of coffee, so if you wish to supercharge development please donate generously:

<a href="https://www.buymeacoffee.com/jonnymaserati" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/arial-yellow.png" alt="Buy Me A Coffee" width="217px" ></a>

## Cloud API

A hosted API for 3D well path planning is available at
[welleng.org](https://api.welleng.org/api/docs). Solve CLC (curve-line-curve)
paths via simple REST calls — no local install, no GPU required.

- Batch solving (up to 100K pairs)
- GPU-accelerated
- Free tier available

See the [interactive docs](https://api.welleng.org/api/docs) to try it out.

## Documentation

[Documentation] is available, though the library evolves quickly so the examples directory is often the best reference.

## Tech

[welleng] uses a number of open source projects:

* [trimesh] — loading and using triangular meshes
* [Flexible Collision Library] — fast collision detection
* [numpy] — scientific computing
* [scipy] — mathematics, science, and engineering
* [vedo] — 3D visualization based on VTK
* [magnetic-field-calculator] — BGS magnetic field calculator API

## Installation

The default install includes core dependencies (numpy, scipy, pandas, etc.) and covers survey generation, error models, and trajectory design. The `easy` extras add 3D visualization (vedo/VTK), magnetic field lookup, network analysis, and mesh import. The `all` extras add mesh-based collision detection, which requires compiled dependencies.

You'll receive an `ImportError` with a suggested install tag if a required optional dependency is missing.

### Default install (core functionality, no visualization)
```
pip install welleng
```

### Easy install (recommended — adds 3D visualization, magnetic field calculator, trimesh, networkx)
```
pip install welleng[easy]
```

### Full install (adds mesh collision detection — requires compiled dependencies)

First install the compiled dependencies. On Ubuntu:
```terminal
sudo apt-get update
sudo apt-get install libeigen3-dev libccd-dev octomap-tools
```
On macOS, use `brew`. On Windows, follow the [FCL install instructions](https://github.com/flexible-collision-library/fcl/blob/master/INSTALL). Then:
```terminal
pip install welleng[all]
```

### Developer install

The project uses [uv](https://github.com/astral-sh/uv) for dependency management:
```terminal
git clone https://github.com/jonnymaserati/welleng.git
cd welleng
uv sync --all-extras
```

Or with plain pip:
```terminal
pip install -e .[all]
```

### Windows
On Windows, `pip install welleng` should work for the default and easy installs. For the full install with mesh collision detection, follow the [FCL install instructions](https://github.com/flexible-collision-library/fcl/blob/master/INSTALL) to set up the compiled dependencies first.

### Colaboratory

For Google Colab, install dependencies with:
```python
!apt-get install -y libeigen3-dev libccd-dev octomap-tools
!pip install welleng[easy] plotly
```

The VTK-based 3D viewer doesn't work in Colab, but plotly does. Here's a quick example:

```python
import welleng as we
import plotly.graph_objects as go

# create a survey
s = we.survey.Survey(
    md=[0., 500., 2000., 5000.],
    inc=[0., 0., 30., 90],
    azi=[0., 0., 30., 90.]
)

# interpolate every 30 m
s_interp = s.interpolate_survey(step=30)

fig = go.Figure()
fig.add_trace(go.Scatter3d(
    x=s_interp.e, y=s_interp.n, z=s_interp.tvd,
    mode='lines', name='interpolated'
))
fig.add_trace(go.Scatter3d(
    x=s.e, y=s.n, z=s.tvd,
    mode='markers', marker=dict(color='red'), name='survey stations'
))
fig.update_scenes(zaxis_autorange="reversed")
fig.show()
```

## Quick Start

Build a pair of well trajectories, compute error ellipses and clearance, and visualize (requires `pip install welleng[all]` for mesh clearance and visualization):

```python
import welleng as we

# construct well paths
connector_reference = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.], inc1=0., azi1=0.,
        pos2=[-100., 0., 2000.], inc2=90, azi2=60,
    ),
    step=50
)
connector_offset = we.survey.from_connections(
    we.connector.Connector(
        pos1=[0., 0., 0.], inc1=0., azi1=225.,
        pos2=[-280., -600., 2000.], inc2=90., azi2=270.,
    ),
    step=50
)

# create surveys with error models
survey_reference = we.survey.Survey(
    md=connector_reference.md,
    inc=connector_reference.inc_deg,
    azi=connector_reference.azi_grid_deg,
    header=we.survey.SurveyHeader(name="reference", azi_reference="grid"),
    error_model='ISCWSA MWD Rev4'
)
survey_offset = we.survey.Survey(
    md=connector_offset.md,
    inc=connector_offset.inc_deg,
    azi=connector_offset.azi_grid_deg,
    start_nev=[100., 200., 0.],
    header=we.survey.SurveyHeader(name="offset", azi_reference="grid"),
    error_model='ISCWSA MWD Rev4'
)

# build well meshes
mesh_reference = we.mesh.WellMesh(survey_reference)
mesh_offset = we.mesh.WellMesh(survey_offset)

# calculate clearance
clearance_ISCWSA = we.clearance.IscwsaClearance(survey_reference, survey_offset)
clearance_mesh = we.clearance.MeshClearance(survey_reference, survey_offset, sigma=2.445)

# print minimum SF
print(f"Min SF (ISCWSA): {min(clearance_ISCWSA.sf):.2f}")
print(f"Min SF (mesh):   {min(clearance_mesh.sf):.2f}")

# visualize
lines = we.visual.get_lines(clearance_mesh)
plot = we.visual.Plotter()
plot.add(mesh_reference, c='red')
plot.add(mesh_offset, c='blue')
plot.add(lines)
plot.show()
plot.close()
```

This results in a quick, interactive visualization of the well meshes. What's interesting about these results is that the ISCWSA method does not explicitly detect a collision in this scenario whereas the mesh method does.

![image](https://user-images.githubusercontent.com/41046859/102106351-c0dd1a00-3e30-11eb-82f0-a0454dfce1c6.png)

For more examples, including how to build a well trajectory by joining up a series of sections created with the `welleng.connector` module (see pic below), check out the [examples] and follow the [jonnymaserati] blog.

![image](https://user-images.githubusercontent.com/41046859/102206410-d56ef000-3ecc-11eb-9f1a-b2a6b45fe479.png)

Well trajectory generated by [build_a_well_from_sections.py]

It's possible to generate data for visualizing well trajectories with [welleng], as can be seen with the rendered scenes below.
![image](https://user-images.githubusercontent.com/41046859/97724026-b78c2e00-1acc-11eb-845d-1220219843a5.png)
ISCWSA Standard Set of Well Paths

The ISCWSA standard set of well paths for evaluating clearance scenarios have been rendered in [blender] above. See the [examples] for the code used to generate a [volve] scene, extracting the data from the [volve] EDM.xml file.

## License

[Apache 2.0](LICENSE)

Please note the terms of the license. Although this software endeavors to be accurate, it should not be used as is for real wells. If you want a production version or wish to develop this software for a particular application, then please get in touch with [jonnycorcutt], but the intent of this library is to assist development.

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job.)

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
