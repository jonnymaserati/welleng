# welleng

[![Open Source Love svg2](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/jonnymaserati/welleng/blob/main/LICENSE)
[![PyPI version](https://img.shields.io/pypi/v/welleng)](https://pypi.org/project/welleng/)
[![Downloads](https://static.pepy.tech/personalized-badge/welleng?period=total&units=international_system&left_color=grey&right_color=orange&left_text=Downloads&kill_cache=1)](https://pepy.tech/project/welleng)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![welleng-tests Actions Status](https://github.com/jonnymaserati/welleng/workflows/welleng-tests/badge.svg)](https://github.com/jonnymaserati/welleng/actions)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.20968887-blue)](https://doi.org/10.5281/zenodo.20968887)

[welleng] is a collection of tools for Wells/Drilling Engineers, with a focus on well trajectory design and analysis.

## Features

- **Survey listings** — generate and interpolate well trajectories using minimum curvature or maximum curvature methods
- **Well bore uncertainty** — ISCWSA MWD Rev 5.11 error model (validated 35/35 sources against all three ISCWSA example workbooks), legacy Rev4 for back-compat, and OWSG **gyro** tool stacks (north-seeking stationary, mixed continuous, gyro-MWD) driven by the new ISCWSA JSON schema and an Excel-formula interpreter
- **Clearance & Separation Factors** — the standard ISCWSA separation rule (within 0.5% of ISCWSA test data), the **exact combined-ellipsoid Mahalanobis method** (`MahalanobisClearance` — same collision/clear verdicts as the rule but up to ~1.47× less conservative, analytic and mesh-free), and a mesh-based method using the [Flexible Collision Library]
- **Well path creation** — the `connector` module builds trajectories between start/end locations automatically, backed by an analytic, vectorized closed-form curve-hold-curve (CLC) point-to-target solver (`sawaryn_analytical` — every solution + the minimum-measured-depth path) with radius-sweep and R1×R2 drillable-region tooling
- **Vertical section, TVD interpolation, project-ahead** — common survey planning tools
- **Torque and drag** — simple torque/drag model with architecture module
- **Visualization** — interactive 3D via [vedo]/VTK or browser-based via plotly (requires `easy` install)
- **Data exchange** — import/export Landmark .wbp files; read EDM datasets
- **World Magnetic Model** — auto-calculates magnetic field data when not supplied

### Selecting an error model

`Survey`/`SurveyHeader` take an `error_model` name — **switch models by changing the
string**. The **default and recommended model is `"ISCWSA MWD Rev5.11"`**: the current
ISCWSA standard, validated 35/35 sources against all three ISCWSA example workbooks.
Omit `error_model` (leave it `None`) for no uncertainty calculation.

```python
import welleng as we

we.error.get_error_models()            # -> list every available model name

survey = we.survey.Survey(
    md, inc, azi, header=header,
    error_model="ISCWSA MWD Rev5.11",  # the standard; change this string to switch
)
cov = survey.err.errors.cov_NEVs       # NEV covariance per station
```

**Model families** (all selectable by name via `error_model=`):
- **Canonical ISCWSA MWD** — `"ISCWSA MWD Rev5.11"` (**default**, validated),
  `"ISCWSA MWD Rev4"` (legacy, to reproduce older results). (The older
  `"ISCWSA MWD Rev5"` name has been retired — use `"ISCWSA MWD Rev5.11"`.)
- **OWSG tool stacks** (Set A, JSON-driven) — the toolcode library:
  `"MWD+SRGM"`, `"MWD+SRGM+SAG"`, `"MWD+SRGM+AX"` (axial-interference correction),
  `"MWD+IFR1"` / `"MWD+IFR1+AX"` (in-field referencing), and the gyro stacks
  `"GYRO-NS"` (north-seeking stationary), `"GYRO-NS-CT"` (mixed continuous),
  `"GYRO-MWD"`. A `_Fl` suffix is the floating-rig variant.

> **Revisions.** welleng ships **Rev5-1 / Rev5.11** as the default. The older **OWSG
> Rev2** (2015) toolcodes can be regenerated from their workbooks with
> `python -m welleng.errors.tools.owsg_to_json`, but are not yet shipped as a
> selectable revision (tracked in `docs/dev/FUTURE_WORK.md`). Validate any model only
> against a **matching-revision** reference.

> **Error model update (welleng 0.10.0).** The MWD Rev 5 model has been brought
> into compliance with the ISCWSA Rev 5.11 example workbooks; select it as
> `"ISCWSA MWD Rev5.11"` (the older `"ISCWSA MWD Rev5"` name has since been
> retired). Its covariance differs slightly from welleng ≤ 0.9.x (the Rev 5.11
> corrections). `"ISCWSA MWD Rev4"` is unchanged for users who need to reproduce
> older results. See `welleng/errors/iscwsa_validate.py` for the validation harness
> used to audit against each ISCWSA example workbook.

> **Gyro support + JSON-driven tool models (welleng 0.11.0).** ISCWSA is
> moving its error-model standard from the legacy Excel workbooks to a
> machine-readable JSON schema at
> [`iscwsa/error-models`](https://github.com/iscwsa/error-models). welleng
> 0.11 ships ahead of that transition: a vendored copy of the schema
> (pinned to upstream SHA `c7af784` at
> `welleng/errors/iscwsa_schema/`), a ~95-line Excel-formula interpreter
> at `welleng/errors/interpreter.py`, and the full OWSG Set A library
> (~100 tool models, including the gyro stacks) generated as ISCWSA
> JSON at `welleng/errors/iscwsa_json/owsg_a/`. Use them via
> `Survey(error_model='GYRO-NS' | 'GYRO-NS-CT' | 'GYRO-MWD' | 'MWD+SRGM' | ...)`.
> See `examples/gyro_survey_example.py` for a side-by-side MWD/gyro
> position-uncertainty comparison.
>
> The legacy hand-coded MWD path is **untouched** — the canonical
> strings (`'ISCWSA MWD Rev4'`, `'ISCWSA MWD Rev5'`, `'ISCWSA MWD Rev5.11'`)
> still route through the production engine, so existing users see no
> behaviour change unless they opt in to a JSON-driven tool name.

> **Parallel-paths conformance harness.** Every term that exists in *both*
> the legacy hand-coded dispatcher and the new JSON+interpreter path is
> diff-tested at machine precision in CI (`tests/test_iscwsa_json_conformance.py`).
> The harness also catalogues schema gaps it surfaces — terms that don't
> evaluate against the current draft schema (cross-station references like
> `MDPrev` / `AzPrev` / `IncPrev`, per-tool calibration constants like
> `NoiseReductionFactor`). Run `python -m welleng.errors.conformance --summary`
> for the agreement matrix. Findings are filed upstream as schema-feedback
> on the ISCWSA Discussions board.

> **Absolute validation against SPE 90408.** The conformance harness above
> compares the two welleng paths (and so cancels any shared scale error); on
> top of that, the gyro weight functions are now checked against the
> *published* position covariances in SPE 90408-MS (Torkildsen et al. 2004)
> Appendix E — the first non-self-referential validation of welleng's gyro
> output. Example Models #1 (XY stationary), #3 (XY stationary→continuous
> hybrid), #5 (XYZ stationary→continuous) and #6 (XYZ stationary) reproduce
> the Appendix E covariances on ISCWSA Well #1 to within the paper's ±1%
> acceptance (the hybrid to ~0.1%), and Model #3 also on Well #2. See
> `tests/test_spe90408_appendix_e.py`. The propagation engine itself is
> independently exact — MWD `cov_NEVs` matches the ISCWSA reference to 5e-5.

> **Analytic curve-hold-curve solver (welleng 0.14.0).** The `Connector`'s CLC
> point-to-target case is now solved in closed form (Sawaryn 2021, SPE-204111-PA)
> rather than iteratively — `sawaryn_analytical` returns *every* solution and the
> minimum-measured-depth path in one batched eigenvalue solve (~0.02 ms/path),
> with a radius sweep that maps the feasible-radius trade-off and the R1×R2
> drillable region (fix one arc radius, read off the maximum usable other).
> **Breaking:** infeasible targets now raise rather than silently tightening below
> the design DLS; opt-in `on_infeasible='max_radius'` returns the gentlest feasible
> curve. Companion paper:
> [doi:10.5281/zenodo.21130979](https://doi.org/10.5281/zenodo.21130979).

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

## Citation

**Author:** Jonathan Corcutt — Corcutt Beheer B.V., Wassenaar, Netherlands · ORCID [0009-0008-1953-7760](https://orcid.org/0009-0008-1953-7760).

If welleng supports your work, please cite the **software** (the concept DOI always resolves to the latest version):

> Corcutt, J. *welleng: open-source well-engineering tools.* Zenodo. <https://doi.org/10.5281/zenodo.20968887>

```bibtex
@software{corcutt_welleng,
  author    = {Corcutt, Jonathan},
  title     = {welleng: open-source well-engineering tools},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20968887},
  url       = {https://doi.org/10.5281/zenodo.20968887}
}
```

If your work relies on the **gyro error-model validation** specifically, please also cite:

> Corcutt, J. (2026). *Reproducing the ISCWSA Gyro Error-Model Test Cases: Implementation Clarifications, Reference-Data Inconsistencies, and an Open-Source Reference Implementation.* Zenodo. <https://doi.org/10.5281/zenodo.20940515>

```bibtex
@misc{corcutt2026gyro,
  author    = {Corcutt, Jonathan},
  title     = {Reproducing the {ISCWSA} Gyro Error-Model Test Cases: Implementation Clarifications, Reference-Data Inconsistencies, and an Open-Source Reference Implementation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20940515},
  url       = {https://doi.org/10.5281/zenodo.20940515}
}
```

If your work uses the **anti-collision (exact Mahalanobis separation factor) method**, please also cite:

> Corcutt, J. (2026). *Making the Exact Wellbore Anti-Collision Boundary Practical: an Efficient, Validated, Open Implementation, and the Cost of the Separation-Rule Approximation.* Zenodo. <https://doi.org/10.5281/zenodo.20976872>

```bibtex
@misc{corcutt2026anticollision,
  author    = {Corcutt, Jonathan},
  title     = {Making the Exact Wellbore Anti-Collision Boundary Practical: an Efficient, Validated, Open Implementation, and the Cost of the Separation-Rule Approximation},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20976872},
  url       = {https://doi.org/10.5281/zenodo.20976872}
}
```

If your work uses the **analytical curve-hold-curve point-to-target solver** (`welleng.sawaryn_analytical`), please also cite:

> Corcutt, J. (2026). *An Open, Vectorized Closed-Form Solver for the 3D Curve-Hold-Curve Point-to-Target Problem.* Zenodo. <https://doi.org/10.5281/zenodo.21130979>

```bibtex
@misc{corcutt2026clc,
  author    = {Corcutt, Jonathan},
  title     = {An Open, Vectorized Closed-Form Solver for the 3D Curve-Hold-Curve Point-to-Target Problem},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21130979},
  url       = {https://doi.org/10.5281/zenodo.21130979}
}
```

**Built on:** welleng stands on NumPy, SciPy, pandas, trimesh, FCL and more, and implements published methods (ISCWSA, Brooks, Sawaryn, …) — see [CITATIONS.md](CITATIONS.md) for the dependency and method references to credit.

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
