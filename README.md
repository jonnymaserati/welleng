# welleng
[![PyPI version](https://badge.fury.io/py/welleng.svg)](https://badge.fury.io/py/welleng)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0)

[welleng] aspires to be a collection of useful tools for Wells/Drilling Engineers, kicking off with a range of well trajectory analysis tools.

  - Generate survey listings and interpolation with minimum curvature
  - Calculate well bore uncertainty data (currently utilizing the ISCWSA MWD Rev4 model)
  - Calculate well bore clearance and Separation Factors (SF)
    - standard ISCWSA method
    - new mesh based method using the [Flexible Collision Library]

## New Features!

  - **Mesh Based Collision Detection:** the current method for determining the Separation Factor between wells is constrained by the frequency and location of survey stations or necessitates interpolation of survey stations in order to determine if Anti-Collision Rules have been violated. Meshing the well bore inherrently interpolates between survey stations.
  - More coming soon!

### Tech

welleng uses a number of open source projects to work properly:

* [trimesh] - awesome library for loading and using triangular meshes
* [numpy] - the fundamental package for scientific computing with Python
* [scipy] - a Python-based ecosystem of open-source software for mathematics, science, and engineering
* [dillinger] - open source editor used to generate this document

### Installation

[welleng] requires [trimesh], [numpy] and [scipy] to run. Other libraries are optional depending on usage. Other than that, it should be an easy pip install to get up and running with welleng and the minimum dependencies.

```sh
$ pip install welleng
```

### Quick Start

Coming soon, but in the meantime take a look at the [examples].

### Todos

 - Generate a scene of offset wells to enable fast screening of collision risks (e.g. hundreds of wells in seconds)
 - Well trajectory planning - construct your own trajectories using a range of methods (and of course, including some novel ones)
 - More error models
 - WebApp for those that just want answers
 - Viewer - a 3D viewer to quickly visualize the data and calculated results

It's possible to generate data for visualizing well trajectories with [welleng], as can be seen with the scene below, but it can me made more simple and intuitive.

![image](https://user-images.githubusercontent.com/41046859/97724026-b78c2e00-1acc-11eb-845d-1220219843a5.png)
The ISCWSA standard set of well paths for evaluating clearance scenarios, fettled with and imported into [blender].

License
----

LGPL v3

Please note the terms of the license. Although this software endeavors to be accurate, it shouldn't be used as is if for real wells. If you want a production version or wish to develop this software for a particular application, then please get in touch with [jonnycorcutt], but the intent of this library is to assist development.

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [jonnycorcutt]: <mailto:jonnycorcutt@gmail.com>
   [welleng]: <https://github.com/jonnymaserati/welleng>
   [Flexible Collision Library]: <https://github.com/flexible-collision-library/fcl>
   [trimesh]: <https://github.com/mikedh/trimesh>
   [dillinger]: <https://github.com/joemccann/dillinger>
   [numpy]: <https://numpy.org/>
   [scipy]: <https://www.scipy.org/>
   [examples]: <https://github.com/jonnymaserati/welleng/tree/main/examples>
   [blender]: <https://www.blender.org/>
