# welleng

welleng aspires to be a collection of useful tools for Wells/Drilling Engineers, kicking off with a range of well trajectory analysis tools.

  - Generate survey listings and interpolation with minimum curvature
  - Calculate well bore uncertainty data (currently utilizing the ISCWSA MWD Rev4 model)
  - Calculate well bore clearance and Separation Factors (SF)
    - standard ISCWSA method
    - new mesh based method

## New Features!

  - Coming soon!

### Tech

welleng uses a number of open source projects to work properly:

* [trimesh] - awesome library for loading and using triangular meshes
* [numpy] - the fundamental package for scientific computing with Python
* [scipy] - a Python-based ecosystem of open-source software for mathematics, science, and engineering
* [dillinger] - open source editor used to generate this document

### Installation

welleng requires [trimesh], [numpy] and [scipy] to run. Other libraries are optional depending on usage. Other than that, it should be an easy pip install to get up and running with welleng.

```sh
$ pip install welleng
```

### Quick Start

Coming soon, but in the meantime take a look at the [examples].

### Todos

 - Well trajectory planning - construct your own trajectories using a range of methods (and of course, including some novel ones)
 - Viewer - a 3D viewer to quickly visualize the data and results
 - More error models

License
----

LGPL v3

**Free Software, Hell Yeah!**

[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [trimesh]: <https://github.com/mikedh/trimesh>
   [dillinger]: <https://github.com/joemccann/dillinger>
   [numpy]: <https://numpy.org/>
   [scipy]: <https://www.scipy.org/>
   [examples]: <https://github.com/jonnymaserati/welleng/tree/main/examples>
