# Installation

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

```
sudo apt-get update
sudo apt-get install libeigen3-dev libccd-dev octomap-tools
```
On a Mac you should be able to install the above with brew and on a Windows machine you'll probably have to build these libraries following the instruction in the link, but it's not too tricky. Once the above are installed, then it should be a simple:

```
pip install welleng[all]
```

For developers, the repository can be cloned and locally installed in your GitHub directory via your preferred Python env (the `dev` branch is usually a version or two ahead of the `main`).

```
git clone https://github.com/jonnymaserati/welleng.git
cd welleng
pip install -e .[all]
```

Make sure you include that `.` in the final line (it's not a typo) as this ensures that any changes to your development version are immediately implemented on save.

### Windows
Detailed instructions for installing [welleng] in a Windows OS can be found in this [post](https://jonnymaserati.github.io/2021/05/11/install-welleng-windows.html).

### Colaboratory
Perhaps the simplest way of getting up and running with [welleng] is with a [colab notebook](https://colab.research.google.com/notebooks/intro.ipynb). The required dependencies can be installed with the following cell:

```
!apt-get install -y xvfb x11-utils libeigen3-dev libccd-dev octomap-tools
!pip install welleng[all]
```