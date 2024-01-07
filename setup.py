"""
The different setup versions method is shamelessly copied from mikedh's trimesh
Python library - thank you!
"""

import os
import sys
from setuptools import setup, find_packages

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__),
    'welleng/version.py')
with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

with open("README.md", "r") as f:
    long_description = f.read()

# with open("requirements.txt") as f:
#     required = f.read().splitlines()

download_url = f'https://github.com/jonnymaserati/welleng/archive/v{__version__}.tar.gz'

# If you only want to generate surveys and errors, these are all that's
# required
requirements_default = set([
    'numpy',
    'scipy',
    'openpyxl',
    'pandas',
    'pint',
    'pyproj',  # required for getting survey parameters
    'PyYAML',
    'setuptools',
    'vedo',
    'vtk'
])

# these can be installed without compiling required
requirements_easy = set([
    'magnetic_field_calculator',    # used to get default mag data for survey
    'networkx',
    'tabulate',
    'trimesh',
    'utm'
])

# this is the troublesome requirement that needs C dependencies
requirements_all = requirements_easy.union([
    'python-fcl',
])

# if someone wants to output a requirements file
# `python setup.py --list-all > requirements.txt`
if '--list-all' in sys.argv:
    # will not include default requirements (numpy)
    print('\n'.join(requirements_all))
    exit()
elif '--list-easy' in sys.argv:
    # again will not include numpy+setuptools
    print('\n'.join(requirements_easy))
    exit()

# if sys.platform == 'win32':
#     requirements_all.append('python-fcl-win32')
# else:
#     requirements_all.append('python-fcl')

setup(
    name='welleng',
    version=__version__,
    description='A collection of Well Engineering tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jonnymaserati/welleng',
    download_url=download_url,
    keywords=[
        'well',
        'trajectory',
        'wellpath',
        'wellbore',
        'drilling',
        'drill',
        'error',
        'separation',
        'minimum curvature',
        'iscwsa',
        'owsg',
        'well engineering',
        'wells',
        'drilling engineering',
        'directional drilling',
        'mwd',
        'survey',
        'covariance',
        'digitalization',
        'automation',
        'volve',
        'witsml',
    ],
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
    # python_requires='>=3.9',
    packages=find_packages(exclude=["tests"]),
    package_data={
        'welleng': [
            'errors/*.yaml',
            'errors/tool_codes/*.yaml',
            'exchange/*.yaml'
        ]
    },
    install_requires=list(requirements_default),
    extras_require={
        'easy': list(requirements_easy),
        'all': list(requirements_all)
    }
)
