"""
The different setup versions method is shamelessly copied from mikedh's trimesh
Python library - thank you!
"""

import os
import sys

from setuptools import find_packages, setup

# load __version__ without importing anything
version_file = os.path.join(
    os.path.dirname(__file__),
    'welleng/version.py'
)
with open(version_file, 'r') as f:
    # use eval to get a clean string of version from file
    __version__ = eval(f.read().strip().split('=')[-1])

with open("README.md", "r") as f:
    long_description = f.read()

download_url = f'https://github.com/corva-ai/welleng/archive/v{__version__}.tar.gz'

# If you only want to generate surveys and errors, these are all that's
# required

requirements_default = [
    'numpy==1.22.4',
    'pint==0.19.2',
    'PyYAML==6.0',
    "pydantic==1.10.2",
    'requests==2.28.0',
    'scipy==1.8.1',
    'setuptools==62.4.0',
    'vtk==9.1.0'
]

# these can be installed without compiling required
requirements_easy = [
    'magnetic_field_calculator==1.0.2',    # used to get default mag data for survey
    'networkx==2.8.4',
    'openpyxl==3.0.10',
    'tabulate==0.8.9',
    'trimesh==3.12.6',
    'utm==0.7.0',
    'vedo==2022.2.3',
]

# this is the troublesome requirement that needs C dependencies
requirements_all = requirements_easy + ['python-fcl==0.6.1']

# if someone wants to output a requirements file
# `python setup.py --list-all > requirements.txt
if '--list-all' in sys.argv:
    requirements = requirements_all + requirements_default
    print(*requirements, sep="\n")

if '--list-easy' in sys.argv:
    print(*requirements_easy, sep="\n")
    exit()

setup(
    name='corva-welleng',
    version=__version__,
    description='A collection of Well Engineering tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/corva-ai/welleng',
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
    author='Mo Kamyab',
    author_email='m.kamyab@corva.ai',
    license='Apache 2.0',
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.8',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.8.*',
    packages=find_packages(exclude=["tests", "unit_tests"]),
    package_data={
        'welleng': [
            'errors/*.yaml',
            'errors/tool_codes/*.yaml',
            'exchange/*.yaml'
        ]
    },
    install_requires=requirements_default,
    extras_require={
        'easy': requirements_easy,
        'all': requirements_all
    }
)
