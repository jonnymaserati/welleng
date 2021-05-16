import os
# import sys
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

requirements_all = [
    'magnetic_field_calculator',
    'matplotlib',
    'networkx',
    'numba',
    'numpy',
    'openpyxl',
    'pandas',
    'python-fcl',
    'PyYAML',
    'scipy',
    'tabulate',
    'trimesh',
    'utm',
    'vedo',
    'vtk',
]

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
        'well engineering',
        'wells',
        'drilling engineering',
        'directional drilling',
        'mwd',
        'survey',
        'covariance'
    ],
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    license='Apache 2.0',
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements_all,
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: Apache Software License',
        # 'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.7',
        'Natural Language :: English',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='==3.7.*',
)
