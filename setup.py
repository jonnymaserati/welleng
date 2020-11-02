from setuptools import setup, find_packages

with open("README.md", "r") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    required = f.read().splitlines()

with open("VERSION") as f:
    version = f.read()
download_url = f'https://github.com/jonnymaserati/welleng/archive/v{version}.tar.gz'

setup(
    name='welleng',
    version=version,    
    description='A collection of Well Engineering tools',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jonnymaserati/welleng',
    download_url=download_url,
    keywords=['well', 'trajectory', 'wellpath', 'wellbore', 'drilling', 'error', 'separation', 'minimum curvature', 'iscwsa'],
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    license='LGPL v3',
    packages=find_packages(exclude=["tests"]),
    install_requires=required,
    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)