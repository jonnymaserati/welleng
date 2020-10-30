from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='welleng',
    version='0.1.1',    
    description='A collection of Well Engineering modules',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jonnymaserati/welleng/archive/v0.1.1-alpha.tar.gz',
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    license='LGPL v3',
    packages=find_packages(),
    install_requires=['pandas',
                      'numpy',
                      'scipy',
                      'trimesh',
                      'math',
                      'openpyxl',                 
                      ],

    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)