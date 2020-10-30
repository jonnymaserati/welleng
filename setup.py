from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='welleng',
    version='0.1.3',    
    description='A collection of Well Engineering modules',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jonnymaserati/welleng',
    download_url='https://github.com/jonnymaserati/welleng/archive/v0.1.3-alpha.tar.gz',
    keywords=['well', 'trajectory', 'wellpath', 'wellbore', 'drilling', 'error', 'separation', 'minimum curvature'],
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    license='LGPL v3',
    packages=find_packages(),
    install_requires=['numpy',
                      'scipy',
                      'trimesh',        
                      ],

    classifiers=[
        'License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)