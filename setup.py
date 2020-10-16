from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='welleng',
    version='0.1.0',    
    description='A collection of Well Engineering modules',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/jonnymaserati/welleng',
    author='Jonathan Corcutt',
    author_email='jonnycorcutt@gmail.com',
    # license='MIT License',
    packages=find_packages(),
    install_requires=['pandas',
                      'numpy',                     
                      ],

    classifiers=[
        # 'License :: OSI Approved :: MIT License',  
        'Operating System :: OS Independent',        
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)