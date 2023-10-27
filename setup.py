
import os
from setuptools import setup

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name = "InfEst",
    version = "0.0.0",
    author = "Bryan C Daniels",
    author_email = "bryan.daniels.1@asu.edu",
    description = ("A python package to estimate information measures from data."),
    license = "MIT license",
    keywords = "",
    url = "https://github.com/Collective-Logic-Lab/InfEst",
    packages=['InfEst',],
    install_requires=[
        'scipy',
        'numpy',
    ],
    long_description=read('README.md'),
    classifiers=[
        "",
    ],
)
