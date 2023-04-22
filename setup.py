import sys
from os import path

import setuptools

description = "Interpolation of fabric and fiber-orientation tensors"


setuptools.setup(
    name="mechinterfabric",
    version="0.0.1",
    author="Julian Karl Bauer",
    author_email="juliankarlbauer@gmx.de",
    description=description,
    long_description=description,
    url="https://github.com/JulianKarlBauer/mechinterfabric",
    packages=setuptools.find_packages(),
    install_requires=[
        "setuptools",
        "numpy",
        "mechkit",
        "scipy",
        "vofotensors >= 1.0.7",
        "natsort",
        "sympy",
        "matplotlib",
        "pandas",
        "natsort",
        "plotly",
    ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
