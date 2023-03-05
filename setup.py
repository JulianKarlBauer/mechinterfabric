import sys
from os import path

import setuptools

description = "Interpolation of fabric and fiber-orientation tensors"

if sys.version_info > (3, 0):
    this_directory = path.abspath(path.dirname(__file__))
    with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = description

setuptools.setup(
    name="mechinterfabric",
    version="0.0.1",
    author="Julian Karl Bauer",
    author_email="JulianKarlBauer@gmx.de",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JulianKarlBauer/mechinterfabric",
    packages=setuptools.find_packages(),
    install_requires=[
        "setuptools",
        "numpy",
        "mechkit",
        "scipy",
        "vofotensors >= 1.0.6",
        "natsort",
        "sympy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
