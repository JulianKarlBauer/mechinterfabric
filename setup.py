import setuptools
from os import path

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

name = "mechinterfabric"

setuptools.setup(
    name=name,
    version="0.0.0,",
    author="Julian Karl Bauer",
    author_email="juliankarlbauer@gmx.de",
    description="Interpolate fabric tensors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/JulianKarlBauer/mechinterfabric/",
    packages=[name],
    package_dir={name: name},
    install_requires=[
        "numpy",
        "scipy",
        # "mechkit>=0.2.6",
        # "natsort",
    ],
    # setup_requires=["pybind11>=2.3", "libcgal-dev", "libeigen3-dev"],
    extras_require={"test": ["pytest", "natsort"]},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
)