import itertools
import operator
from pprint import pprint

import matplotlib.pyplot as plt
import mechkit
import numpy as np
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=5)


def evenly_distributed_vectors_on_sphere(nbr_vectors=1000):
    """
    Define nbr_vectors evenly distributed vectors on a sphere
    Using the golden spiral method kindly provided by
    stackoverflow-user "CR Drost"
    https://stackoverflow.com/a/44164075/8935243
    """
    from numpy import pi, cos, sin, arccos, arange

    indices = arange(0, nbr_vectors, dtype=float) + 0.5

    phi = arccos(1 - 2 * indices / nbr_vectors)
    theta = pi * (1 + 5**0.5) * indices

    x, y, z = cos(theta) * sin(phi), sin(theta) * sin(phi), cos(phi)
    orientations = np.column_stack((x, y, z))
    return orientations


con = mechkit.notation.Converter()

cubic_transformations = [
    np.array(tup)
    for tup in [
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (0, 0, -1), (1, 0, 0)),
        ((0, -1, 0), (0, 0, 1), (-1, 0, 0)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
        ((0, 0, -1), (0, -1, 0), (-1, 0, 0)),
        ((0, 0, -1), (0, 1, 0), (1, 0, 0)),
        ((0, 0, -1), (1, 0, 0), (0, -1, 0)),
        ((0, 0, 1), (-1, 0, 0), (0, -1, 0)),
        ((0, 0, 1), (0, -1, 0), (1, 0, 0)),
        ((0, 0, 1), (0, 1, 0), (-1, 0, 0)),
        ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (0, 0, -1), (-1, 0, 0)),
        ((0, 1, 0), (0, 0, 1), (1, 0, 0)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ]
]


def cast_and_normalize(vec):
    vec = np.array(vec)
    return vec / np.linalg.norm(vec)


def make_cubic(fibers):
    workload = list(zip(*list(itertools.product(cubic_transformations, fibers))))
    cubic_fibers = list(map(operator.matmul, *workload))
    return cubic_fibers


def analyse(tensor):
    analysis = mechinterfabric.FOT4Analysis(tensor).analyse()


###############################################

print(f"\n{-1/15}< cubics <{2/45}\n")

experiments = {
    "min": [
        (1, 0, 0),
    ],
    "iso": evenly_distributed_vectors_on_sphere(100),
    "max": [
        (1, 1, 1),
    ],
    **{
        f"circles_{length_x}": [
            (length_x, np.cos(phi), np.sin(phi))
            for phi in np.deg2rad(np.linspace(0, 360, 6, endpoint=False))
        ]
        for length_x in [0.5, 1, 3, 5, 10]
    },
}

# Cast
experiments = {
    key: [cast_and_normalize(fiber) for fiber in fibers]
    for key, fibers in experiments.items()
}

# Make cubic
experiments = {key: make_cubic(fibers=fibers) for key, fibers in experiments.items()}


ingest = {}
for name, fibers in experiments.items():
    fot4 = mechkit.fabric_tensors.first_kind_discrete(orientations=fibers, order=4)
    fot4_mandel = con.to_mandel6(fot4)
    dev4 = mechkit.operators.dev(fot4)
    dev4_mandel = con.to_mandel6(dev4)

    analysis = mechinterfabric.FOT4Analysis(fot4_mandel).analyse()

    ingest[name] = {
        "fot4_mandel": fot4_mandel,
        "dev4_mandel": dev4_mandel,
        "analysis": analysis,
    }

    # print(f"fot4_mandel=\n{fot4_mandel}")
    print(f"\n\n{name}:")
    print(f"dev4_mandel:\n{dev4_mandel}")
    print(f"dev(analysis.reconstructed_dev):\n{analysis.reconstructed_dev}")

for name, fibers in experiments.items():

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    origin = np.zeros(3)
    for fiber in fibers:
        ax.quiver(
            *origin,
            *fiber,
            arrow_length_ratio=0.01,
        )
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_title(name)
