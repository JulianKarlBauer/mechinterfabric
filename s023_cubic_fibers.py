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

fibers = [
    np.array(tup)
    for tup in [
        (1, 1, 1),
        (1, 0, 0),
    ]
]
workload = list(zip(*list(itertools.product(cubic_transformations, fibers))))
cubic_fibers = list(map(operator.matmul, *workload))


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
for fiber in cubic_fibers:
    ax.quiver(0, 0, 0, *fiber)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])

fot4 = mechkit.fabric_tensors.first_kind_discrete(orientations=cubic_fibers, order=4)
fot4_mandel = con.to_mandel6(fot4)
dev4 = mechkit.operators.dev(fot4)
dev4_mandel = con.to_mandel6(dev4)

print(f"fot4_mandel=\n{fot4_mandel}")
print(f"dev4_mandel=\n{dev4_mandel}")
print(f"{-1/15}< cubics <{2/45}")


# d_i = {
#     1: [],
#     2: [],
#     3: [],
# }
# norm = []
# indices = {
#     1: np.s_[0, 1],
#     2: np.s_[0, 2],
#     3: np.s_[1, 2],
# }
# angles = np.linspace(0, 180, 180 + 1)
# for angle in angles:
#     rotation = mechinterfabric.utils.get_rotation_by_vector(
#         angle * np.array(rotation_axis), degrees=True
#     )
#     rotated = mechinterfabric.utils.rotate_to_mandel(
#         deviator_in_eigensystem_fot2, rotation
#     )
#     for i in [1, 2, 3]:
#         d_i[i].append(rotated[indices[i]])
#     norm.append(_calc_norm_upper_right_quadrant(rotated))

# for i in [1, 2, 3]:
#     plt.plot(angles, d_i[i], label=f"d_{i}")
# plt.plot(angles, norm, label=f"Norm upper right quadrant")

# plt.gca().set_prop_cycle(None)
# for i in [1, 2, 3]:
#     plt.plot(
#         angles,
#         np.ones_like(angles) * kwargs[f"d{i}"],
#         label=f"given d_{i}",
#         linestyle="dashed",
#     )


# plt.legend()
