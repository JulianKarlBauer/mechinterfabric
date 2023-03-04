from pprint import pprint

import mechkit
import numpy as np
import scipy.spatial
import sympy as sp
import vofotensors
from vofotensors.abc import alpha1
from vofotensors.abc import rho1

import mechinterfabric

factor_alpha1, factor_rho1 = sp.symbols("factor_alpha1 factor_rho1", real=True)

np.set_printoptions(linewidth=100000)

converter = mechkit.notation.Converter()


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d3, d9

    return sp.lambdify(
        [alpha1, d3, d9],
        vofotensors.fabric_tensors.N4s_parametric["trigonal"]["alpha1_d3_d9"],
    )


################################################

FOT4 = lambdified_parametrization()(**{"alpha1": 0, "d3": 0.0125, "d9": 0.0325})

######
# Rotate

# angle = 52
# rotation_vector = np.array([0, 1, 2])
# rotation = scipy.spatial.transform.Rotation.from_rotvec(
#     angle * rotation_vector, degrees=True
# )
# Q = rotation.as_matrix()
# FOT4_rotated = mechinterfabric.utils.rotate_mandel(FOT4, Q=Q)

FOT4_rotated = mechinterfabric.utils.rotate_fot4_randomly(FOT4)

print(FOT4)
print(FOT4_rotated)


analysis = mechinterfabric.FOT4Analysis(FOT4_rotated)
analysis.get_eigensystem()
decomposition = analysis.FOT4_spectral_decomposition

reconstructed = mechinterfabric.utils.rotate_to_mandel(
    analysis.FOT4.tensor, analysis.eigensystem
)
print(reconstructed)

mins = []
angle_min = 0
angle_max = 180
angle_nbr = 1800
for angle in np.linspace(angle_min, angle_max, angle_nbr):
    rotation = scipy.spatial.transform.Rotation.from_rotvec(
        angle * np.array([1, 0, 0]), degrees=True
    ).as_matrix()
    rotated = mechinterfabric.utils.rotate_to_mandel(reconstructed, Q=rotation)
    difference = FOT4 - rotated
    # print(f"Difference of rotated by {angle} degree is\n{difference}")
    mins.append(np.min(difference))

perfect_index = np.argmin(np.abs(mins))
perfect_angle = perfect_index / angle_nbr * (angle_max - angle_min)


perfect_rotation = scipy.spatial.transform.Rotation.from_rotvec(
    perfect_angle * np.array([1, 0, 0]), degrees=True
).as_matrix()
perfect_rotated = mechinterfabric.utils.rotate_to_mandel(
    reconstructed, Q=perfect_rotation
)
print(f"perfect_rotated = \n{perfect_rotated}")

# Consequence: Trigonal FOT are not rotational symmetric around axis of "transversely isotropic" second order part
# Question: Which additional step for trigonal symmetric FOT4 can be used to obtain the additional rotation around the x-axis?

# trigonal_rotations = [
#     ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
#     ((-1 / 2, -sqrt(3) / 2, 0), (sqrt(3) / 2, -1 / 2, 0), (0, 0, 1)),
#     ((-1 / 2, sqrt(3) / 2, 0), (-sqrt(3) / 2, -1 / 2, 0), (0, 0, 1)),
#     ((1 / 2, -sqrt(3) / 2, 0), (-sqrt(3) / 2, -1 / 2, 0), (0, 0, -1)),
#     ((1 / 2, sqrt(3) / 2, 0), (sqrt(3) / 2, -1 / 2, 0), (0, 0, -1)),
#     ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
# ]

assert np.allclose(reconstructed, FOT4)
