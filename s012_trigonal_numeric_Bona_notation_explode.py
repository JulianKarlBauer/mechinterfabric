from pprint import pprint

import mechkit
import numpy as np
import scipy
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=5)


con = mechkit.notation.Converter()

deviator = mechinterfabric.deviators.deviators[3]
print(deviator.__name__)


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d3, d9

    return sp.lambdify(
        [alpha1, d3, d9],
        vofotensors.fabric_tensors.N4s_parametric["trigonal"]["alpha1_d3_d9"],
    )


# def alpha(i, j):
#     def kronecker(i, j):
#         return int(i == j)
#     return i * kronecker(i, j) + (1 - kronecker(i, j)) * (9 - i - j)


fot4 = lambdified_parametrization()(**{"alpha1": 0, "d3": 0.0125, "d9": 0.0325})
deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))
print(f"deviator=\n{deviator}")

# Convert into Bona notation
rotation_from_x_axis_into_Bona_convention = (
    scipy.spatial.transform.Rotation.from_matrix(
        np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    ).as_matrix()
)

# Adjust convention: Preferred axis of Bona is z-axis, not x-axis
rotation_90_around_initial_z = scipy.spatial.transform.Rotation.from_rotvec(
    np.pi / 2.0 * np.array([0, 0, 1], dtype=np.float64)
).as_matrix()
rotation_90_around_initial_y = scipy.spatial.transform.Rotation.from_rotvec(
    np.pi / 2.0 * np.array([0, 1, 0], dtype=np.float64)
).as_matrix()
# Adjust convention of monoclinic mirror plane being perpendicuar to z-axis in JKB diss and parametrizations
rotation_90_around_initial_x = scipy.spatial.transform.Rotation.from_rotvec(
    np.pi / 2.0 * np.array([1, 0, 0], dtype=np.float64)
).as_matrix()
rotation_from_x_axis_into_Bona_convention = (
    rotation_90_around_initial_x
    @ rotation_90_around_initial_y
    @ rotation_90_around_initial_z
)


deviator_bona = mechinterfabric.utils.rotate_to_mandel(
    deviator, rotation_from_x_axis_into_Bona_convention
)
print(f"deviator_bona=\n{np.round(deviator_bona, 5)}")

nbr_angles = 900
offset = 1
nbr_entries = 4
angles = np.zeros((nbr_angles, nbr_entries + 1), dtype=np.float64)
for index_angle, angle_applied in enumerate(np.linspace(0, 90, nbr_angles)):
    angles[index_angle, 0] = angle_applied

    # Apply rotation
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle_applied * np.array([0, 0, 1]), degrees=True
    )
    deviator_bona_rotated = mechinterfabric.utils.rotate_to_mandel(
        deviator_bona, Q=rotation
    )

    spectral_decomposition = (
        mechinterfabric.decompositions.SpectralDecompositionDeviator4(
            deviator_bona_rotated
        )
    )
    spectral_decomposition.get_symmetry()

    indices_eigenvectors_double_eigenvalues = np.array(
        spectral_decomposition.eigen_values_indices[0:2]
    ).flatten()

    # print(
    #     f"spectral_decomposition.deviator=\n{np.round(spectral_decomposition.deviator, 5)}"
    # )

    for index_counter, index in enumerate(indices_eigenvectors_double_eigenvalues):
        eval = spectral_decomposition.eigen_values[index]
        evec = spectral_decomposition.eigen_vectors[:, index]

        tensor = con.to_tensor(evec)

        B = tensor[:-1, 0:-1]
        b = tensor[:-1, -1]

        # a3 = B[0, 0]
        # a6 = B[0, 1]
        # a4 = b[0]
        # a5 = b[-1]

        a3 = tensor[0, 0]
        a6 = tensor[0, 1]
        a4 = tensor[1, 2]
        a5 = tensor[0, 2]

        eta = np.arccos(a3 / (np.sqrt(a3**2 + a6**2)))
        theta = np.arccos(a4 / (np.sqrt(a4**2 + a5**2)))

        angle = (theta - eta) / 3.0
        angles[index_angle, index_counter + 1] = np.rad2deg(angle)

        rotation = mechinterfabric.utils.get_rotation_by_vector(
            vector=angle * np.array([0, 0, 1]), degrees=False
        )

        # tensor_rotated = np.einsum("ij, kl, jl->ik", rotation, rotation, tensor)
        deviator_reconstructed_rotated = mechinterfabric.utils.rotate_to_mandel(
            spectral_decomposition.deviator, Q=rotation
        )

        ########################

        # if np.allclose(deviator_reconstructed_rotated, deviator, atol=1e-2, rtol=1e-2):

        print(f"\nDouble Eigenvalue {eval} has eigen tensor")
        print(tensor)

        # print(f"B={B}")
        # print(f"b={b}")
        # print(f"a3={a3}\ta6={a6}\ta4={a4}\ta5={a5}")

        print(f"eta={np.rad2deg(eta)}°")
        print(f"theta={np.rad2deg(theta)}°")
        print(f"angle={np.rad2deg(angle)}°")

        print(
            f"Coincidence = {np.allclose(deviator_reconstructed_rotated, deviator_bona)}"
        )
        # difference = deviator_reconstructed_rotated - deviator_bona
        # print(f"difference=\n{np.round(difference, 5)}")
        print(
            f"deviator_reconstructed_rotated=\n{np.round(deviator_reconstructed_rotated, 5)}"
        )

        print()


from matplotlib import pyplot as plt

for index in range(nbr_entries):
    index = offset + index

    plt.plot(angles[:, 0], angles[:, index], label=str(index))

plt.plot(angles[:, 0], -angles[:, 0], color="black", label="reference")
plt.legend()
# plt.show()
