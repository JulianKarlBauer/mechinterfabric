from pprint import pprint

import mechkit
import numpy as np
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


fot4 = lambdified_parametrization()(**{"alpha1": 0, "d3": 0.0125, "d9": 0.0325})
deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))
print(f"deviator=\n{deviator}")

# Apply random rotation
rotation = mechinterfabric.utils.get_rotation_by_vector(
    vector=51 * np.array([1, 1.5, 1.3]), degrees=True
)
# rotation = mechinterfabric.utils.get_random_rotation()
fot4_rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)
deviator_rotated = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4_rotated)))
# print(f"deviator_rotated=\n{deviator_rotated}")

# Apply transv-iso logic
analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
analysis.get_eigensystem()
fot4_reconstructed = mechinterfabric.utils.rotate_to_mandel(
    analysis.FOT4.tensor, analysis.eigensystem
)
deviator_reconstructed = con.to_mandel6(
    mechkit.operators.dev(con.to_tensor(fot4_reconstructed))
)
# print(f"deviator_reconstructed=\n{deviator_reconstructed}")


spectral_decomposition = mechinterfabric.decompositions.SpectralDecompositionDeviator4(
    mechinterfabric.core.FiberOrientationTensor4(FOT=fot4_reconstructed)
)
spectral_decomposition.get_symmetry()

indices_eigenvectors_double_eigenvalues = np.array(
    spectral_decomposition.eigen_values_indices[0:2]
).flatten()


for index in indices_eigenvectors_double_eigenvalues:
    eval = spectral_decomposition.eigen_values[index]
    evec = spectral_decomposition.eigen_vectors[:, index]

    tensor = con.to_tensor(evec)

    # B = tensor[1:, 1:]
    # b = tensor[0, 1:]

    # a3 = B[0, 0]
    # a6 = B[0, 1]
    # a4 = b[0]
    # a5 = b[-1]

    a3 = tensor[2, 2]
    a6 = tensor[1, 2]
    a4 = tensor[0, 1]
    a5 = tensor[0, 2]

    eta = np.arccos(a3 / (np.sqrt(a3**2 + a6**2)))
    theta = np.arccos(a4 / (np.sqrt(a4**2 + a5**2)))

    angle = (theta - eta) / 3.0

    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle * np.array([1, 0, 0]), degrees=False
    )

    # tensor_rotated = np.einsum("ij, kl, jl->ik", rotation, rotation, tensor)
    deviator_reconstructed_rotated = mechinterfabric.utils.rotate_to_mandel(
        spectral_decomposition.deviator, Q=rotation
    )

    # index_pos_d9 = np.s_[1, 5]
    # if deviator_reconstructed_rotated[index_pos_d9] <= 0:
    #     print("######################## \tinverted\t########################")
    #     rotation_changing_sign = np.array(
    #         [[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64
    #     )
    #     deviator_reconstructed_rotated = mechinterfabric.utils.rotate_to_mandel(
    #         spectral_decomposition.deviator, Q=rotation_changing_sign
    #     )

    ########################

    # if np.allclose(deviator_reconstructed_rotated, deviator, atol=1e-2, rtol=1e-2):

    # print(f"\nDouble Eigenvalue {eval} has eigen tensor")
    # print(tensor)
    # print(f"tensor_rotated = {tensor_rotated}")

    # print(f"B={B}")
    # print(f"b={b}")
    print(f"a3={a3}\ta6={a6}\ta4={a4}\ta5={a5}")

    print(f"eta={np.rad2deg(eta)}°")
    print(f"theta={np.rad2deg(theta)}°")
    print(f"angle={np.rad2deg(angle)}°")

    print(f"Coincidence = {np.allclose(deviator_reconstructed_rotated, deviator)}")
    difference = deviator_reconstructed_rotated - deviator
    print()

    # print(f"difference=\n{np.round(difference, 5)}")
