from pprint import pprint

import mechkit
import numpy as np
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=4)


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

rotation = mechinterfabric.utils.get_rotation_by_vector(
    vector=51 * np.array([1, 0, 0]), degrees=True
)
deviator_rotated = mechinterfabric.utils.rotate_to_mandel(deviator, Q=rotation)
print(f"deviator_rotated=\n{deviator_rotated}")

eigenvalues, eigenvectors = np.linalg.eigh(deviator)
print("###########")
print(f"eigenvalues = {eigenvalues}")


def sort_eigensystem(vals, vecs):
    locator = mechinterfabric.decompositions.EigensystemLocatorTransvTetraTrigo(
        spectral_decomposition=None
    )
    (
        vals,
        vecs,
    ) = locator.cast_eigvalsVects_of_eigenvect_to_sign_order_convention_of_reference(
        vals, vecs
    )

    vals_sorted, eigensystem = mechinterfabric.utils.sort_eigen_values_and_vectors(
        eigen_values=vals, eigen_vectors=vecs
    )
    return vals_sorted, eigensystem


for eval, evec in zip(eigenvalues, eigenvectors.T):
    print(f"#######################\nEigenvalue {eval} has eigen tensor")
    tensor = con.to_tensor(evec)
    print(tensor)

    # eigenvalues2, eigenvectors2 = np.linalg.eigh(tensor)
    # for eval2, evec2 in zip(eigenvalues2, eigenvectors2.T):
    #     print(f"\t\t Eigenvalue {eval2} has eigen tensor")
    #     print(f"\t\t{evec2}")

    vals, vecs = np.linalg.eigh(tensor)
    vals_sorted, vecs_sorted = sort_eigensystem(vals, vecs)

    candiate = mechinterfabric.utils.rotate_to_mandel(deviator_rotated, Q=vecs_sorted)
    print(candiate)
