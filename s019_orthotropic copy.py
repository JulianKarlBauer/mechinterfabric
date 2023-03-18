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


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


kwargs = {
    "la1": 1 / 3,
    "la2": 1 / 3,
    "d1": 0.05,
    "d2": 0.033,
    "d3": 0.011,
}  # Assert decrease in d_i for increasing i
fot4 = lambdified_parametrization_triclinic()(
    d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, **kwargs
)
deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))
print(f"deviator=\n{deviator}")
# print(f"fot4=\n{fot4}")
print()

rotation = mechinterfabric.utils.get_random_rotation()


# Apply random rotation
fot4_rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)
deviator_rotated = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4_rotated)))

spectral_decomposition = mechinterfabric.decompositions.SpectralDecompositionDeviator4(
    deviator_rotated
)

for value, vector in zip(
    spectral_decomposition.eigen_values, spectral_decomposition.eigen_vectors.T
):
    if not np.isclose(value, 0.0):
        tensor = con.to_tensor(vector)
        vals, vecs = np.linalg.eigh(tensor)

        one_over_sqrt_two = 1 / np.sqrt(2)
        if not np.allclose(vals, [-one_over_sqrt_two, 0, one_over_sqrt_two]):
            (
                vals_sorted,
                eigensystem,
            ) = mechinterfabric.utils.sort_eigen_values_and_vectors(
                eigen_values=np.abs(vals), eigen_vectors=vecs
            )
            print(f"value={value}")
            print(f"vals={vals}")
            print(f"vals_sorted={vals_sorted}")
            print(f"vecs=\n{np.round(vecs,4)}")
            print(f"eigensystem=\n{np.round(eigensystem,4)}")
            back = mechinterfabric.utils.rotate_to_mandel(
                deviator_rotated, Q=eigensystem
            )
            print(f"back=\n{np.round(back,4)}")

            tol = 1e-2
            assert np.allclose(back, deviator, atol=tol)


# t = mechkit.operators.Sym()(np.random.rand(3, 3))
# Q = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
# np.einsum("ij,kl,jl->ik", Q, Q, t)
