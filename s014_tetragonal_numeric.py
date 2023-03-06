from pprint import pprint

import mechkit
import numpy as np
import scipy
import sympy as sp
import vofotensors

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=4)


con = mechkit.notation.Converter()


def lambdified_parametrization():
    from vofotensors.abc import alpha1, d1, d3

    return sp.lambdify(
        [alpha1, d1, d3],
        vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"],
    )


fot4 = lambdified_parametrization()(**{"alpha1": 0, "d1": 0.025, "d3": 0.013})
assert np.min(np.linalg.eigh(fot4)[0]) >= 0, "has to be positive semi definite"

# fot4 = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))

angles = np.arange(0, 90 + 45, 45)
candidates = []
for angle in angles:
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle * np.array([1, 0, 0]), degrees=True
    )

    rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)

    indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
    residuum = np.linalg.norm(rotated[indices])
    if residuum <= 1e-5:
        print(f"angle={angle} with fot4=\n {np.round(rotated, 6)}")
        candidates.append(rotated)

print("#######")

for angle, candidate in zip(angles, candidates):
    assert np.linalg.det(candidate) >= 0
    assert np.min(np.linalg.eigh(candidate)[0]) >= 0
    fot2 = np.tensordot(con.to_tensor(fot4), np.eye(3))
    assert np.allclose(fot2, np.eye(3) / 3)
    sym = mechkit.operators.Sym()
    assert np.allclose(sym(candidate), candidate)

    # assert np.allclose(candidate, fot4)
    deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(candidate)))

    print(f"angle={angle} with deviator=\n {np.round(deviator, 6)}")

    assert np.isclose(deviator[0, 2] + deviator[1, 2], -deviator[2, 2])

    # Prepare mathematica
    # tmp = vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"]
    # str(tmp).replace('[', '{').replace(']', '}')

    # tmp = vofotensors.fabric_tensors.N4s_parametric["trigonal"]['alpha1_d3_d9']
    # str(tmp).replace('[', '{').replace(']', '}')

    d1 = deviator[0, 2]
    d3 = deviator[1, 2]
    assert (-1 / 15 <= d1) and (d1 <= 2 / 45)
    assert -1 / 15 <= d1
    assert d3 <= 1 / 210 * (14 - 105 * d1)

print("###################")

# # Generated from
# {
#     "tetragonal": (
#         (
#             (-ONE, ZERO, ZERO),
#             (ZERO, -ONE, ZERO),
#             (ZERO, ZERO, ONE),
#         ),
#         (
#             (ONE, ZERO, ZERO),
#             (ZERO, ZERO, ONE),
#             (ZERO, -ONE, ZERO),
#         ),
#     )
# }
for matrix in {
    "tetragonal": [
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 0, -1), (0, -1, 0)),
        ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
        ((1, 0, 0), (0, 0, 1), (0, -1, 0)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ],
}["tetragonal"]:

    rotation = scipy.spatial.transform.Rotation.from_matrix(matrix)
    # print(rotation.as_rotvec())
    rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation.as_matrix())
    assert np.allclose(rotated, fot4)
    print(rotated)
