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


fot4 = lambdified_parametrization()(**{"alpha1": 0, "d1": 0.025, "d3": 0.015})
assert np.min(np.linalg.eigh(fot4)[0]) >= 0, "has to be positive semi definite"

# fot4 = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))

angles = np.linspace(0, 180, 5)
candidates = []
for angle in angles:
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle * np.array([1, 0, 0]), degrees=True
    )

    rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)

    indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
    residuum = np.linalg.norm(rotated[indices])
    if residuum <= 1e-5:
        print(f"angle={angle}\n {np.round(rotated, 6)}")
        candidates.append(rotated)


for candidate in candidates:
    assert np.linalg.det(candidate) >= 0
    assert np.min(np.linalg.eigh(candidate)[0]) >= 0
    fot2 = np.tensordot(con.to_tensor(fot4), np.eye(3))
    assert np.allclose(fot2, np.eye(3) / 3)
    sym = mechkit.operators.Sym()
    assert np.allclose(sym(candidate), candidate)


print("###################")


for matrix in {
    "tetragonal": [
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        # 45 rotations follow
        ((0, -1, 0), (-1, 0, 0), (0, 0, -1)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (-1, 0, 0), (0, 0, 1)),
        ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    ],
}["tetragonal"]:

    rotation = scipy.spatial.transform.Rotation.from_matrix(matrix)
    # print(rotation.as_rotvec())
    rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation.as_matrix())
    print(rotated)
