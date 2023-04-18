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


con = mechkit.notation.Converter()


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


alphas = False
kwargs = {
    "la1": 1 / 2,
    "la2": 1 / 4,
    "d1": 0.05,
    "d2": 0.033,
    "d3": 0.011,
}  # Assert decrease in d_i for increasing i
# alphas = True
# kwargs = {
#     "alpha1": 0,
#     "alpha3": -1 / 6,
#     "d1": -0.06,
#     "d2": -0.015,
#     "d3": 0.02,
# }


if alphas:
    (
        kwargs["la1"],
        kwargs["la2"],
    ) = mechinterfabric.utils.to_lambda1_lambda2(kwargs["alpha1"], kwargs["alpha3"])
    del kwargs["alpha1"]
    del kwargs["alpha3"]

fot4 = lambdified_parametrization_triclinic()(
    d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, **kwargs
)
deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))
print(f"deviator=\n{deviator}")
print()

rotation = mechinterfabric.utils.get_random_rotation()


# Apply random rotation
fot4_rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)
deviator_rotated = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4_rotated)))

spectral_decomposition = mechinterfabric.decompositions.SpectralDecompositionDeviator4(
    deviator_rotated
)

fot2_rotated = np.tensordot(con.to_tensor(fot4_rotated), np.eye(3))
decomposition_fot2_rotated = mechinterfabric.decompositions.SpectralDecompositionFOT2(
    mechinterfabric.core.FiberOrientationTensor2(fot2_rotated)
)
decomposition_fot2_rotated.get_symmetry()
eigensystem = decomposition_fot2_rotated.FOT2_rotation


fot4_in_eigensystem_fot2 = mechinterfabric.utils.rotate_to_mandel(
    fot4_rotated, eigensystem
)
deviator_in_eigensystem_fot2 = mechinterfabric.utils.rotate_to_mandel(
    deviator_rotated, eigensystem
)


def _calc_norm_upper_right_quadrant(mandel):
    indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
    return np.linalg.norm(mandel[indices])


d_i = {
    1: [],
    2: [],
    3: [],
}
norm = []
indices = {
    1: np.s_[0, 1],
    2: np.s_[0, 2],
    3: np.s_[1, 2],
}
angles = np.linspace(0, 180, 180 + 1)
for angle in angles:
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        angle * np.array([1, 0, 0]), degrees=True
    )
    rotated = mechinterfabric.utils.rotate_to_mandel(
        deviator_in_eigensystem_fot2, rotation
    )
    for i in [1, 2, 3]:
        d_i[i].append(rotated[indices[i]])
    norm.append(_calc_norm_upper_right_quadrant(rotated))

for i in [1, 2, 3]:
    plt.plot(angles, d_i[i], label=f"d_{i}")
plt.plot(angles, norm, label=f"Norm upper right quadrant")

plt.gca().set_prop_cycle(None)
for i in [1, 2, 3]:
    plt.plot(
        angles,
        np.ones_like(angles) * kwargs[f"d{i}"],
        label=f"given d_{i}",
        linestyle="dashed",
    )


plt.legend()
