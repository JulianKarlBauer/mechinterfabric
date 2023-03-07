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

deviator = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4)))

disturbed = mechinterfabric.utils.rotate_to_mandel(
    deviator,
    Q=mechinterfabric.utils.get_rotation_by_vector(
        vector=np.random.rand() * 360 * np.array([1, 0, 0]), degrees=True
    ),
)


def calc_residuum(angle):
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle * np.array([1, 0, 0]), degrees=True
    )
    rotated = mechinterfabric.utils.rotate_to_mandel(disturbed, Q=rotation)
    indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
    return np.linalg.norm(rotated[indices])


# Brute force try some angles
angles = np.linspace(0, 90, 600)
angles_difference = angles[1] - angles[0]
residuum = np.zeros((len(angles)), dtype=np.float64)
for index, angle in enumerate(angles):
    residuum[index] = calc_residuum(angle=angle)

best_index = np.argmin(residuum)

solution = scipy.optimize.minimize_scalar(
    calc_residuum,
    bounds=(
        angles[best_index] - angles_difference,
        angles[best_index] + angles_difference,
    ),
    method="bounded",
)

optimized_angle = solution.x
additional_rotation = mechinterfabric.utils.get_rotation_by_vector(
    vector=optimized_angle * np.array([1, 0, 0]), degrees=True
)
deviator_optimized = mechinterfabric.utils.rotate_to_mandel(
    disturbed,
    Q=additional_rotation,
)

deviator_alternative = mechinterfabric.utils.rotate_to_mandel(
    deviator_optimized,
    Q=mechinterfabric.utils.get_rotation_by_vector(
        vector=45 * np.array([1, 0, 0]), degrees=True
    ),
)

print(f"deviator=\n{deviator}")
print(f"deviator_optimized=\n{deviator_optimized}")
print(f"deviator_alternative=\n{deviator_alternative}")

assert np.allclose(deviator, deviator_optimized) or np.allclose(
    deviator, deviator_alternative
)

# from matplotlib import pyplot as plt
# plt.plot(angles, residuum, color="black", label="reference")
# plt.legend()
# # plt.show()
