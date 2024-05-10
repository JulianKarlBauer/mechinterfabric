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

# # Select random rotation
# angle = 52
# rotation = mechinterfabric.utils.get_rotation_by_vector(
#     vector=angle * np.array([1, 0, 0]), degrees=True
# )
rotation = mechinterfabric.utils.get_random_rotation()


# Apply random rotation
fot4_rotated = mechinterfabric.utils.rotate_to_mandel(fot4, Q=rotation)
deviator_rotated = con.to_mandel6(mechkit.operators.dev(con.to_tensor(fot4_rotated)))

# deviator_reconstructed = deviator_rotated
# Apply transv-iso logic
analysis = mechinterfabric.FOT4Analysis(fot4_rotated)
analysis.get_eigensystem()  # Note, this step nowadays always does optimization on trigonal data
fot4_reconstructed = mechinterfabric.utils.rotate_to_mandel(
    analysis.FOT4.tensor, analysis.eigensystem
)
deviator_reconstructed = con.to_mandel6(
    mechkit.operators.dev(con.to_tensor(fot4_reconstructed))
)


angles = np.linspace(0, 60, 180)
angles_difference = angles[1] - angles[0]
residuum = np.zeros((len(angles)), dtype=np.float64)


def calc_residuum(angle):
    rotation = mechinterfabric.utils.get_rotation_by_vector(
        vector=angle * np.array([1, 0, 0]), degrees=True
    )
    rotated = mechinterfabric.utils.rotate_to_mandel(deviator_reconstructed, Q=rotation)
    indices = np.s_[[0, 0, 0, 1, 1, 2, 2], [3, 4, 5, 3, 4, 3, 4]]
    return np.linalg.norm(rotated[indices]), rotated


for index, angle in enumerate(angles):
    residuum[index], rotated = calc_residuum(angle=angle)
    # print(rotated)
    # print()


best_index = np.argmin(residuum)
best_angle = angles[best_index]
additional_rotation = mechinterfabric.utils.get_rotation_by_vector(
    vector=best_angle * np.array([1, 0, 0]), degrees=True
)
deviator_optimized = mechinterfabric.utils.rotate_to_mandel(
    deviator_reconstructed,
    Q=additional_rotation,
)


print(f"deviator_optimized=\n{deviator_optimized}")

# Fine tune
import scipy


res = scipy.optimize.minimize_scalar(
    lambda x: calc_residuum(x)[0],
    bounds=(
        angles[best_index] - angles_difference,
        angles[best_index] + angles_difference,
    ),
    method="bounded",
)

angle_perfect = res.x
additional_rotation = mechinterfabric.utils.get_rotation_by_vector(
    vector=angle_perfect * np.array([1, 0, 0]), degrees=True
)
deviator_perfect = mechinterfabric.utils.rotate_to_mandel(
    deviator_reconstructed,
    Q=additional_rotation,
)

transform = np.eye(3)
index_positive_value = np.s_[1, 5]
if deviator_perfect[index_positive_value] <= 0.0:
    # Apply orthogonal transform which changes signs of specific column of off-orthogonal part
    transform = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
    deviator_perfect = mechinterfabric.utils.rotate_to_mandel(
        deviator_perfect, Q=transform
    )

print(f"deviator_perfect=\n{deviator_perfect}")

assert np.allclose(deviator, deviator_perfect, atol=1e-7, rtol=1e-7)

###############
# Test on transform-level

eigensystem = transform @ additional_rotation @ analysis.eigensystem

rot = scipy.spatial.transform.Rotation
eigensystem_by_rotation = (
    rot.from_matrix(transform)
    * rot.from_matrix(additional_rotation)
    * rot.from_matrix(analysis.eigensystem)
).as_matrix()
assert np.allclose(eigensystem, eigensystem_by_rotation)

fot4_reconstructed = mechinterfabric.utils.rotate_to_mandel(fot4_rotated, Q=eigensystem)

tmp = mechinterfabric.utils.rotate_to_mandel(fot4_rotated, Q=analysis.eigensystem)
tmp = mechinterfabric.utils.rotate_to_mandel(tmp, Q=additional_rotation)
fot4_reconstructed_stepwise = mechinterfabric.utils.rotate_to_mandel(tmp, Q=transform)

print(f"fot4={fot4}")
print(f"fot4_reconstructed={fot4_reconstructed}")
print(f"fot4_reconstructed_stepwise={fot4_reconstructed_stepwise}")
# assert np.allclose(fot4_reconstructed, fot4)
assert np.allclose(fot4_reconstructed_stepwise, fot4)

from matplotlib import pyplot as plt

# plt.plot(angles, np.ones_like(angles) * angle, color="black", label="reference")
plt.plot(angles, residuum)
plt.legend()
# plt.show()
