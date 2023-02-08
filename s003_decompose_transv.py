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

parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric[
    "transv_isotropic"
]["alpha1_rho1"]
parametrization = sp.lambdify([alpha1, rho1], parametrization_symbolic)


def linear_interpolation(start, end, factor):
    return start + factor * (end - start)


alpha_by_factor = linear_interpolation(
    start=sp.sympify("-1/3"),
    end=sp.sympify("2/3"),
    factor=factor_alpha1,
)
rho_by_factor = linear_interpolation(
    start=alpha1 * alpha1 / sp.S(8) - alpha1 / sp.S(42) - sp.sympify("1/ 90"),
    end=alpha1 / sp.S(56) + sp.sympify("1 / 60"),
    factor=factor_rho1,
)

normalized_parametrization_symbolic = parametrization_symbolic.subs(
    {rho1: rho_by_factor}
).subs({alpha1: alpha_by_factor})

normalized_parametrization = sp.lambdify(
    [factor_alpha1, factor_rho1], normalized_parametrization_symbolic
)

points = [
    {"alpha1": -1 / 3, "rho1": 3 / 280, "factor_alpha1": 0, "factor_rho1": 0},
    {"alpha1": -1 / 3, "rho1": 3 / 280, "factor_alpha1": 0, "factor_rho1": 1},
    {"alpha1": 2 / 3, "rho1": 1 / 35, "factor_alpha1": 1, "factor_rho1": 0},
    {"alpha1": 2 / 3, "rho1": 1 / 35, "factor_alpha1": 1, "factor_rho1": 1},
    {
        "alpha1": 0,
        "rho1": 0,
        "factor_alpha1": 1 / 3,
        "factor_rho1": 2 / 5,
    },  # sp.solve(rho_by_factor.subs({alpha1:sp.S(0)}), factor_rho1)
]

for point in points:
    assert np.allclose(
        parametrization(point["alpha1"], point["rho1"]),
        normalized_parametrization(point["factor_alpha1"], point["factor_rho1"]),
    )

################################################

FOT4 = normalized_parametrization(1 / 3, 1 / 2)

######
# Rotate
angle = 52
rotation_vector = np.array([1, 0, 1])
rotation = scipy.spatial.transform.Rotation.from_rotvec(
    angle * rotation_vector, degrees=True
)
Q = rotation.as_matrix()


def rotate(mandel, Q):
    return converter.to_mandel6(
        mechinterfabric.utils.rotate(converter.to_tensor(mandel), Q=Q)
    )


FOT4_rotated = rotate(FOT4, Q=Q)

print(FOT4)
print(FOT4_rotated)


analyser = mechinterfabric.FourthOrderFabricAnalyser()

analysis = analyser.analyse(FOT4_rotated)
FOT4_reconstructed = converter.to_mandel6(
    mechinterfabric.utils.rotate(analysis.FOT4_tensor, analysis.eigensystem)
)
assert np.allclose(FOT4, FOT4_reconstructed)
