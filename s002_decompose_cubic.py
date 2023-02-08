from pprint import pprint

import mechkit
import numpy as np
import scipy.spatial
import sympy as sp
import vofotensors
from vofotensors.abc import d1

import mechinterfabric

np.set_printoptions(linewidth=100000)

converter = mechkit.notation.Converter()

parametrization_symbolic = vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"]
parametrization = sp.lambdify([d1], parametrization_symbolic)

limits = [-1 / 15, 2 / 45]

d1 = limits[0]

N4 = parametrization(d1=d1)

######
# Rotate
angle = 52
rotation_vector = np.array([0, 0, 1])
rotation = scipy.spatial.transform.Rotation.from_rotvec(
    angle * rotation_vector, degrees=True
)
Q = rotation.as_matrix()


def rotate(mandel, Q):
    return converter.to_mandel6(
        mechinterfabric.utils.rotate(converter.to_tensor(mandel), Q=Q)
    )


N4_rotated = rotate(N4, Q=Q)


# print(N4)
# print(N4_rotated)

analyser = mechinterfabric.FourthOrderFabricAnalyser()
for d1 in np.linspace(*limits, 5):
    print(f"\n#### d1={d1}")
    FOT4 = parametrization(d1=d1)
    FOT4_rotated = rotate(FOT4, Q=Q)
    analysis = analyser.analyse(FOT4_rotated)
    FOT4_reconstructed = converter.to_mandel6(
        mechinterfabric.utils.rotate(analysis.FOT4_tensor, analysis.eigensystem)
    )
    assert np.allclose(FOT4, FOT4_reconstructed)
