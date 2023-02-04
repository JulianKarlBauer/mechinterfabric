import vofotensors
from pprint import pprint
from vofotensors.abc import d1
import sympy as sp
import numpy as np
import scipy.spatial
import mechkit
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
result = analyser.analyse(N4)
print(result)
