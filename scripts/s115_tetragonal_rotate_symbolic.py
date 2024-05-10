from pprint import pprint

import mechkit
import numpy as np
import sympy as sp
import vofotensors

from mechinterfabric import symbolic
from mechinterfabric.abc import *

np.set_printoptions(linewidth=100000, precision=4)


con = mechkit.notation.ConverterSymbolic()

# rotation = mechinterfabric.utils.get_rotation_by_vector(
#     vector=45 * np.array([1, 0, 0]), degrees=True
# )
# np.array([[ 1.    ,  0.    ,  0.    ],
#        [ 0.    ,  0.7071, -0.7071],
#        [ 0.    ,  0.7071,  0.7071]])
# 1/np.sqrt(2)
# 0.7071067811865476


# N4 = vofotensors.fabric_tensors.N4s_parametric["tetragonal"]["alpha1_d1_d3"]
parametrization = sp.Matrix(
    vofotensors.deviators_4.dev4s_parametric["tetragonal"]["d1_d3"]
)
factor = sp.S(1) / sp.sqrt(sp.S(2))
zero = sp.S(0)
rotation = np.array(
    [[sp.S(1), zero, zero], [zero, factor, -factor], [zero, factor, factor]]
)
rotated = sp.simplify(
    sp.Matrix(symbolic.actively_rotate_mandel(parametrization, rotation))
)

print(f"parametrization=\n{parametrization.__repr__()}")
print(f"rotated=\n{rotated.__repr__()}")

print("#########################")

rotation = np.array(
    [
        [sp.S(1), zero, zero],
        [zero, sp.cos(alpha), -sp.sin(alpha)],
        [zero, sp.sin(alpha), sp.cos(alpha)],
    ]
)
rotated = sp.simplify(
    sp.Matrix(symbolic.actively_rotate_mandel(parametrization, rotation))
)


print(f"parametrization=\n{parametrization.__repr__()}")
print(f"rotated=\n{rotated.__repr__()}")
