from pprint import pprint

import mechkit
import numpy as np
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

fot2 = np.tensordot(con.to_tensor(fot4), np.eye(3))

print(fot2)
