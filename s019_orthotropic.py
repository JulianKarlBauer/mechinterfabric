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


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


kwargs = {"la1": 1 / 3, "la2": 1 / 3, "d1": 0.05, "d2": 0.033, "d3": 0.011}
fot4 = lambdified_parametrization_triclinic()(
    d4=0, d5=0, d6=0, d7=0, d8=0, d9=0, **kwargs
)
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
