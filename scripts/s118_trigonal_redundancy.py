from pprint import pprint

import mechkit
import numpy as np
import sympy as sp

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

con = mechkit.notation.ConverterSymbolic()

deviator = mechinterfabric.deviators.deviators[3]
print(deviator.__name__)
tensor = deviator()

one = sp.S(1)
zero = sp.S(0)
orthotropic_transforms = [
    ((-one, zero, zero), (zero, -one, zero), (zero, zero, one)),
    ((-one, zero, zero), (zero, one, zero), (zero, zero, -one)),
    ((one, zero, zero), (zero, -one, zero), (zero, zero, -one)),
    ((one, zero, zero), (zero, one, zero), (zero, zero, one)),
]


for transform in orthotropic_transforms:
    rotated = sp.Matrix(symbolic.actively_rotate_mandel(tensor, Q=transform))

    print(rotated.__repr__())
