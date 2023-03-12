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
orthotropic_transforms = (
    [
        ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
    ],
)
