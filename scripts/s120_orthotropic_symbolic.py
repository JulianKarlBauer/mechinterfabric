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


fot4 = vofotensors.fabric_tensors.N4s_parametric["orthotropic"]["la1_la2_d1_d2_d3"]
print(str(fot4).replace("[", "{").replace("]", "}"))
