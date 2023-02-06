from pprint import pprint
import sympy as sp
import numpy as np
import mechkit
import mechinterfabric
from mechinterfabric.abc import *

deviators = mechinterfabric.deviators.deviators
for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(dev())
        eigenvalues = deviator.eigenvals()
        eigenvectors = deviator.eigenvects()
        print("\n###########")
        print(dev.__name__)
        print(eigenvalues)
        print(eigenvectors)
