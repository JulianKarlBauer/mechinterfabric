from pprint import pprint
import sympy as sp
import numpy as np
import mechkit
import mechinterfabric
from mechinterfabric.abc import *

con = mechkit.notation.ConverterSymbolic()

deviators = mechinterfabric.deviators.deviators
for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(dev())
        eigenvalues = deviator.eigenvals()
        eigenvectors = deviator.eigenvects()

        print("\n###########")
        print(dev.__name__)
        print(eigenvalues)
        for ev in eigenvectors:
            nicer = np.array([row for row in ev[2][0]])
            print(f"Multiplicity={ev[1]}")
            print(con.to_tensor(nicer))
        # print(eigenvectors)
