from pprint import pprint

import mechkit
import numpy as np
import sympy as sp

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

con = mechkit.notation.ConverterSymbolic()

deviators = mechinterfabric.deviators.deviators[0:4]

for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(dev())

        eigenvalues = deviator.eigenvals()
        print("\n###########")
        print(dev.__name__)
        print(eigenvalues)

        eigenvectors = deviator.eigenvects()
        for ev in eigenvectors:
            nicer = np.array([row for row in ev[2][0]])
            print(f"Multiplicity={ev[1]}")
            print(con.to_tensor(nicer))
        # print(eigenvectors)


######################################################
# Rotate and inspect again


alpha = sp.symbols("alpha", real=True)

Q = np.array(
    (
        (sp.cos(alpha), -sp.sin(alpha), z),
        (sp.sin(alpha), sp.cos(alpha), z),
        (z, z, one),
    ),
)


print("#############")

for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(symbolic.actively_rotate_mandel(dev(), Q))

        eigenvalues = deviator.eigenvals()
        print("\n###########")
        print(dev.__name__)
        print(eigenvalues)

        eigenvectors = deviator.eigenvects()
        for ev in eigenvectors:
            nicer = np.array([row for row in ev[2][0]])
            print(f"Multiplicity={ev[1]}")
            print(con.to_tensor(nicer))
        # print(eigenvectors)
