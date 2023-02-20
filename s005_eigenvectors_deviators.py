from pprint import pprint

import mechkit
import numpy as np
import sympy as sp

import mechinterfabric
from mechinterfabric import symbolic
from mechinterfabric.abc import *

con = mechkit.notation.ConverterSymbolic()

deviators = mechinterfabric.deviators.deviators[0:4]


def inspect(deviator):
    matrix = sp.Matrix(deviator)

    eigenvalues = matrix.eigenvals()
    print("\n###########")
    print(eigenvalues)

    eigenvectors = matrix.eigenvects()
    for ev in eigenvectors:
        print(f"Multiplicity={ev[1]}")
        for eigenspace in ev[2]:
            nicer = np.array([row for row in eigenspace])
            print(con.to_tensor(nicer))
    # print(eigenvectors)


for dev in deviators:
    if dev.__name__ != "triclinic":
        print(dev.__name__)
        inspect(deviator=dev())
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
        print(dev.__name__)
        deviator = sp.Matrix(symbolic.actively_rotate_mandel(dev(), Q))

        inspect(deviator=deviator)
