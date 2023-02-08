from pprint import pprint
import sympy as sp
import numpy as np
import mechinterfabric
from mechinterfabric.abc import *
from mechinterfabric import symbolic


def myprint(dev, eigenvalues):
    print(dev.__name__, "with multiplicity = ", list(eigenvalues.values()))
    print(eigenvalues)


deviators = mechinterfabric.deviators.deviators
for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(dev())
        eigenvalues = deviator.eigenvals()
        myprint(dev, eigenvalues)


######################################################
# Rotate and inspect again
# Problem: Invariant eigenvalues change due to rotation.
# This is likely a problem of the identification of multiplicity of eigenvalues within the symbolic package

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
        # eigenvalues = {
        #     sp.trigsimp(key): val for key, val in deviator.eigenvals().items()
        # }
        myprint(dev, eigenvalues)
