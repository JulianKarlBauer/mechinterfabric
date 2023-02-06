from pprint import pprint
import sympy as sp
import numpy as np
import mechkit
import mechinterfabric
from mechinterfabric.abc import *

deviators = mechinterfabric.deviators.deviators[0:1]
for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(dev())
        eigenvalues = deviator.eigenvals()
        print(dev.__name__, eigenvalues.values())


alpha = sp.symbols("alpha", real=True)

Q = np.array(
    (
        (sp.cos(alpha), -sp.sin(alpha), z),
        (sp.sin(alpha), sp.cos(alpha), z),
        (z, z, one),
    ),
)

con = mechkit.notation.ConverterSymbolic()


def mandel(tensor):
    return con.to_mandel6(to_numpy(tensor))


def tensorr(mandel6):
    return con.to_tensor(to_numpy(mandel6))


def to_numpy(inp):
    return np.array(inp.tolist(), dtype=sp.Symbol)


def actively_rotate_tensor(tensor, Q):
    order = len(tensor.shape)
    for i in range(order):
        tensor = np.tensordot(Q, tensor, axes=[[1], [i]])
    return tensor


def actively_rotate_mandel(inp, Q):
    tensor = tensorr(inp)
    rotated = actively_rotate_tensor(tensor=tensor, Q=Q)
    return mandel(rotated)


print("#############")

for dev in deviators:
    if dev.__name__ != "triclinic":
        deviator = sp.Matrix(actively_rotate_mandel(dev(), Q))
        eigenvalues = deviator.eigenvals()
        print(dev.__name__, eigenvalues.values())
