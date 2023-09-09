import mechkit
import numpy as np
import sympy as sp


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
