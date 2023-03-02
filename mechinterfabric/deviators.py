#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import sympy as sp

from .abc import *


##################
# Fourth-Order Deviators


def copy_upper_triangle(matrix):
    r"""Copy upper triangle to lower triangle, i.e. make symmetric"""
    index_lower_triangle = np.tril_indices(6, -1)
    matrix[index_lower_triangle] = matrix.T[index_lower_triangle]
    return matrix


def cubic():
    return copy_upper_triangle(
        d1
        * np.array(
            [
                [-sp.S(2), sp.S(1), sp.S(1), z, z, z],
                [z, -sp.S(2), sp.S(1), z, z, z],
                [z, z, -sp.S(2), z, z, z],
                [z, z, z, sp.S(2), z, z],
                [z, z, z, z, sp.S(2), z],
                [z, z, z, z, z, sp.S(2)],
            ],
            dtype=object,
        )
    )


def transversely_isotropic():
    return copy_upper_triangle(
        d3
        * np.array(
            [
                [sp.S(8), -sp.S(4), -sp.S(4), z, z, z],
                [z, sp.S(3), sp.S(1), z, z, z],
                [z, z, sp.S(3), z, z, z],
                [z, z, z, sp.S(2), z, z],
                [z, z, z, z, -sp.S(8), z],
                [z, z, z, z, z, -sp.S(8)],
            ],
            dtype=object,
        )
    )


def tetragonal():
    return copy_upper_triangle(
        np.array(
            [
                [-sp.S(2) * d1, d1, d1, z, z, z],
                [z, -(d1 + d3), d3, z, z, z],
                [z, z, -(d1 + d3), z, z, z],
                [z, z, z, sp.S(2) * d3, z, z],
                [z, z, z, z, sp.S(2) * d1, z],
                [z, z, z, z, z, sp.S(2) * d1],
            ],
            dtype=object,
        )
    )


def trigonal():
    return copy_upper_triangle(
        np.array(
            [
                [sp.S(8) * d3, -sp.S(4) * d3, -sp.S(4) * d3, z, z, z],
                [z, sp.S(3) * d3, d3, z, z, sqrt_two * d9],
                [z, z, sp.S(3) * d3, z, z, -sqrt_two * d9],
                [z, z, z, sp.S(2) * d3, sp.S(2) * d9, z],
                [z, z, z, z, -sp.S(2) * sp.S(4) * d3, z],
                [z, z, z, z, z, -sp.S(2) * sp.S(4) * d3],
            ],
            dtype=object,
        )
    )


def orthotropic():
    return copy_upper_triangle(
        np.array(
            [
                [-(d1 + d2), d1, d2, z, z, z],
                [z, -(d1 + d3), d3, z, z, z],
                [z, z, -(d2 + d3), z, z, z],
                [z, z, z, sp.S(2) * d3, z, z],
                [z, z, z, z, sp.S(2) * d2, z],
                [z, z, z, z, z, sp.S(2) * d1],
            ],
            dtype=object,
        )
    )


def monoclinic():
    tmp = -(d8 + d9)
    return orthotropic() + copy_upper_triangle(
        np.array(
            [
                [z, z, z, z, z, sqrt_two * d8],
                [z, z, z, z, z, sqrt_two * d9],
                [z, z, z, z, z, sqrt_two * tmp],
                [z, z, z, z, sp.S(2) * tmp, z],
                [z, z, z, z, z, z],
                [z, z, z, z, z, z],
            ],
            dtype=object,
        )
    )


def triclinic():
    comp_03 = -(d4 + d5)
    comp_14 = -(d6 + d7)
    comp_25 = -(d8 + d9)
    return orthotropic() + copy_upper_triangle(
        np.array(
            [
                [z, z, z, sqrt_two * comp_03, sqrt_two * d6, sqrt_two * d8],
                [z, z, z, sqrt_two * d4, sqrt_two * comp_14, sqrt_two * d9],
                [z, z, z, sqrt_two * d5, sqrt_two * d7, sqrt_two * comp_25],
                [z, z, z, z, sp.S(2) * comp_25, sp.S(2) * comp_14],
                [z, z, z, z, z, sp.S(2) * comp_03],
                [z, z, z, z, z, z],
            ],
            dtype=object,
        )
    )


deviators = [
    cubic,
    transversely_isotropic,
    tetragonal,
    trigonal,
    orthotropic,
    monoclinic,
    triclinic,
]
