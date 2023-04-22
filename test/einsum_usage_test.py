#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial.transform import Rotation

import mechinterfabric


def test_change_sign_of_columns():
    m = np.random.rand(3, 3)
    m_changed = np.einsum("j, ij->ij", np.array([1, -1, -1]), m)
    assert np.allclose(m_changed, np.array([m[:, 0], -m[:, 1], -m[:, 2]]).T)


def test_change_of_sign_equal_to_rotation():

    variants = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    base = np.eye(3)

    signs_changed = [np.einsum("j, ij->ij", variant, base) for variant in variants]

    rotations = mechinterfabric.utils.get_orthotropic_sym_rotations()

    for index in range(4):
        assert np.allclose(signs_changed[index], rotations[index])


if __name__ == "__main__":
    test_change_of_sign_equal_to_rotation()
