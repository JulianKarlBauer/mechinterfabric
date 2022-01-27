#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation


def test_change_sign_of_columns():
    m = np.random.rand(3, 3)
    m_changed = np.einsum("j, ij->ij", np.array([1, -1, -1]), m)
    assert np.allclose(m_changed, np.array([m[:, 0], -m[:, 1], -m[:, 2]]).T)


def test_change_of_sign_equal_to_rotation():

    variants = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    base = np.eye(3)

    signs_changed = [np.einsum("j, ij->ij", variant, base) for variant in variants]

    rotations = [
        np.eye(3),
        Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix(),
        Rotation.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix(),
        Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix(),
    ]

    for index in range(4):
        assert np.allclose(signs_changed[index], rotations[index])


if __name__ == "__main__":
    test_change_of_sign_equal_to_rotation()
