#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import averageQuaternions
import pytest


def some_random_orientations(nbr=10):
    randoms = Rotation.random(nbr)
    for item in randoms:
        yield item


def some_random_orientation_pairs(nbr=10):

    for index in range(10):
        yield Rotation.random(2)


@pytest.mark.parametrize("rot_2", some_random_orientations())
def test_average_quaternion(rot_2):
    rot_1 = Rotation.from_rotvec(np.zeros((3)))

    quat_1 = rot_1.as_quat()
    quat_2 = rot_2.as_quat()

    bunch = np.vstack([quat_1, quat_2])

    weights = np.array([1, 1]) / 2

    quat_av_ref = averageQuaternions.weightedAverageQuaternions(Q=bunch, w=weights)

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=bunch, weights=weights
    )

    assert np.allclose(quat_av_ref, quat_av) or np.allclose(quat_av_ref, -quat_av)


@pytest.mark.parametrize("rotations", some_random_orientation_pairs())
def test_average_quaternions_scipy_rotation_mean(rotations):

    weights = np.ones((len(rotations))) / len(rotations)

    bunch = np.vstack([rot.as_quat() for rot in rotations])

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=bunch, weights=weights
    )

    quat_av_ref = rotations.mean(weights=weights).as_quat()

    print(quat_av)
    print(quat_av_ref)

    assert np.allclose(quat_av_ref, quat_av) or np.allclose(quat_av_ref, -quat_av)


if __name__ == "__main__":
    test_average_quaternions_scipy_rotation_mean()
