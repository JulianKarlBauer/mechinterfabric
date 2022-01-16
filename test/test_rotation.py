import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
import averageQuaternions


def test_average_quaternion():
    rot_1 = Rotation.from_rotvec(0 * np.random.rand(3))
    rot_2 = Rotation.from_rotvec(np.pi / 4 * np.random.rand(3))

    quat_1 = rot_1.as_quat()
    quat_2 = rot_2.as_quat()

    bunch = np.vstack([quat_1, quat_2])

    weights = np.array([1, 1]) / 2

    quat_av_ref = averageQuaternions.weightedAverageQuaternions(Q=bunch, w=weights)

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=bunch, weights=weights
    )

    assert np.allclose(quat_av_ref, quat_av)


def test_average_quaternions_scipy_rotation_mean():
    rotations = Rotation.from_rotvec(
        np.array([0 * np.random.rand(3), np.pi / 4 * np.random.rand(3)])
    )

    weights = np.ones((len(rotations))) / len(rotations)

    bunch = np.vstack([rot.as_quat() for rot in rotations])

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=bunch, weights=weights
    )

    quat_av_ref = rotations.mean(weights=weights).as_quat()

    print(quat_av)
    print(quat_av_ref)

    assert np.allclose(quat_av, quat_av_ref)

    print("asd")


if __name__ == "__main__":
    test_average_quaternions_scipy_rotation_mean()
