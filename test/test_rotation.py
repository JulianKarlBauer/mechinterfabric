import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric


def test_average_quaternion():
    rot_1 = Rotation.from_rotvec(0 * np.random.rand(3))
    rot_2 = Rotation.from_rotvec(np.pi / 4 * np.random.rand(3))

    quat_1 = rot_1.as_quat()
    quat_2 = rot_2.as_quat()

    bunch = np.vstack([quat_1, quat_2])

    weights = np.array([1, 1]) / 2

    quat_av_ref = mechinterfabric.external.weightedAverageQuaternions(
        Q=bunch, w=weights
    )

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=bunch, weights=weights
    )

    assert np.allclose(quat_av_ref, quat_av)
