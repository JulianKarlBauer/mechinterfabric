import numpy as np
from scipy.spatial.transform import Rotation


def compare_matrix(r1, r2):
    return np.allclose(r1.as_matrix(), r2.as_matrix())


def compare_rot_vec(r1, r2):
    return np.allclose(r1.as_rotvec(), r2.as_rotvec())


def test_consistency():
    rot = Rotation.random()

    from_matrix = Rotation.from_matrix(rot.as_matrix())
    assert compare_matrix(rot, from_matrix)
    assert compare_rot_vec(rot, from_matrix)

    from_rotvec = Rotation.from_rotvec(rot.as_rotvec())
    assert compare_matrix(rot, from_rotvec)
    assert compare_rot_vec(rot, from_rotvec)
