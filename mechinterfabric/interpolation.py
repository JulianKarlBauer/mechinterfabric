import numpy as np
from scipy.spatial.transform import Rotation
import mechinterfabric
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem


def average_N2(N2s, weights):
    assert N2s.shape == (len(weights), 3, 3)

    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )

    quats = np.array(list(map(lambda x: Rotation.from_matrix(x).as_quat(), rotations)))

    N2_av_in_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))

    quat_av = mechinterfabric.rotation.average_quaternion(
        quaternions=quats, weights=weights
    )

    rotation_av = Rotation.from_quat(quat_av).as_matrix()

    N2_av = np.einsum("mi, nj, mn->ij", rotation_av, rotation_av, N2_av_in_eigen)

    return N2_av, N2_av_in_eigen, rotation_av