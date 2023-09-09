import numpy as np
import scipy
from scipy.linalg import expm
from scipy.linalg import logm
from scipy.spatial.transform import Rotation

from mechinterfabric import utils


def average_quaternion(quaternions, weights, verbose=False):
    """
    See https://arc.aiaa.org/doi/abs/10.2514/1.28949
    and https://github.com/christophhagen/averaging-quaternions

    Use scipy.spatila.transform.Rotation.mean instead
    """
    assert quaternions.shape == (len(weights), 4)

    matrix = np.einsum("ij,ik,i->jk", quaternions, quaternions, weights)

    if verbose:
        eva, evs = scipy.linalg.eigh(matrix)

        print(eva)
        ev = evs[:, -1]
    else:
        _, ev = scipy.linalg.eigh(matrix, subset_by_index=[3, 3])

    return np.ravel(ev)


def average_scipy_spatial_Rotation(matrices, weights):
    return Rotation.from_matrix(matrices).mean(weights=weights).as_matrix()


def average_quaternion_naive(matrices, weights):
    quaternions = Rotation.from_matrix(matrices).as_quat()

    # Take naive weighted average on normalized quaternions
    mean_quat = np.einsum("m, mi->i", weights, quaternions)
    # Normalize quaternion
    mean_quat = mean_quat / np.linalg.norm(mean_quat)

    return Rotation.from_quat(mean_quat).as_matrix()


def intermediate_rotation_by_weighted_sum_quats_normalized(matrices, weights):
    quats = Rotation.from_matrix(matrices).as_quat()
    quats_weighted_sum = np.einsum("ij, i->j", quats, weights)
    rotation_av = Rotation.from_quat(
        quats_weighted_sum / np.linalg.norm(quats_weighted_sum)
    ).as_matrix()
    return rotation_av


def average_Manton2004(matrices, weights):
    """Implement iterative algorithm Manton2004"""

    # print(matrices)

    tolerance = 1e-4

    # Init
    mean = average_scipy_spatial_Rotation(matrices, weights)
    # mean = average_quaternion_naive(matrices, weights)

    while True:

        mean_inverse = np.linalg.inv(mean)

        A = np.zeros((3, 3))
        for index in range(len(weights)):
            weight = weights[index]
            matrix = matrices[index]

            log = logm(mean_inverse @ matrix)
            if np.iscomplex(log).any():
                message = "Attention, matrices are too far away from each other, returning eye"
                # print(weights)
                # raise utils.ExceptionMechinterfabric(message)

                print(message)
                log = np.real(log)

                # return np.eye(3)

            A += weight * log

        if np.linalg.norm(A) <= tolerance:
            break

        mean = mean @ expm(A)

    return mean
