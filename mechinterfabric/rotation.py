import numpy as np
import scipy
from scipy.linalg import expm, logm


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


def average_Manton2004(matrices, weights):
    """Implement iterative algorithm Manton2004"""

    tolerance = 1e-4

    # Init
    mean = matrices[0]

    while True:
        # print(mean)

        mean_inverse = np.linalg.inv(mean)

        A = np.zeros((3, 3))
        for index in range(len(weights)):
            weight = weights[index]
            matrix = matrices[index]

            A += weight * logm(mean_inverse @ matrix)

        if np.linalg.norm(A) <= tolerance:
            break

        mean = mean @ expm(A)

    return mean
