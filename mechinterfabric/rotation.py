import numpy as np
import scipy


def average_quaternion(quaternions, weights):
    """
    See https://arc.aiaa.org/doi/abs/10.2514/1.28949
    and https://github.com/christophhagen/averaging-quaternions
    """
    assert quaternions.shape == (len(quaternions), 4)

    matrix = np.einsum("ij,ik,i->jk", quaternions, quaternions, weights)

    _, ev = scipy.linalg.eigh(matrix, subset_by_index=[3, 3])

    return np.ravel(ev)
