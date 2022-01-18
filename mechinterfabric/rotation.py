import numpy as np
import scipy


def average_quaternion(quaternions, weights, verbose=False):
    """
    See https://arc.aiaa.org/doi/abs/10.2514/1.28949
    and https://github.com/christophhagen/averaging-quaternions
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
