import numpy as np


class ExceptionMechinterfabric(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def get_rotation_matrix_into_eigensystem(tensor):
    # Eigenvectors given by eigh are orthogonal but unsorted and might be left-handed
    eigen_values, eigen_vectors = np.linalg.eigh(tensor)

    # Sort it
    idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # Make it right-handed
    ev0 = eigen_vectors[:, 0]
    ev1 = eigen_vectors[:, 1]
    ev2 = eigen_vectors[:, 2]
    ev0_cross_ev1 = np.cross(ev0, ev1)
    if not np.allclose(ev0_cross_ev1, ev2):
        if np.allclose(ev0_cross_ev1, -ev2):
            eigen_vectors[:, 2] = -eigen_vectors[:, 2]
        else:
            raise ExceptionMechinterfabric("Check this")
    return eigen_values, eigen_vectors.T
