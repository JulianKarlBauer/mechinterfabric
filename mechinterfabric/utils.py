import itertools

import numpy as np


class ExceptionMechinterfabric(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def get_eigenvalues_and_rotation_matrix_into_eigensystem(tensor):
    # Eigenvectors given by eigh are orthonormal, i.e. orthogonal and normalized, but
    # unsorted,
    # not necessarily right-handed
    # differ in sign-conventions, as -v is an eigenvector if v is one

    eigen_values, eigen_vectors = np.linalg.eigh(tensor)

    # Use arbitrary convention if multiple eigenvalues coincide
    if not len(np.unique(eigen_values)) == 3:
        # Sort by string representation of vectors
        eigen_vector_representation = [str(eigen_vectors[:, i]) for i in range(3)]
        # Sort first by eigen_values then by eigen_vector_representation
        idx = np.lexsort((eigen_vector_representation, eigen_values))[::-1]
    else:
        idx = eigen_values.argsort()[::-1]
    eigen_values = eigen_values[idx]
    eigen_vectors = eigen_vectors[:, idx]

    # Get single vectors
    ev0 = eigen_vectors[:, 0]
    ev1 = eigen_vectors[:, 1]
    ev2 = eigen_vectors[:, 2]

    # Make it right-handed
    ev0_cross_ev1 = np.cross(ev0, ev1)
    if not np.allclose(ev0_cross_ev1, ev2):
        if np.allclose(ev0_cross_ev1, -ev2):
            eigen_vectors[:, 2] = -eigen_vectors[:, 2]
        else:
            raise ExceptionMechinterfabric("Check this")

    matrix_into_eigen = eigen_vectors

    assert_orthonormal_right_handed_rotation(matrix=matrix_into_eigen)

    return eigen_values, matrix_into_eigen


def assert_orthonormal_right_handed_rotation(matrix):
    vx = matrix[:, 0]
    vy = matrix[:, 1]
    vz = matrix[:, 2]

    vecs = [vx, vy, vz]

    # Norm
    for vec in [vx, vy, vz]:
        assert np.isclose(np.linalg.norm(vec), 1)

    # Orthogonal
    for v1, v2 in itertools.combinations(vecs, 2):
        assert np.isclose(np.dot(v1, v2), 0)

    # Right-handed
    assert np.allclose(np.cross(vx, vy), vz)

    # Assert no reflection
    if np.linalg.det(matrix) < 0:
        print("det=", np.linalg.det(matrix))
        raise Exception("reflection")


def rotate(tensor, Q):
    return np.einsum("...mi, ...nj, ...ok, ...pl, ...mnop->...ijkl", Q, Q, Q, Q, tensor)
