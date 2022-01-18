import numpy as np
from scipy.spatial.transform import Rotation
import itertools


class ExceptionMechinterfabric(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def get_rotation_matrix_into_eigensystem(tensor, verbose=False):
    # Eigenvectors given by eigh are orthonormal, i.e. orthogonal and normalized, but
    # unsorted,
    # not necessarily right-handed
    # differ in sign-conventions, as -v is an eigenvector if v is one

    eigen_values, eigen_vectors = np.linalg.eigh(tensor)

    # Sort it
    if True:
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

    else:

        if verbose:
            print(eigen_values)
        idx = eigen_values.argsort()[::-1]
        if verbose:
            print(idx)
            print(eigen_vectors)

        eigen_values = eigen_values[idx]
        eigen_vectors = eigen_vectors[:, idx]
        if verbose:
            print(eigen_values)
            print(eigen_vectors)

    # Get single vectors
    ev0 = eigen_vectors[:, 0]
    ev1 = eigen_vectors[:, 1]
    ev2 = eigen_vectors[:, 2]

    # Apply convention an signs
    if ev0[0] < 0.0:
        eigen_vectors[:, 0] = -eigen_vectors[:, 0]

    if ev1[1] < 0.0:
        eigen_vectors[:, 1] = -eigen_vectors[:, 1]

    # Make it right-handed
    ev0_cross_ev1 = np.cross(ev0, ev1)
    if not np.allclose(ev0_cross_ev1, ev2):
        if np.allclose(ev0_cross_ev1, -ev2):
            eigen_vectors[:, 2] = -eigen_vectors[:, 2]
        else:
            raise ExceptionMechinterfabric("Check this")

    # Assert no reflection
    if np.linalg.det(eigen_vectors.T) < 0:
        print("det=", np.linalg.det(eigen_vectors.T))
        raise Exception("reflection")

    matrix_into_eigen = eigen_vectors.T

    assert_ortho_normal_right_handed(matrix=matrix_into_eigen)

    return eigen_values, matrix_into_eigen


# def get_rotation_into_eigensystem(tensor, verbose=False):
#     eigen_values, matrix_into_eigen = get_rotation_matrix_into_eigensystem(
#         tensor=tensor, verbose=verbose
#     )
#     return eigen_values, Rotation.from_matrix(matrix_into_eigen)


def assert_ortho_normal_right_handed(matrix):
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
