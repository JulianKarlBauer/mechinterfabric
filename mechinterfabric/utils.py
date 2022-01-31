import numpy as np
from scipy.spatial.transform import Rotation
import itertools


class ExceptionMechinterfabric(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def assert_notation_N2(N2s, weights):
    assert N2s.shape == (len(weights), 3, 3), "N2s must be in tensor notation"


def assert_notation_N4(N4s, weights):
    assert N4s.shape == (len(weights), 3, 3, 3, 3), "N4s must be in tensor notation"


def get_rotation_matrix_into_eigensystem(
    tensor, verbose=False, convention_on_signs=True
):
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

    if convention_on_signs:
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


def get_orthotropic_sym_rotations(as_dict=False):
    rotations = [
        np.eye(3),
        Rotation.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix(),
        Rotation.from_rotvec(np.pi * np.array([0, 1, 0])).as_matrix(),
        Rotation.from_rotvec(np.pi * np.array([0, 0, 1])).as_matrix(),
    ]
    if not as_dict:
        return rotations
    else:
        labels = [
            "no flip",
            "flip yz",
            "flip xz",
            "flip xy",
        ]
        return {labels[index]: rot for index, rot in enumerate(rotations)}
