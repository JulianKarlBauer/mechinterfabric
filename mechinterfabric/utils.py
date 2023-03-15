import itertools

import mechkit
import numpy as np
import scipy


class ExceptionMechinterfabric(Exception):
    """Exception wrapping all exceptions of this package"""

    pass


def get_eigenvalues_and_rotation_matrix_into_eigensystem(tensor):
    # Eigenvectors given by eigh are orthonormal, i.e. orthogonal and normalized, but
    # unsorted,
    # not necessarily right-handed
    # differ in sign-conventions, as -v is an eigenvector if v is one

    eigen_values, eigen_vectors = np.linalg.eigh(tensor)

    eigen_values, eigen_vectors = sort_eigen_values_and_vectors(
        eigen_values, eigen_vectors
    )

    assert_orthonormal_right_handed_rotation(matrix=eigen_vectors)

    return eigen_values, eigen_vectors


def sort_eigen_values_and_vectors(eigen_values, eigen_vectors):

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

    return eigen_values, eigen_vectors


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


converter = mechkit.notation.Converter()


def rotate_fot4_randomly(fot4):
    return rotate_to_mandel(fot4, Q=get_random_rotation())


def rotate_to_mandel(mandel, Q):
    if isinstance(Q, np.ndarray) and (Q.shape == (3, 3)):
        return converter.to_mandel6(rotate(converter.to_tensor(mandel), Q=Q))
    elif isinstance(Q, list) or (
        isinstance(Q, np.ndarray) and (Q.shape[-2:] == (3, 3))
    ):
        for transform in Q:
            mandel = converter.to_mandel6(
                rotate(converter.to_tensor(mandel), Q=transform)
            )
        return mandel
    else:
        raise ExceptionMechinterfabric("Do not understand argument Q")


def append_transform(old, new):
    if isinstance(old, list):
        return [*old, new]
    elif isinstance(old, np.ndarray) and (old.shape == (3, 3)):
        np.concatenate([old[np.nnewaxis, :, :], new[np.newaxis, :, :]], axis=0)
    elif isinstance(old, np.ndarray) and (old.shape[-2:] == (3, 3)):
        return np.concatenate([old, new[np.newaxis, :, :]], axis=0)


def get_rotation_by_vector(vector, degrees=False):
    rotation = scipy.spatial.transform.Rotation.from_rotvec(vector, degrees=degrees)
    return rotation.as_matrix()


def get_random_rotation():
    angle = 2 * np.pi * np.random.rand(1)
    rotation_vector = np.array(np.random.rand(3))
    rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
    return get_rotation_by_vector(vector=angle * rotation_vector, degrees=False)


def dev_in_mandel(mandel):
    tensor = converter.to_tensor(mandel)
    return converter.to_mandel6(mechkit.operators.dev(tensor, order=len(tensor.shape)))


def handle_near_zero_negatives(value):
    # Catch problem with very small negative numbers
    if np.isclose(value, 0.0):
        value = 0.0
    return value
