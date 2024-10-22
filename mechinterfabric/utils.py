import itertools

import mechkit
import numpy as np
import scipy
from scipy.spatial.transform import Rotation

converter = mechkit.notation.Converter()


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


def chain_rotations(old, new):
    return old @ new


def rotate_fot4_randomly(fot4):
    return rotate_to_mandel(fot4, Q=get_random_rotation())


def rotate_to_mandel(mandel, Q):
    if isinstance(Q, np.ndarray) and (Q.shape == (3, 3)):
        return converter.to_mandel6(rotate(converter.to_tensor(mandel), Q=Q))
    else:
        raise ExceptionMechinterfabric("Do not understand argument Q")


def get_rotation_by_vector(vector, degrees=False):
    rotation = scipy.spatial.transform.Rotation.from_rotvec(vector, degrees=degrees)
    return rotation.as_matrix()


def get_random_rotation():
    angle = 2 * np.pi * np.random.rand(1)
    rotation_vector = np.array(np.random.rand(3))
    rotation_vector = rotation_vector / np.linalg.norm(rotation_vector)
    return get_rotation_by_vector(vector=angle * rotation_vector, degrees=False)


def get_random_vector():
    rotation_vector = np.array(np.random.rand(3))
    rotation = get_random_rotation()
    return np.einsum("ij,  j->i", rotation, rotation_vector)


def dev_in_mandel(mandel):
    tensor = converter.to_tensor(mandel)
    return converter.to_mandel6(mechkit.operators.dev(tensor, order=len(tensor.shape)))


def handle_near_zero_negatives(value):
    # Catch problem with very small negative numbers
    if np.isclose(value, 0.0):
        value = 0.0
    return value


def to_lambda1_lambda2(alpha1, alpha3):
    la1 = 1.0 / 3.0 + alpha1 - alpha3 / 2.0
    la2 = 1.0 / 3.0 - alpha1 / 2.0 - alpha3 / 2.0
    return la1, la2


def to_alpha1_alpha3(la1, la2):
    alpha1 = 2 / 3 * (la1 - la2)
    alpha3 = 4 / 3 * (1 / 2 - la1 / 2 - la2)
    return alpha1, alpha3


##########################################
# Merged


def assert_notation_N2(N2s, weights):
    assert N2s.shape == (len(weights), 3, 3), "N2s must be in tensor notation"


def assert_notation_N4(N4s, weights):
    assert N4s.shape == (len(weights), 3, 3, 3, 3), "N4s must be in tensor notation"


def get_rotation_matrix_into_eigensystem(
    tensor, verbose=False, convention_on_signs=False
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


def apply_rotation(rotations, tensors):
    return np.einsum(
        "...mi, ...nj, ...ok, ...pl, ...mnop->...ijkl",
        rotations,
        rotations,
        rotations,
        rotations,
        tensors,
    )


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


def get_additional_rotation_into_unique_eigensystem(N4_tensor_in_eigen):
    transforms_raw = get_orthotropic_sym_rotations(as_dict=True)

    index_d8 = np.s_[0, 0, 0, 1]
    index_d6 = np.s_[0, 0, 0, 2]

    d8 = N4_tensor_in_eigen[index_d8]
    d6 = N4_tensor_in_eigen[index_d6]

    if (0.0 <= d8) and (0.0 <= d6):
        key = "no flip"
    elif (0.0 <= d8) and (d6 < 0.0):
        key = "flip xy"
    elif (d8 < 0.0) and (0.0 <= d6):
        key = "flip xz"
    elif (d8 < 0.0) and (d6 < 0.0):
        key = "flip yz"
    else:
        raise Exception("Unexpected")

    return transforms_raw[key]


def get_rotation_matrix_into_unique_N4_eigensystem_detailed(N4s):

    assert N4s.shape[-4:] == (3, 3, 3, 3)
    assert len(N4s.shape) > 4, "Wrap single N4 into array"

    I2 = mechkit.tensors.Basic().I2

    N2s = np.einsum("...ijkl,kl->...ij", N4s, I2)

    # Get rotations into eigensystem
    eigenvals, rotations_non_unique = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )
    rotations_non_unique = np.array(rotations_non_unique)

    # Rotate each N4 into one of four possible eigensystem
    N4s_eigen_non_unique = apply_rotation(rotations=rotations_non_unique, tensors=N4s)

    # Get unique eigensystem
    additional_rotation = np.array(
        [
            get_additional_rotation_into_unique_eigensystem(N4_tensor_in_eigen=N4_eigen)
            for N4_eigen in N4s_eigen_non_unique
        ]
    )

    rotations = np.einsum(
        "...ij,...jk->...ik", rotations_non_unique, additional_rotation
    )

    return rotations, eigenvals


def get_rotation_matrix_into_unique_N4_eigensystem(N4s):
    return get_rotation_matrix_into_unique_N4_eigensystem_detailed(N4s)[0]
