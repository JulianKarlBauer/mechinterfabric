import itertools

import mechkit
import numpy as np
from scipy.spatial.transform import Rotation

import mechinterfabric
from mechinterfabric import rotation
from mechinterfabric import utils
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem


def interpolate_N2_naive(N2s, weights):
    utils.assert_notation_N2(N2s, weights)

    return np.diag(np.einsum("m, mij->ij", weights, N2s))


def interpolate_N2_decomp(N2s, weights, closest_eigensystems=False):
    utils.assert_notation_N2(N2s, weights)

    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )

    if closest_eigensystems:
        rotations = np.array(get_closest_rotation_matrices(*rotations))

    # Average eigenvalues and
    # cast to tensor second order
    N2_av_in_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))

    # Average with scipy.spatila.transform.Rotation().mean()
    rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

    N2_av = np.einsum("mi, nj, mn->ij", rotation_av, rotation_av, N2_av_in_eigen)

    return N2_av, N2_av_in_eigen, rotation_av


def angle_between_two_rotations(rotmatrix_1, rotmatrix_2):

    quat_1 = Rotation.from_matrix(rotmatrix_1).as_quat()
    quat_2 = Rotation.from_matrix(rotmatrix_2).as_quat()

    scalar = np.einsum("i,i->", quat_1, quat_2)

    angle = np.arccos(2.0 * scalar * scalar - 1.0)

    return angle


def get_closest_rotation_matrices(rotmatrix_1, rotmatrix_2):

    assert rotmatrix_1.shape == rotmatrix_2.shape == (3, 3)

    variants = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    combinations = list(itertools.combinations(list(range(len(variants))), 2))

    pairs_of_matrices = np.array(
        [
            [
                np.einsum("j, ij->ij", variants[index_1], rotmatrix_1),
                np.einsum("j, ij->ij", variants[index_2], rotmatrix_2),
            ]
            for (index_1, index_2) in combinations
        ]
    )

    angles = np.array(
        [
            angle_between_two_rotations(rotmat_1, rotmat_2)
            for (rotmat_1, rotmat_2) in pairs_of_matrices
        ]
    )

    print(angles)

    index_minimal_angle = angles.argsort()[0]

    closest_1, closest_2 = pairs_of_matrices[index_minimal_angle]

    return closest_1, closest_2


def interpolate_N4_naive(N4s, weights):
    utils.assert_notation_N4(N4s, weights)

    return np.einsum("m, mijkl->ijkl", weights, N4s)


def interpolate_N4_decomp_extended_return_values(
    N4s,
    weights,
    closest_eigensystems=False,
    func_interpolation_rotation=rotation.average_scipy_spatial_Rotation,
):
    utils.assert_notation_N4(N4s, weights)

    I2 = mechkit.tensors.Basic().I2

    N2s = np.einsum("mijkl,kl->mij", N4s, I2)

    # Get rotations into eigensystem
    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )
    rotations = np.array(rotations)

    if closest_eigensystems:
        rotations = np.array(get_closest_rotation_matrices(*rotations))

    # Get average rotation
    rotation_av = func_interpolation_rotation(matrices=rotations, weights=weights)

    # Rotate each N4 into it's eigensystem
    N4s_eigen = utils.apply_rotation(rotations=rotations, tensors=N4s)

    # Average components in eigensystems
    N4_av_eigen = np.einsum("m, mijkl->ijkl", weights, N4s_eigen)

    # Rotate back to world COS
    N4_av = utils.apply_rotation(rotations=rotation_av.T, tensors=N4_av_eigen)

    # Check if N4_av[I2] == N2_av
    N4_av_I2_eigen = np.einsum("ijkl,kl->ij", N4_av_eigen, I2)
    N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))
    assert np.allclose(N4_av_I2_eigen, N2_av_eigen)

    return N4_av, N4_av_eigen, rotation_av, N2_av_eigen, N4s_eigen, rotations


def interpolate_N4_decomp(N4s, weights):
    return interpolate_N4_decomp_extended_return_values(N4s, weights)[0]


def interpolate_N4_decomp_unique_rotation_extended_return_values(
    N4s,
    weights,
    validate=True,
    # func_interpolation_rotation=rotation.average_scipy_spatial_Rotation,
    func_interpolation_rotation=rotation.average_Manton2004,
    **kwargs,
):

    utils.assert_notation_N4(N4s, weights)

    (
        rotations,
        eigenvals,
    ) = utils.get_rotation_matrix_into_unique_N4_eigensystem_detailed(N4s=N4s)

    # Rotate each N4 into the unique eigensystem
    N4s_eigen = utils.apply_rotation(rotations=rotations, tensors=N4s)

    # Get average rotation
    rotation_av = func_interpolation_rotation(
        matrices=rotations, weights=weights, **kwargs
    )

    # Average components in eigensystems
    N4_av_eigen = np.einsum("m, mijkl->ijkl", weights, N4s_eigen)

    # Rotate back to world COS
    N4_av = utils.apply_rotation(rotations=rotation_av.T, tensors=N4_av_eigen)

    if validate:
        # Check if N4_av[I2] == N2_av
        N4_av_I2_eigen = np.einsum(
            "ijkl,kl->ij", N4_av_eigen, mechkit.tensors.Basic().I2
        )
        N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))
        assert np.allclose(N4_av_I2_eigen, N2_av_eigen)

    return N4_av, N4_av_eigen, rotation_av, N4s_eigen, rotations


def interpolate_N4_decomp_unique_rotation(N4s, weights):
    return interpolate_N4_decomp_unique_rotation_extended_return_values(N4s, weights)[0]


def interpolate_N4_decomp_unique_rotation_analysis_extended_return_values(
    N4s,
    weights,
    validate=True,
    func_interpolation_rotation=rotation.average_Manton2004,
    **kwargs,
):

    utils.assert_notation_N4(N4s, weights)

    def get_eigensystem(N4):
        analysis = mechinterfabric.FOT4Analysis(N4)
        analysis.get_eigensystem()
        return analysis.eigensystem

    # Get rotations into eigensystem
    rotations = np.array([get_eigensystem(N4) for N4 in N4s])

    # Rotate each N4 into the eigensystem
    N4s_eigen = utils.apply_rotation(rotations=rotations, tensors=N4s)

    # Get average rotation
    rotation_av = func_interpolation_rotation(
        matrices=rotations, weights=weights, **kwargs
    )

    # Average components in eigensystems
    N4_av_eigen = np.einsum("m, mijkl->ijkl", weights, N4s_eigen)

    # Rotate back to world COS
    N4_av = utils.apply_rotation(rotations=rotation_av.T, tensors=N4_av_eigen)

    return N4_av, N4_av_eigen, rotation_av, N4s_eigen, rotations


def interpolate_N4_decomp_unique_rotation_analysis(N4s, weights):
    return interpolate_N4_decomp_unique_rotation_analysis_extended_return_values(
        N4s, weights
    )[0]
