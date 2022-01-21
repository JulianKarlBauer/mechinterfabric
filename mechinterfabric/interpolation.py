import numpy as np
from scipy.spatial.transform import Rotation
from mechinterfabric.utils import get_rotation_matrix_into_eigensystem
import mechkit
from mechinterfabric import utils


def interpolate_N2_naive(N2s, weights):
    utils.assert_notation_N2(N2s, weights)

    return np.diag(np.einsum("m, mij->ij", weights, N2s))


def interpolate_N2_decomp(N2s, weights):
    utils.assert_notation_N2(N2s, weights)

    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )

    # Average eigenvalues and
    # cast to tensor second order
    N2_av_in_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))

    # Average with scipy.spatila.transform.Rotation().mean()
    rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

    N2_av = np.einsum("mi, nj, mn->ij", rotation_av, rotation_av, N2_av_in_eigen)

    return N2_av, N2_av_in_eigen, rotation_av


def apply_rotation(rotations, tensors):
    return np.einsum(
        "...mi, ...nj, ...ok, ...pl, ...mnop->...ijkl",
        rotations,
        rotations,
        rotations,
        rotations,
        tensors,
    )


def interpolate_N4_naive(N4s, weights):
    utils.assert_notation_N4(N4s, weights)

    return np.einsum("m, mijkl->ijkl", weights, N4s)


def interpolate_N4_decomp_extended_return_values(N4s, weights):
    utils.assert_notation_N4(N4s, weights)

    I2 = mechkit.tensors.Basic().I2

    N2s = np.einsum("mijkl,kl->mij", N4s, I2)

    # Get rotations into eigensystem
    eigenvals, rotations = zip(
        *[get_rotation_matrix_into_eigensystem(N2) for N2 in N2s]
    )
    rotations = np.array(rotations)

    # Get average rotation
    rotation_av = Rotation.from_matrix(rotations).mean(weights=weights).as_matrix()

    # Rotate each N4 into it's eigensystem
    N4s_eigen = apply_rotation(rotations=rotations, tensors=N4s)

    # Average components in eigensystems
    N4_av_eigen = np.einsum("m, mijkl->ijkl", weights, N4s_eigen)

    # Rotate back to world COS
    N4_av = apply_rotation(rotations=rotation_av.T, tensors=N4_av_eigen)

    # Check if N4_av[I2] == N2_av
    N4_av_I2_eigen = np.einsum("ijkl,kl->ij", N4_av_eigen, I2)
    N2_av_eigen = np.diag(np.einsum("i, ij->j", weights, eigenvals))
    assert np.allclose(N4_av_I2_eigen, N2_av_eigen)

    return N4_av, N4_av_eigen, rotation_av, N2_av_eigen, N4s_eigen, rotations


def interpolate_N4_decomp(N4s, weights):
    return interpolate_N4_decomp_extended_return_values(N4s, weights)[0]
