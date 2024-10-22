import mechkit
import numpy as np
from mechkit import operators


con = mechkit.notation.Converter()


def get_unit_vectors(nbr_points=40):
    phi = np.linspace(0.0, 2.0 * np.pi, nbr_points)
    theta = np.linspace(0.0, np.pi, nbr_points)
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones_like(phi), np.cos(theta))
    return np.array([x, y, z])


def get_special_unit_vectors():
    phi = np.linspace(0.0, 2.0 * np.pi, 3)
    theta = np.linspace(0.0, np.pi, 3)
    x = np.outer(np.cos(phi), np.sin(theta))
    y = np.outer(np.sin(phi), np.sin(theta))
    z = np.outer(np.ones_like(phi), np.cos(theta))
    return np.array([x, y, z])


class DistributionDensityTruncateAfter4:
    def __init__(self, N4):
        N4 = con.to_tensor(N4)
        N2 = np.einsum("ijkk->ij", N4)
        self.D2 = operators.dev(N2, order=2)
        self.D4 = operators.dev(N4, order=4)

    def project_on_vectors(self, vectors):
        return self.calc_scalars(vectors=vectors) * vectors

    def calc_scalars(self, vectors):
        n = vectors
        moment2 = np.einsum("i..., j...->ij...", n, n)
        moment4 = np.einsum("i..., j..., k..., l...->ijkl...", n, n, n, n)
        return (
            1.0
            + 15.0 / 2.0 * np.einsum("ij, ij...->...", self.D2, moment2)
            + 315.0 / 8.0 * np.einsum("ijkl, ijkl...->...", self.D4, moment4)
        ) / (4.0 * np.pi)


def limit_scaling(scalars, limit_scalar):
    maximum_scalar = np.max(scalars)
    if np.max(scalars) > limit_scalar:
        scalars = scalars * (limit_scalar / maximum_scalar)
    return scalars


def limit_vectors(vectors, limit_scalar):
    maximum_scalar = np.max(np.linalg.norm(vectors, axis=0))
    if maximum_scalar > limit_scalar:
        vectors = vectors * (limit_scalar / maximum_scalar)
    return vectors


def shift_by_origin(xyz, origin):
    return xyz + np.array(origin)[:, np.newaxis, np.newaxis]


def project_vectors_onto_N4(N4, vectors):
    return np.einsum(
        "ijkl, j..., k..., l...->i...", con.to_tensor(N4), vectors, vectors, vectors
    )


def project_vectors_onto_N4_to_scalars(N4, vectors):
    return np.einsum(
        "ijkl, i..., j..., k..., l...->...",
        con.to_tensor(N4),
        vectors,
        vectors,
        vectors,
        vectors,
    )


def get_approx_FODF_by_N4(N4, vectors):

    distribution = DistributionDensityTruncateAfter4(N4=N4)

    xyz = distribution.project_on_vectors(vectors)

    return xyz


def get_glyph(N4, vectors):

    xyz = project_vectors_onto_N4(N4=N4, vectors=vectors)

    return xyz


def get_quartics(N4, vectors):

    xyz = vectors * project_vectors_onto_N4_to_scalars(N4=N4, vectors=vectors)

    return xyz
