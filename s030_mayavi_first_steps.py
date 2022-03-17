import numpy as np
la = np.linalg
from mayavi import mlab
#mlab.init_notebook()

EYE3 = np.eye(3)
PHI, THETA = np.mgrid[-np.pi:np.pi:100j, 0.:np.pi:50j]
X = np.cos(PHI) * np.sin(THETA)
Y = np.sin(PHI) * np.sin(THETA)
Z = np.cos(THETA)

P = np.array([X, Y, Z])

#########################
# Helper function for sampling FOTs

def create_random_spherical_distribtution(N):
    """create 4th order FOT from random fibers

    Parameters
    ----------
    N : int
        number of random fibers

    Returns
    -------
    fiber_vecs : np.ndarray(3, N)
        unit fiber vectors
    """

    phi_rand = np.random.uniform(0.0, 2.0 * np.pi, N)
    theta_rand = np.arccos(1.0 - 2.0 * np.random.uniform(0.0, 1.0, N))

    x = np.cos(phi_rand) * np.sin(theta_rand)
    y = np.sin(phi_rand) * np.sin(theta_rand)
    z = np.cos(theta_rand)

    fiber_vecs = np.array([x, y, z])

    return fiber_vecs


def fot_from_fibervecs(fiber_vecs):
    _, N = fiber_vecs.shape
    return np.einsum("iI, jI, kI, lI -> ijkl", *4 * (fiber_vecs,)) / float(N)


#########################
# Helper functions for creating plot data

def project_fourth_order_tensor(A4, x0=np.array([0.0, 0.0, 0.0])):
    """project symmetric 4th order tensor onto the 2-sphere

    Parameters
    ----------
    A4 : np.ndarray(3, 3, 3, 3)
        the fourth order tensor
    x0 : np.ndarray(3,), optional
        translation vector, by default np.array([0.0, 0.0, 0.0])

    Returns
    -------
    np.ndarray(3, N, N/2)
        array of vectors as results of projection
    """
    return (
        np.einsum("ijkl, jIJ, kIJ, lIJ -> iIJ", A4, *3 * (P,))
        + x0[:, np.newaxis, np.newaxis]
    )


def odf_from_fot(N4):
    """use Fourier's series to recover the underlying ODF of a given 2nd order FOT

    Parameters
    ----------
    N4 : np.ndarray(3, 3, 3, 3)
        the 4th-order FOT

    Returns
    -------
    np.ndarray(N, N/2)
        the value of the recovered ODF on the sampled 2 sphere
    """

    N2 = np.einsum("ijkl, kl -> ij", N4, EYE3)
    B2 = N2 - EYE3 / 3.0  # 2nd order deviator

    F2 = np.einsum("iIJ, jIJ -> ijIJ", P, P) - EYE3[:, :, np.newaxis, np.newaxis]

    # deviatoric (direction dependent) part of A4
    B4 = N4 - 1.0 / 7.0 * np.einsum("ij, kl->ijkl", EYE3, N2)
    B4 -= 1.0 / 7.0 * np.einsum("ik, jl -> ijkl", EYE3, N2)
    B4 -= 1.0 / 7.0 * np.einsum("il, jk -> ijkl", EYE3, N2)
    B4 -= 1.0 / 7.0 * np.einsum("jk, il -> ijkl", EYE3, N2)
    B4 -= 1.0 / 7.0 * np.einsum("jl, ik -> ijkl", EYE3, N2)
    B4 -= 1.0 / 7.0 * np.einsum("kl, ij -> ijkl", EYE3, N2)
    B4 += 1.0 / 35.0 * np.einsum("ij,kl->ijkl", EYE3, EYE3)
    B4 += 1.0 / 35.0 * np.einsum("ik,jl->ijkl", EYE3, EYE3)
    B4 += 1.0 / 35.0 * np.einsum("il,jk->ijkl", EYE3, EYE3)

    # isotropic Part of 1st rank dyadic 4th order tensor
    F4 = np.einsum("iIJ, jIJ, kIJ, lIJ -> ijklIJ", *4 * (P,))
    F4 -= 1.0 / 7.0 * np.einsum("ij, kIJ, lIJ -> ijklIJ", EYE3, P, P)
    F4 -= 1.0 / 7.0 * np.einsum("ik, jIJ, lIJ -> ijklIJ", EYE3, P, P)
    F4 -= 1.0 / 7.0 * np.einsum("il, jIJ, kIJ -> ijklIJ", EYE3, P, P)
    F4 -= 1.0 / 7.0 * np.einsum("jk, iIJ, lIJ -> ijklIJ", EYE3, P, P)
    F4 -= 1.0 / 7.0 * np.einsum("jl, iIJ, kIJ -> ijklIJ", EYE3, P, P)
    F4 -= 1.0 / 7.0 * np.einsum("kl, iIJ, jIJ -> ijklIJ", EYE3, P, P)
    F4 += (
        1.0
        / 35.0
        * np.einsum("ij, kl -> ijkl", EYE3, EYE3)[:, :, :, :, np.newaxis, np.newaxis]
    )
    F4 += (
        1.0
        / 35.0
        * np.einsum("ik, jl -> ijkl", EYE3, EYE3)[:, :, :, :, np.newaxis, np.newaxis]
    )
    F4 += (
        1.0
        / 35.0
        * np.einsum("il, jk -> ijkl", EYE3, EYE3)[:, :, :, :, np.newaxis, np.newaxis]
    )

    psi = 0.25 / np.pi + 15.0 / 8.0 / np.pi * np.einsum("ij, ijIJ -> IJ", B2, F2)
    psi += 315.0 / 32.0 / np.pi * np.einsum("ijklIJ, ijkl-> IJ", F4, B4)

    return psi

#########################


fiber_vecs1 = create_random_spherical_distribtution(4)
fiber_vecs1 = EYE3
fiber_vecs2 = create_random_spherical_distribtution(12)

N4_rand1 = fot_from_fibervecs(fiber_vecs1)
N4_rand2 = fot_from_fibervecs(fiber_vecs2)

delta = np.array([2.0, 0.0, 0.0])
proj1 = project_fourth_order_tensor(N4_rand1)
proj2 = project_fourth_order_tensor(N4_rand2, x0=delta)
fig = mlab.figure(figure="FOT")
mlab.mesh(*proj1, colormap="viridis", opacity=.7)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs1.T))), *fiber_vecs1, scale_factor=0.5, color=(0, 0, 1)
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs1.T))),
    *-1.0 * fiber_vecs1,
    scale_factor=0.5,
    color=(1, 0, 0),
)
mlab.mesh(*proj2, colormap="viridis", opacity=.7)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs2.T))) + delta[:, np.newaxis],
    *fiber_vecs2,
    scale_factor=0.5,
    color=(0, 0, 1),
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs2.T))) + delta[:, np.newaxis],
    *-1.0 * fiber_vecs2,
    scale_factor=0.5,
    color=(1, 0, 0),
)
mlab.orientation_axes()
fig.scene._lift()

psi1 = odf_from_fot(N4_rand1)
psi2 = odf_from_fot(N4_rand2)
fig2 = mlab.figure(figure="ODF")
mlab.mesh(
    *psi1 * P, scalars=-np.sign(psi1), opacity=0.7
)  # blue is positive, red is negative
mlab.mesh(
    *psi2 * P + delta[:, np.newaxis, np.newaxis], scalars=-np.sign(psi2), opacity=0.7
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs1.T))), *fiber_vecs1, scale_factor=0.5, color=(0, 0, 1)
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs1.T))),
    *-1.0 * fiber_vecs1,
    scale_factor=0.5,
    color=(1, 0, 0),
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs2.T))) + delta[:, np.newaxis],
    *fiber_vecs2,
    scale_factor=0.5,
    color=(0, 0, 1),
)
mlab.quiver3d(
    *np.zeros((3, len(fiber_vecs2.T))) + delta[:, np.newaxis],
    *-1.0 * fiber_vecs2,
    scale_factor=0.5,
    color=(1, 0, 0),
)
mlab.orientation_axes()
fig2.scene._lift()

mlab.show()








