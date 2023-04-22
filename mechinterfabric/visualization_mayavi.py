import mechkit
import numpy as np

import mechinterfabric

con = mechkit.notation.Converter()


################################################
# Mayavi


def plot_N_COS_FODF_mayavi(
    fig,
    N4,
    origin=[0, 0, 0],
    nbr_points=100,
):

    vectors = mechinterfabric.visualization.get_unit_vectors(nbr_points=nbr_points)

    distribution = mechinterfabric.visualization.DistributionDensityTruncateAfter4(
        N4=N4
    )

    scalars = distribution.calc_scalars(vectors)

    scalars = mechinterfabric.visualization.limit_scaling(scalars, limit_scalar=0.55)

    values = scalars * vectors + np.array(origin)[:, np.newaxis, np.newaxis]

    from mayavi import mlab

    added = mlab.mesh(
        *values,
        scalars=-np.sign(scalars),
        opacity=1,
        figure=fig,
        # colormap="jet",
    )  # blue is positive, red is negative
    return added


def plot_stepwise_interpolation_N4_along_x_mayavi(
    fig,
    N1,
    N2,
    nbr_points=5,
    nbr_vectors=100,
    scale=1,
    method=None,
    origin_y=0,
    origin_z=0,
):

    if method is None:
        method = mechinterfabric.interpolation.interpolate_N4_decomp

    # offset = 0.3
    # ax.set_xlim((0 - offset) * scale, (1 + offset) * scale)
    # ax.set_ylim((-0.5 - offset) * scale, (0.5 + offset) * scale)
    # ax.set_zlim((-0.5 - offset) * scale, (0.5 + offset) * scale)

    weights_N2 = np.linspace(0, 1, nbr_points)
    weights = np.array([1.0 - weights_N2, weights_N2]).T

    origins = np.vstack(
        [
            np.linspace(0, scale, nbr_points),
            np.ones((nbr_points)) * origin_y,
            np.ones((nbr_points)) * origin_z,
        ]
    ).T

    bunch = []
    for index in range(nbr_points):

        N4s = np.array([N1, N2])
        current_weights = weights[index]
        origin = origins[index]

        if index == 0:
            N4_av = N1
        elif index == nbr_points - 1:
            N4_av = N2
        else:
            N4_av = method(N4s=N4s, weights=current_weights)

        new = plot_N_COS_FODF_mayavi(
            fig=fig,
            N4=N4_av,
            origin=origin,
            nbr_points=nbr_vectors,
        )
        bunch.append(new)

    return bunch
