import mechkit
import numpy as np
import plotly.graph_objects as go

import mechinterfabric
from mechinterfabric import visualization

con = mechkit.notation.Converter()


def get_data(N4, method, origin=[0, 0, 0], nbr_points=100):

    vectors = visualization.get_unit_vectors(nbr_points=nbr_points)

    if method == "fodf":
        distribution = visualization.DistributionDensityTruncateAfter4(N4=N4)
        scalars = distribution.calc_scalars(vectors)
    elif method == "glyph":
        scalars = visualization.project_vectors_onto_N4_to_scalars(
            N4=N4, vectors=vectors
        )
    else:
        raise mechinterfabric.utils.ExceptionMechinterfabric(
            "Unknown projection method requested"
        )
    scalars_limited = visualization.limit_scaling(scalars, limit_scalar=0.55)

    xyz = visualization.shift_b_origin(xyz=scalars_limited * vectors, origin=origin)

    return xyz, scalars_limited


def add_N4_plotly(
    fig,
    N4,
    origin=[0, 0, 0],
    nbr_points=100,
    options=None,
    method="fodf",
):

    if options is None:
        ############################
        # Define colors

        colorscale = [
            [0, "rgb(1.0, 0.5, 0.05)"],
            [1, "rgb(0.2, 0.7, 0.2)"],
        ]

        options = dict(
            showscale=False,
            colorscale=colorscale,
        )

    xyz, scalars = get_data(N4=N4, origin=origin, nbr_points=nbr_points, method=method)

    # Problem: If all signs of given FODF are homogeneous,
    # the color is neither max nor min of colorscheme but middle
    # Hacky solution: Set one single point to alternative color to
    # get both +1 and -1 as limits of color mpa
    surfacecolor = np.sign(scalars)
    if np.all(surfacecolor == surfacecolor[0, 0]):
        surfacecolor[0, 0] = -surfacecolor[0, 0]

    surface = go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        surfacecolor=surfacecolor,
        **options,
    )

    fig.add_trace(surface)


def plot_stepwise_interpolation_N4_along_x(
    fig, N1, N2, nbr_points=5, scale=1, method=None, nbr_vectors=100
):

    if method is None:
        method = mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation

    weights_N2 = np.linspace(0, 1, nbr_points)
    weights = np.array([1.0 - weights_N2, weights_N2]).T

    origins = np.vstack(
        [
            np.linspace(0, scale, nbr_points),
            np.zeros((nbr_points)),
            np.zeros((nbr_points)),
        ]
    ).T

    for index in range(nbr_points):

        N4s = np.array([con.to_tensor(N4) for N4 in [N1, N2]])
        current_weights = weights[index]
        origin = origins[index]

        N4_av = method(N4s=N4s, weights=current_weights)

        add_N4_plotly(
            fig=fig,
            N4=N4_av,
            origin=origin,
            nbr_points=nbr_vectors,
        )

    return fig