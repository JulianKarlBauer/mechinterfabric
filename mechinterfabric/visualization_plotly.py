import mechkit
import numpy as np
import plotly.graph_objects as go

import mechinterfabric
from mechinterfabric import visualization

con = mechkit.notation.Converter()


def get_data(N4, method, vectors, origin=[0, 0, 0], limit_scalar=0.55):

    if method == "fodf":
        xyz = mechinterfabric.visualization.get_approx_FODF_by_N4(
            N4=N4, vectors=vectors
        )

    elif method == "glyph":
        xyz = mechinterfabric.visualization.get_glyph(N4=N4, vectors=vectors)

    elif method == "quartic":
        xyz = mechinterfabric.visualization.get_quartics(N4=N4, vectors=vectors)

    else:
        raise mechinterfabric.utils.ExceptionMechinterfabric(
            "Unknown projection method requested"
        )

    # raise Exception()
    scalars = np.linalg.norm(xyz, axis=0) * np.sign(
        np.einsum("i..., i...->...", vectors, xyz)
    )

    xyz = visualization.limit_vectors(vectors=xyz, limit_scalar=limit_scalar)

    xyz = visualization.shift_by_origin(xyz=xyz, origin=origin)

    return xyz, scalars


def get_default_options():
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

    return options


def fake_data_for_color(scalars):
    # Problem: If all signs of given FODF are homogeneous,
    # the color is neither max nor min of colorscheme but middle
    # Hacky solution: Set one single point to alternative color to
    # get both +1 and -1 as limits of color mpa
    surfacecolor = np.sign(scalars)
    if np.all(surfacecolor == surfacecolor[0, 0]):
        surfacecolor[0, 0] = -surfacecolor[0, 0]

    return surfacecolor


def add_N4_plotly(
    fig,
    N4,
    origin=[0, 0, 0],
    nbr_points=100,
    options=None,
    method="fodf",
    limit_scalar=0.55,
    vectors=None,
):

    if options is None:
        options = get_default_options()

    if vectors is None:
        vectors = visualization.get_unit_vectors(nbr_points=nbr_points)

    xyz, scalars = get_data(
        N4=N4,
        origin=origin,
        method=method,
        vectors=vectors,
        limit_scalar=limit_scalar,
    )

    surfacecolor = fake_data_for_color(scalars)

    surface = go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        surfacecolor=surfacecolor,
        **options,
    )

    fig.add_trace(surface)


def plot_stepwise_interpolation_N4_along_x(
    fig,
    N1,
    N2,
    nbr_points=5,
    scale=1,
    method=None,
    nbr_vectors=100,
    limit_scalar=0.55,
    method_visualization="fodf",
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
            limit_scalar=limit_scalar,
            method=method_visualization,
        )

    return fig


def add_pseudo_cylinder(fig, origin, rotation, nbr_points=50, ratio=60, limit=10):

    # limit = 0.5 * ratio

    vectors = mechinterfabric.visualization.get_unit_vectors(nbr_points=nbr_points)
    vectors[0, ...] = vectors[0, ...] * ratio
    vectors[0, ...] = np.clip(vectors[0, ...], -limit, limit)

    xyz = np.einsum("ji, i...->j...", rotation, vectors)

    xyz = mechinterfabric.visualization.shift_by_origin(xyz=xyz, origin=origin)

    surface = go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        # surfacecolor=surfacecolor,
        showscale=False,
        colorscale=[
            [0, "rgb(0.2, 0.7, 0.2)"],
            [1, "rgb(0.2, 0.7, 0.2)"],
        ],
        # **mechinterfabric.visualization_plotly.get_default_options(),
    )

    fig.add_trace(surface)
