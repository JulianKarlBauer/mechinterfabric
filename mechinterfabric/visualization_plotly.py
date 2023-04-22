import numpy as np
import plotly.graph_objects as go

import mechinterfabric
from mechinterfabric import visualization


def get_data(
    N4,
    origin=[0, 0, 0],
    nbr_points=100,
):

    vectors = visualization.get_unit_vectors(nbr_points=nbr_points)

    distribution = visualization.DistributionDensityTruncateAfter4(N4=N4)

    scalars = distribution.calc_scalars(vectors)
    scalars_limited = visualization.limit_scaling(scalars, limit_scalar=0.55)

    xyz = visualization.shift_b_origin(xyz=scalars_limited * vectors, origin=origin)

    return xyz, scalars_limited


def add_N4_plotly(fig, N4, origin=[0, 0, 0], nbr_points=100, options={}):

    xyz, scalars = get_data(N4=N4, origin=origin, nbr_points=nbr_points)

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
