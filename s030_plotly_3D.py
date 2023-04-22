import matplotlib.pyplot as plt
import mechkit
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import vofotensors
from plotly.subplots import make_subplots

import mechinterfabric
from mechinterfabric.abc import *

############################
# Set figure

fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"is_3d": True}]],
    subplot_titles=[
        "Subplot title",
    ],
)
fig.update_layout(scene_aspectmode="data")


############################
# Define colors

cmap = plt.get_cmap("tab10")
colorscale = [
    [0, "rgb(1.0, 0.5, 0.05)"],
    [1, "rgb(0.2, 0.7, 0.2)"],
]


offset = 1
options = dict(
    showscale=False,
    colorscale=colorscale,
)


############################
# N4


def limit_scaling(scalars, limit_scalar):
    maximum_scalar = np.max(scalars)
    if np.max(scalars) > limit_scalar:
        scalars = scalars * (limit_scalar / maximum_scalar)
    return scalars


def get_data(
    N4,
    origin=[0, 0, 0],
    nbr_points=100,
):

    vectors = mechinterfabric.visualization.get_unit_vectors(nbr_points=nbr_points)

    distribution = mechinterfabric.visualization.DistributionDensityTruncateAfter4(
        N4=N4
    )

    scalars = distribution.calc_scalars(vectors)
    scalars_limited = limit_scaling(scalars, limit_scalar=0.55)

    xyz = scalars_limited * vectors + np.array(origin)[:, np.newaxis, np.newaxis]

    return xyz, scalars_limited


def add_N4_plotly(fig, N4, origin=[0, 0, 0], nbr_points=100, options={}):

    xyz, scalars = get_data(N4=N4, origin=origin, nbr_points=nbr_points)

    surface = go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        surfacecolor=np.sign(scalars),
        **options,
    )

    fig.add_trace(surface)


#################################################


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


add_N4_plotly(
    fig=fig,
    N4=lambdified_parametrization_triclinic()(
        la1=1 / 2,
        la2=1 / 4,
        d1=0.05,
        d2=0.033,
        d3=0.011,
        d4=0,
        d5=0,
        d6=0,
        d7=0,
        d8=0,
        d9=0,
    ),
    origin=[0, 0, 0],
    options=options,
)

add_N4_plotly(
    fig=fig,
    N4=lambdified_parametrization_triclinic()(
        la1=1 / 2,
        la2=1 / 3,
        d1=0.03,
        d2=0.023,
        d3=-0.01,
        d4=0,
        d5=0,
        d6=0,
        d7=0,
        d8=0,
        d9=0,
    ),
    origin=[-0.5, 0, 0],
    options=options,
)

add_N4_plotly(
    fig=fig,
    N4=mechkit.fabric_tensors.Basic().N4["iso"],
    origin=[-1, 0, 0],
    options=options,
)


fig.show()
