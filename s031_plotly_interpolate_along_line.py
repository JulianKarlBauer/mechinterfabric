import matplotlib.pyplot as plt
import mechkit
import numpy as np
import plotly.graph_objects as go
import sympy as sp
import vofotensors
from plotly.subplots import make_subplots

import mechinterfabric
from mechinterfabric import visualization_plotly
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


fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(showticklabels=False, visible=False),
        zaxis=dict(showticklabels=False, visible=False),
    )
)


############################
# N4


def lambdified_parametrization_triclinic():
    return sp.lambdify(
        [la1, la2, d1, d2, d3, d4, d5, d6, d7, d8, d9],
        vofotensors.fabric_tensors.N4s_parametric["triclinic"][
            "la1_la2_d1_d2_d3_d4_d5_d6_d7_d8_d9"
        ],
    )


first = lambdified_parametrization_triclinic()(
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
)

second = lambdified_parametrization_triclinic()(
    la1=0.7,
    la2=0.2,
    d1=0.03,
    d2=0.023,
    d3=-0.01,
    d4=0,
    d5=0,
    d6=0,
    d7=0,
    d8=0,
    d9=0,
)

# converter = mechkit.notation.ExplicitConverter()
# N4s = converter.convert(
#     source="mandel6",
#     target="tensor",
#     quantity="stiffness",
#     inp=np.array([first, second]),
# )

# N4 = mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation(
#     N4s=N4s, weights=np.array([0.5, 0.5])
# )

for seed in [5, 10]:
    np.random.seed(5)
    first = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(8, 3), order=4
    )
    second = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(5, 3), order=4
    )

    visualization_plotly.plot_stepwise_interpolation_N4_along_x(
        fig=fig,
        N1=first,
        N2=second,
        nbr_points=5,
        scale=2.5,
        method=None,
        nbr_vectors=300,
    )

    fig.show()