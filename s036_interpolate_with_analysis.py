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
        f"title",
    ],
)
fig.update_layout(scene_aspectmode="data")

fig.update_layout(
    scene=dict(
        xaxis=dict(showticklabels=False, visible=False),
        yaxis=dict(showticklabels=False, visible=False),
        zaxis=dict(showticklabels=False, visible=False),
        camera=dict(projection=dict(type="orthographic")),
    )
)


############################
# N4


def lambdified_parametrization_cubic():
    return sp.lambdify(
        [d1],
        vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"],
    )


if True:

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
        d4=0.0,
        d5=0,
        d6=0,
        d7=0,
        d8=0,
        d9=0,
    )
    np.random.seed(0)
    second = mechinterfabric.utils.rotate_to_mandel(
        mandel=first,
        Q=mechinterfabric.utils.get_random_rotation(),
    )
else:
    first = lambdified_parametrization_cubic()(d1=2 / 45)
    second = lambdified_parametrization_cubic()(d1=-1 / 15)


for name, tensor in zip(["first", "second"], [first, second]):
    analysis = mechinterfabric.FOT4Analysis(FOT4=tensor)
    analysis.analyse()

    eigensystem = analysis.eigensystem_rotation.as_rotvec()
    parameters = {key: np.round(value, 9) for key, value in analysis.parameters.items()}
    print(f"{name} = N4({parameters}) \n Eigensystem={eigensystem}\n")

visualization_plotly.plot_stepwise_interpolation_N4_along_x(
    fig=fig,
    N1=first,
    N2=second,
    nbr_points=5,
    scale=1.9,
    method=mechinterfabric.interpolation.interpolate_N4_decomp_unique_rotation_analysis,
    nbr_vectors=300,
    limit_scalar=0.22,
)


# fig.show()
