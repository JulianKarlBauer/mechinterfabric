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
# N4


def lambdified_parametrization_cubic():
    return sp.lambdify(
        [d1],
        vofotensors.fabric_tensors.N4s_parametric["cubic"]["d1"],
    )


first = lambdified_parametrization_cubic()(d1=2 / 45)
second = lambdified_parametrization_cubic()(d1=-1 / 15)


for name, tensor in zip(["first", "second"], [first, second]):
    analysis = mechinterfabric.FOT4Analysis(FOT4=tensor)
    analysis.analyse()

    eigensystem = analysis.eigensystem_rotation.as_rotvec()
    parameters = {key: np.round(value, 9) for key, value in analysis.parameters.items()}
    print(f"{name} = N4({parameters}) \n Eigensystem={eigensystem}\n")


for method_visualization in ["fodf", "quartic", "glyph"]:

    ############################
    # Set figure

    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=0.5, y=1.35, z=0.6),
    )

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"is_3d": True}]],
        subplot_titles=[
            f"title",
        ],
    )
    fig.update_layout(
        scene_aspectmode="data",
        scene_camera=camera,
        title=method_visualization,
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(showticklabels=False, visible=False),
            yaxis=dict(showticklabels=False, visible=False),
            zaxis=dict(showticklabels=False, visible=False),
            camera=dict(projection=dict(type="orthographic")),
        )
    )

    visualization_plotly.plot_stepwise_interpolation_N4_along_x(
        fig=fig,
        N1=first,
        N2=second,
        nbr_points=5,
        scale=1.9,
        method=None,
        nbr_vectors=300,
        limit_scalar=0.22,
        method_visualization=method_visualization,
    )

    fig.show()
