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


for seed in [5, 10]:
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

    np.random.seed(seed)
    first = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(8, 3), order=4
    )
    second = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(5, 3), order=4
    )

    for name, tensor in zip(["first", "second"], [first, second]):
        analysis = mechinterfabric.FOT4Analysis(FOT4=tensor)
        analysis.analyse()
        print(f"[name] = N4({analysis.parameters})")

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
