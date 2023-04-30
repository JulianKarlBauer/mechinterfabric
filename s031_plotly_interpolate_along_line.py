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


for seed in [5]:  # [0 , 6, 14]:  # [5, 10]:
    ############################
    # Set figure

    print(f"Seed={seed}")

    fig = make_subplots(
        rows=1,
        cols=1,
        specs=[[{"is_3d": True}]],
        subplot_titles=[
            f"{seed}",
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

    np.random.seed(seed)

    first = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(8, 3), order=4
    )
    second = mechkit.fabric_tensors.first_kind_discrete(
        orientations=np.random.rand(5, 3), order=4
    )

    # first = mechkit.fabric_tensors.first_kind_discrete(
    #     orientations=np.array(
    #         [mechinterfabric.utils.get_random_vector() for i in range(8)]
    #     ),
    #     order=4,
    # )
    # second = mechkit.fabric_tensors.first_kind_discrete(
    #     orientations=np.array(
    #         [mechinterfabric.utils.get_random_vector() for i in range(5)]
    #     ),
    #     order=4,
    # )

    for name, tensor in zip(["first", "second"], [first, second]):
        analysis = mechinterfabric.FOT4Analysis(FOT4=tensor)
        analysis.analyse()

        eigensystem = analysis.eigensystem_rotation.as_rotvec()
        parameters = {
            key: np.round(value, 4) for key, value in analysis.parameters.items()
        }
        print(f"{name} = N4({parameters}) \n Eigensystem={eigensystem}\n")

    visualization_plotly.plot_stepwise_interpolation_N4_along_x(
        fig=fig,
        N1=first,
        N2=second,
        nbr_points=5,
        scale=2.2,
        method=None,
        nbr_vectors=300,
    )

    print("##############\n\n")

    # fig.show()
