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
    )
)
