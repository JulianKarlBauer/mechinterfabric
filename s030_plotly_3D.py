import matplotlib.pyplot as plt
import mechkit
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import mechinterfabric

# Equation of ring cyclide
# see https://en.wikipedia.org/wiki/Dupin_cyclide

a, b, d = 1.32, 1.0, 0.8
c = a**2 - b**2
u, v = np.mgrid[0 : 2 * np.pi : 100j, 0 : 2 * np.pi : 100j]
x = (d * (c - a * np.cos(u) * np.cos(v)) + b**2 * np.cos(u)) / (
    a - c * np.cos(u) * np.cos(v)
)
y = b * np.sin(u) * (a - d * np.cos(v)) / (a - c * np.cos(u) * np.cos(v))
z = b * np.sin(v) * (c * np.cos(u) - d) / (a - c * np.cos(u) * np.cos(v))

fig = make_subplots(
    rows=1,
    cols=1,
    specs=[[{"is_3d": True}]],
    subplot_titles=[
        "Subplot title",
    ],
)

cmap = plt.get_cmap("tab10")
# colorscale = [[0, "rgb" + str(cmap(1)[0:3])], [1, "rgb" + str(cmap(2)[0:3])]]
colorscale = [
    [0, "rgb(1.0, 0.5, 0.05)"],
    [1, "rgb(0.17, 0.63, 0.17)"],
]


offset = 1
options = dict(
    surfacecolor=np.ones_like(x),
    showscale=False,
    colorscale=colorscale,
)


# fig.add_trace(
#     go.Surface(
#         x=x,
#         y=y,
#         z=z,
#         **options,
#     ),
#     1,
#     1,
# )

############################
# N4


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

    maximum_scalar = np.max(scalars)
    limit_scalar = 0.55
    if np.max(scalars) > limit_scalar:
        scalars = scalars * (limit_scalar / maximum_scalar)

    xyz = scalars * vectors + np.array(origin)[:, np.newaxis, np.newaxis]

    return xyz, scalars


N4 = mechkit.fabric_tensors.Basic().N4["iso"]
xyz, scalars = get_data(N4=N4)

fig.add_trace(
    go.Surface(
        x=xyz[0],
        y=xyz[1],
        z=xyz[2],
        **options,
    ),
    1,
    1,
)

fig.add_trace(
    go.Surface(
        x=xyz[0] + offset,
        y=xyz[1],
        z=xyz[2],
        **options,
    ),
    1,
    1,
)

fig.layout.scene.update(
    aspectmode="data",
    # aspectmode="cube",
    # aspectratio=dict(x=1, y=1, z=1),
)
# fig.update_layout(title_text="Title")
# fig.update_layout(scene_aspectmode="cube")
# fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=1, y=1, z=2))
# fig.update_layout(scene_aspectmode="data")


fig.show()
